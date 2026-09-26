"""
ShareGPT 训练脚本（严格对齐 train_sharegpt_final.py 的日志/产物输出）

产物严格一致：
- config.json
- training_losses.npy
- training_log.csv
- test_loss_history.json
- results.json
- init_info.json
- best_model/
- final_model/
- tokenizer 文件保存到 out_dir 根目录
- 可选：init_spectra/ final_spectra/

说明：
- 训练方式与 train_sharegpt_final.py 一致：DataLoader + iter 轮转 + grad_accum + eval_interval 评估
- 评估方式与 final 一致：训练中可选 full_test_eval，否则只评估 10 个 batch；训练后一定完整评估计算 mean/std
"""

import os
os.environ['DISABLE_NPU_FUSED_ATTENTION'] = '1'
os.environ['NPU_FUSED_INFER_ATTENTION'] = '0'

import sys
import time
import argparse
import random
import json
import csv
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# 检查 PEFT
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError as e:
    print("错误: 未找到peft库，请先安装: pip install peft")
    print(f"详细错误: {e}")
    PEFT_AVAILABLE = False


# ==================== 工具函数（与 train_sharegpt_final.py 保持一致）====================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def convert_to_json_serializable(obj):
    """转换 numpy 类型"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None or isinstance(obj, (int, float, str)):
        return obj
    else:
        return str(obj)


def compute_gradient_norm(model):
    """计算模型梯度的L2范数"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_auc_intervals(losses):
    """计算不同阶段的AUC"""
    intervals = {
        'auc_0_100': (0, 100),
        'auc_100_500': (100, 500),
        'auc_500_1000': (500, 1000),
        'auc_1000_2500': (1000, 2500),
        'auc_0_500': (0, 500),
        'auc_0_2500': (0, 2500),
    }

    results = {}
    for name, (start, end) in intervals.items():
        if len(losses) >= end:
            results[name] = float(sum(losses[start:end]))
        else:
            results[name] = None

    return results


# ==================== 导入 FDT 模块 ====================
print("\n" + "="*70)
print("🔧 加载 FDT 初始化模块")
print("="*70)

for mod in ['fdt_init', 'measure_alpha']:
    if mod in sys.modules:
        del sys.modules[mod]

FDT_INIT_PATH = '/root/nvme0n1/Noneq_Neural_Network/FDT_Init'

for path in sys.path[:]:
    if 'FDT_Init' in path and path != FDT_INIT_PATH:
        sys.path.remove(path)

if FDT_INIT_PATH in sys.path:
    sys.path.remove(FDT_INIT_PATH)
sys.path.insert(0, FDT_INIT_PATH)

try:
    from fdt_init import apply_fdt_to_lora
    from measure_alpha import analyze_lora_spectra, verify_fdt_initialization
    
    FDT_INIT_AVAILABLE = True
    print("[导入] ✓ FDT 模块加载成功")
    
except ImportError as e:
    print(f"[导入] ✗ 失败: {e}")
    raise

print("="*70 + "\n")


# ==================== ShareGPT 线性化（兼容多种字段）====================
def _join_turns(turns, sep="\n"):
    parts = []
    for t in turns:
        if not isinstance(t, dict):
            continue

        if "role" in t and "content" in t:
            role = str(t.get("role") or "").lower()
            text = str(t.get("content") or "").strip()
        elif "from" in t and "value" in t:
            role = str(t.get("from") or "").lower()
            text = str(t.get("value") or "").strip()
        else:
            if "human" in t:
                role = "user"
                text = str(t.get("human") or "").strip()
            elif "assistant" in t:
                role = "assistant"
                text = str(t.get("assistant") or "").strip()
            else:
                role = "unknown"
                text = ""

        if not text:
            continue

        if role in ["human", "user"]:
            parts.append(f"User: {text}")
        elif role in ["gpt", "assistant", "bot"]:
            parts.append(f"Assistant: {text}")
        elif role == "system":
            parts.append(f"System: {text}")
        else:
            parts.append(text)

    return sep.join(parts).strip()


def linearize_example(ex):
    if not isinstance(ex, dict):
        return ""

    if isinstance(ex.get("text"), str) and ex["text"].strip():
        return ex["text"].strip()

    if isinstance(ex.get("content"), str) and ex["content"].strip():
        return ex["content"].strip()

    turns = None
    if isinstance(ex.get("conversations"), list):
        turns = ex["conversations"]
    elif isinstance(ex.get("messages"), list):
        turns = ex["messages"]
    elif isinstance(ex.get("conversation"), list):
        turns = ex["conversation"]

    if isinstance(turns, list) and turns:
        return _join_turns(turns)

    if ("instruction" in ex) and ("output" in ex):
        instr = str(ex.get("instruction") or "").strip()
        inp = str(ex.get("input") or "").strip()
        outp = str(ex.get("output") or "").strip()
        body = "\n".join(s for s in [instr, inp, outp] if s).strip()
        return body

    return ""


def iter_examples(paths):
    for fp in paths:
        opener = gzip.open if fp.endswith(".gz") else open
        with opener(fp, "rt", encoding="utf-8") as f:
            first = f.read(1)
            if not first:
                continue
            f.seek(0)

            if first == "[":
                try:
                    arr = json.load(f)
                except Exception as e:
                    print(f"[警告] 读取失败（跳过）{fp}: {e}")
                    continue
                for ex in arr:
                    yield ex
            else:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception as e:
                        if i <= 3:
                            print(f"[警告] 解析失败 {fp}:{i}: {e}（跳过该行）")


def load_sharegpt_texts(data_files):
    kept = 0
    dropped = 0
    texts = []
    for ex in iter_examples(data_files):
        if isinstance(ex, dict):
            ex.pop("category", None)

        txt = linearize_example(ex)
        if not txt or len(txt.strip()) < 10:
            dropped += 1
            continue
        texts.append(txt.strip())
        kept += 1

    print(f"[Dataset] 文本提取: 保留 {kept} 条，丢弃 {dropped} 条")
    return texts


def train_test_split_texts(texts, test_ratio, seed):
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    cut = int(len(texts) * (1.0 - test_ratio))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    return train_texts, test_texts


# ==================== Dataset（与 final 的 BenchmarkDataset 对齐）====================
class BenchmarkDataset(Dataset):
    """评测数据集类（与 train_sharegpt_final.py 一致）"""

    def __init__(self, tokenizer, examples, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        print(f"[Dataset] Tokenization {len(examples)} 个样本...")

        for idx, text in enumerate(examples):
            if len(text.strip()) < 10:
                continue

            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )

            self.examples.append({
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
            })

            if (idx + 1) % 500 == 0:
                print(f"  进度: {idx+1}/{len(examples)}")

        print(f"[Dataset] ✓ 完成: {len(self.examples)} 个有效样本\n")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': item['input_ids'].clone(),
        }


# ==================== 参数（尽量对齐 final；仅额外提供 data_files 用于本地 ShareGPT）====================
def get_args():
    ap = argparse.ArgumentParser(description="ShareGPT strict-final logging trainer")

    ap.add_argument("--dataset", type=str, default="sharegpt", choices=["sharegpt"], help="数据集名称（固定 sharegpt）")
    ap.add_argument("--num_samples", type=int, default=0, help="训练样本数（0=全部）")

    ap.add_argument("--model_path", type=str, default="/opt/pangu/openPangu-Embedded-7B-V1.1", help="预训练模型路径")
    # ap.add_argument("--model_path", type=str, default="/root/nvme0n1/Noneq_Neural_Network/pretrained_models/1/Qwen2.5-7B/qwen/Qwen2___5-7B", help="预训练模型路径")    

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, nargs='+', default=["q_proj", "v_proj"])

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_iters", type=int, default=2500)
    ap.add_argument("--eval_interval", type=int, default=50)
    ap.add_argument("--grad_accum_steps", type=int, default=4)

    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--use_fdt_init", action="store_true")
    ap.add_argument("--fdt_alpha", type=float, default=1.1)
    ap.add_argument("--fdt_method", type=str, default="fft", choices=["fft", "ar"])
    ap.add_argument("--verify_fdt", action="store_true")
    ap.add_argument("--plot_spectra", action="store_true")
    ap.add_argument("--init_preset", type=str, default=None, choices=["baseline", "soft", "medium", "strong"])

    ap.add_argument("--record_gradnorm", action="store_true", help="记录每步的梯度范数")
    ap.add_argument("--full_test_eval", action="store_true", help="完整测试集评估（不限制batch数）")
    ap.add_argument("--measure_final_spectrum", action="store_true", help="训练结束后测量功率谱")

    ap.add_argument("--out_dir", type=str, required=True,default="outputs_sharegpt")
    ap.add_argument("--seed", type=int, default=1107)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--device", type=str, default="npu:1")

    # 额外：本地 ShareGPT 文件路径（默认使用你机器上已存在的文件）
    ap.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        default=["/root/nvme0n1/Noneq_Neural_Network/pretrained_models/sharegpt_datasets/computer_en_26k.jsonl"],
        help="ShareGPT 本地 json/jsonl(.gz) 文件路径列表"
    )
    ap.add_argument("--test_ratio", type=float, default=0.1, help="从本地文件切分 test 的比例")

    return ap.parse_args()


# ==================== 主函数 =====================
def main():
    if not PEFT_AVAILABLE:
        print("错误: PEFT库不可用")
        return

    args = get_args()

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 保存配置（与 final 一致：json.dump(vars(args))）
    config_file = os.path.join(args.out_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ 配置保存: {config_file}\n")

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设备设置（与 final 一致的风格）
    device = torch.device(args.device)
    device_type = args.device.split(":")[0]

    if device_type == "npu":
        try:
            import torch_npu
            torch_npu.npu.set_device(device)
            torch_npu.npu.manual_seed_all(args.seed)
            print(f"[设备] ✓ NPU 初始化成功: {device}\n")
        except Exception as e:
            print(f"[设备] ❌ NPU 初始化失败: {e}")
            return
    else:
        print(f"[设备] 使用: {device}\n")

    # ==================== 步骤 1: 加载模型 ====================
    print("=" * 70)
    print("📦 步骤 1: 加载模型")
    print("=" * 70)

    print(f"[模型] 路径: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[模型] ✓ Tokenizer: vocab_size={tokenizer.vocab_size}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    total_params, _ = count_parameters(model)
    print(f"[模型] ✓ 加载成功: {total_params/1e9:.2f}B 参数\n")

    # ==================== 步骤 2: 应用 LoRA ====================
    print("=" * 70)
    print("🔧 步骤 2: 应用 LoRA")
    print("=" * 70)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print(f"[LoRA] r={args.lora_r}, α={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"[LoRA] target_modules={args.lora_target_modules}")

    model = get_peft_model(model, lora_config)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"[LoRA] ✓ 可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)\n")

    # ==================== 步骤 3: FDT 初始化 ====================
    print("=" * 70)
    print("🎯 步骤 3: FDT 初始化")
    print("=" * 70)

    init_start_time = time.time()

    if args.init_preset:
        preset_configs = {
            "baseline": {"use_fdt": False, "alpha": None, "name": "PEFT Default (Kaiming+Zero)"},
            "soft": {"use_fdt": True, "alpha": 0.8, "name": "FDT-Soft (α=0.8)"},
            "medium": {"use_fdt": True, "alpha": 1.1, "name": "FDT-Medium (α=1.1)"},
            "strong": {"use_fdt": True, "alpha": 1.5, "name": "FDT-Strong (α=1.5)"},
        }
        config = preset_configs[args.init_preset]
        print(f"[预设] {config['name']}\n")
        if config["use_fdt"]:
            args.use_fdt_init = True
            args.fdt_alpha = config["alpha"]

    init_info = {
        "use_fdt": args.use_fdt_init,
        "preset": args.init_preset,
        "alpha": None,
        "method": "peft_default",
        "lora_a_init": "kaiming_uniform",
        "lora_b_init": "zero",
        "init_time_seconds": None,
        "measured_alphas_init": {},
        "measured_alphas_final": {},
        "verification_passed": None,
    }

    if args.use_fdt_init:
        if not FDT_INIT_AVAILABLE or apply_fdt_to_lora is None:
            print("[FDT] ❌ FDT 模块不可用，回退到 PEFT 默认初始化")
            args.use_fdt_init = False
        else:
            print(f"[FDT] 应用初始化: α={args.fdt_alpha:.2f}, 方法={args.fdt_method}")

            # 兼容不同版本签名
            applied = False
            for kwargs in (
                {"alpha": args.fdt_alpha, "method": args.fdt_method, "verbose": args.verbose},
                {"alpha": args.fdt_alpha, "method": args.fdt_method},
                {"alpha": args.fdt_alpha},
            ):
                try:
                    apply_fdt_to_lora(model, **kwargs)
                    applied = True
                    break
                except TypeError:
                    continue

            if not applied:
                print("[FDT] ❌ apply_fdt_to_lora 调用失败，回退到默认初始化")
                args.use_fdt_init = False
            else:
                init_info["alpha"] = args.fdt_alpha
                init_info["method"] = args.fdt_method
                print("[FDT] ✓ 初始化完成")

                if args.verify_fdt and verify_fdt_initialization is not None:
                    print("\n[FDT] 验证初始化质量...")
                    verify_success = verify_fdt_initialization(
                        model,
                        target_alpha=args.fdt_alpha,
                        tolerance=0.15,
                        verbose=True
                    )
                    init_info["verification_passed"] = verify_success

                if args.plot_spectra and analyze_lora_spectra is not None:
                    print("\n[FDT] 分析初始功率谱...")
                    spectra_dir = os.path.join(args.out_dir, "init_spectra")
                    os.makedirs(spectra_dir, exist_ok=True)

                    alphas = analyze_lora_spectra(
                        model,
                        save_dir=spectra_dir,
                        plot_top_n=3,
                        verbose=args.verbose
                    )
                    init_info["measured_alphas_init"] = {k: float(v) for k, v in alphas.items()}
                    print(f"[FDT] ✓ 功率谱保存到: {spectra_dir}")
    else:
        print("[FDT] 使用 PEFT 默认初始化 (Kaiming Uniform + Zero) (Baseline)")

    init_info["init_time_seconds"] = float(time.time() - init_start_time)
    print(f"[FDT] 初始化耗时: {init_info['init_time_seconds']*1000:.2f} ms\n")

    # ==================== 步骤 4: 加载数据集（ShareGPT 本地文件）====================
    print("=" * 70)
    print("📊 步骤 4: 加载数据集 (SHAREGPT)")
    print("=" * 70)

    for fp in args.data_files:
        if not os.path.exists(fp):
            print(f"[数据] ❌ 文件不存在: {fp}")
            return

    texts = load_sharegpt_texts(args.data_files)
    if len(texts) < 10:
        print("[数据] ❌ 有效样本过少，无法训练")
        return

    train_texts, test_texts = train_test_split_texts(texts, test_ratio=args.test_ratio, seed=args.seed)

    if args.num_samples > 0 and args.num_samples < len(train_texts):
        train_texts = train_texts[:args.num_samples]
        print(f"[数据] 限制训练集: {len(train_texts)} 样本")

    # 与 final 一致：若非 full_test_eval，先限制 test 样本上限为 1000（快速评估）
    if not args.full_test_eval:
        max_test_samples = 1000
        if len(test_texts) > max_test_samples:
            test_texts = test_texts[:max_test_samples]
            print(f"[数据] 限制测试集: {max_test_samples} 样本（快速评估）")
    else:
        print(f"[数据] 使用完整测试集: {len(test_texts)} 样本")

    print(f"\n[数据] 数据划分:")
    print(f"  • 训练: {len(train_texts)} 样本")
    print(f"  • 测试: {len(test_texts)} 样本")

    print(f"\n[数据] 样本预览 (前 150 字符):")
    print(f"  {train_texts[0][:150]}...\n")

    train_dataset = BenchmarkDataset(tokenizer, train_texts, args.max_length)
    test_dataset = BenchmarkDataset(tokenizer, test_texts, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[数据] DataLoader:")
    print(f"  • 训练: {len(train_loader)} 批次")
    print(f"  • 测试: {len(test_loader)} 批次\n")

    # ==================== 步骤 5: 配置优化器 ====================
    print("=" * 70)
    print("⚙️ 步骤 5: 配置优化器")
    print("=" * 70)

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params_list,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_iters
    )

    print(f"[优化器] AdamW (lr={args.lr:.2e}, wd={args.weight_decay})")
    print(f"[调度器] Warmup {args.warmup_steps} 步")
    print(f"[梯度] 裁剪阈值={args.max_grad_norm}\n")

    # 与 final 一致：内存监控
    if device_type == "npu":
        import torch_npu
        start_memory = torch_npu.npu.memory_allocated(device)
        peak_memory = start_memory
    else:
        start_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        peak_memory = start_memory

    training_log = []
    training_losses = []
    test_losses_history = []
    best_loss = float("inf")

    # ==================== 步骤 6: 训练 ====================
    print("=" * 70)
    print("🚀 步骤 6: 开始训练")
    print("=" * 70)

    model.train()

    data_iter = iter(train_loader)
    start_time = time.time()

    print(f"\n[训练] {args.max_iters} 步")
    print("-" * 70)

    for step in range(1, args.max_iters + 1):
        step_start_time = time.time()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        # NPU FP16 转换（与 final 一致风格）
        if device_type == "npu":
            batch = {
                k: v.half() if v.dtype in [torch.float32, torch.float64] else v
                for k, v in batch.items()
            }

        try:
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ 异常损失 (Step {step}): {loss.item()}")
                optimizer.zero_grad()
                continue

            loss.backward()

        except Exception as e:
            print(f"\n⚠️ 训练错误 (Step {step}): {e}")
            optimizer.zero_grad()
            continue

        grad_norm = None
        if args.record_gradnorm:
            grad_norm = compute_gradient_norm(model)

        if step % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params_list, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        current_loss = loss.item() * args.grad_accum_steps
        training_losses.append(current_loss)

        current_lr = scheduler.get_last_lr()[0]

        if device_type == "npu":
            current_memory = torch_npu.npu.memory_allocated(device)
        else:
            current_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        peak_memory = max(peak_memory, current_memory)

        step_time = time.time() - step_start_time

        # 字段顺序严格对齐 final：step, train_loss, learning_rate, grad_norm, step_time_ms, memory_gb, test_loss
        log_entry = {
            "step": step,
            "train_loss": current_loss,
            "learning_rate": current_lr,
            "grad_norm": grad_norm,
            "step_time_ms": step_time * 1000,
            "memory_gb": (current_memory - start_memory) / 1e9,
            "test_loss": None,
        }

        if step % args.eval_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            avg_train = float(np.mean(training_losses[-args.eval_interval:]))

            model.eval()
            test_losses = []

            with torch.no_grad():
                max_eval_batches = None if args.full_test_eval else 10
                for batch_idx, test_batch in enumerate(test_loader):
                    if max_eval_batches and batch_idx >= max_eval_batches:
                        break

                    test_batch = {k: v.to(device) for k, v in test_batch.items()}
                    if device_type == "npu":
                        test_batch = {
                            k: v.half() if v.dtype in [torch.float32, torch.float64] else v
                            for k, v in test_batch.items()
                        }

                    try:
                        test_outputs = model(**test_batch)
                        test_losses.append(test_outputs.loss.item())
                    except Exception:
                        break

            model.train()

            test_loss_avg = float(np.mean(test_losses)) if test_losses else None

            if test_loss_avg:
                test_losses_history.append({"step": step, "test_loss": test_loss_avg})
                log_entry["test_loss"] = test_loss_avg

            metric_str = ""
            if step >= 100:
                loss_100 = training_losses[99]
                metric_str += f", L@100={loss_100:.4f}"
            if step >= 500:
                loss_500 = training_losses[499]
                auc_500 = sum(training_losses[:500])
                metric_str += f", L@500={loss_500:.4f}, AUC(0-500)={auc_500:.2f}"

            memory_gb = (peak_memory - start_memory) / 1e9

            log = f"[{step:5d}/{args.max_iters}] "
            log += f"训练={current_loss:.4f}, 均值={avg_train:.4f}"
            if test_loss_avg:
                log += f", 测试={test_loss_avg:.4f}"
            log += f", lr={current_lr:.2e}, {elapsed:.1f}s"
            log += f", Mem={memory_gb:.2f}GB"
            if args.record_gradnorm and grad_norm:
                log += f", GradNorm={grad_norm:.4f}"
            log += metric_str
            print(log)

            metric = test_loss_avg if test_loss_avg else avg_train
            if metric < best_loss:
                best_loss = metric
                best_model_path = os.path.join(args.out_dir, "best_model")
                model.save_pretrained(best_model_path)
                print(f"  → 保存最佳模型 (损失={best_loss:.4f})")

        training_log.append(log_entry)

    total_time = time.time() - start_time

    print("-" * 70)
    print(f"[训练] ✓ 完成! 耗时: {total_time/60:.2f} 分钟")
    print(f"[训练] 最佳损失: {best_loss:.4f}")
    print(f"[训练] 峰值内存: {(peak_memory - start_memory) / 1e9:.2f} GB\n")

    # ==================== 步骤 7: 测试集完整评估 ====================
    print("=" * 70)
    print("🧪 步骤 7: 测试集完整评估")
    print("=" * 70)

    print("[测试] 使用训练完成的模型进行完整评估")
    model.eval()
    test_losses = []

    print(f"[测试] 在 {len(test_loader)} 个批次上评估...")

    with torch.no_grad():
        for idx, test_batch in enumerate(test_loader):
            test_batch = {k: v.to(device) for k, v in test_batch.items()}
            if device_type == "npu":
                test_batch = {
                    k: v.half() if v.dtype in [torch.float32, torch.float64] else v
                    for k, v in test_batch.items()
                }

            try:
                test_outputs = model(**test_batch)
                test_losses.append(test_outputs.loss.item())
            except Exception as e:
                print(f"\n  ⚠️ 批次 {idx} 评估失败: {str(e)[:100]}")
                continue

            if (idx + 1) % 50 == 0:
                print(f"  进度: {idx+1}/{len(test_loader)}")

    test_loss = float(np.mean(test_losses)) if test_losses else float("inf")
    test_std = float(np.std(test_losses)) if test_losses else 0.0

    print("\n[测试] 结果:")
    print(f"  • 平均损失: {test_loss:.4f}")
    print(f"  • 标准差: {test_std:.4f}")
    print(f"  • 有效批次: {len(test_losses)}/{len(test_loader)}\n")

    # ==================== 步骤 7.5: 训练后功率谱测量 ====================
    if args.use_fdt_init and args.measure_final_spectrum and analyze_lora_spectra is not None:
        print("=" * 70)
        print("📊 步骤 7.5: 训练后功率谱测量")
        print("=" * 70)

        spectra_dir_final = os.path.join(args.out_dir, "final_spectra")
        os.makedirs(spectra_dir_final, exist_ok=True)

        print("[FDT] 分析训练后的功率谱...")
        alphas_final = analyze_lora_spectra(
            model,
            save_dir=spectra_dir_final,
            plot_top_n=3,
            verbose=args.verbose
        )
        init_info["measured_alphas_final"] = {k: float(v) for k, v in alphas_final.items()}
        print(f"[FDT] ✓ 训练后功率谱保存到: {spectra_dir_final}")

        if init_info["measured_alphas_init"]:
            print("\n[FDT] 功率谱演化:")
            for key in init_info["measured_alphas_init"]:
                if key in init_info["measured_alphas_final"]:
                    alpha_init = init_info["measured_alphas_init"][key]
                    alpha_final = init_info["measured_alphas_final"][key]
                    delta = alpha_final - alpha_init
                    print(f"  {key}: {alpha_init:.3f} → {alpha_final:.3f} (Δ={delta:+.3f})")
        print()

    # ==================== 步骤 8: 保存结果（严格对齐 final）====================
    print("=" * 70)
    print("💾 步骤 8: 保存结果")
    print("=" * 70)

    losses_file = os.path.join(args.out_dir, "training_losses.npy")
    np.save(losses_file, np.array(training_losses))
    print(f"[保存] ✓ 训练损失数组: {losses_file}")

    csv_file = os.path.join(args.out_dir, "training_log.csv")
    with open(csv_file, "w", newline="") as f:
        if training_log:
            fieldnames = training_log[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_log)
    print(f"[保存] ✓ 训练日志CSV: {csv_file}")

    test_history_file = os.path.join(args.out_dir, "test_loss_history.json")
    with open(test_history_file, "w") as f:
        json.dump(test_losses_history, f, indent=2)
    print(f"[保存] ✓ 测试集损失历史: {test_history_file}")

    auc_metrics = compute_auc_intervals(training_losses)

    early_convergence = {
        "loss_at_100": float(training_losses[99]) if len(training_losses) >= 100 else None,
        "loss_at_200": float(training_losses[199]) if len(training_losses) >= 200 else None,
        "loss_at_500": float(training_losses[499]) if len(training_losses) >= 500 else None,
        "loss_at_1000": float(training_losses[999]) if len(training_losses) >= 1000 else None,
    }

    results = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "init_method": "FDT" if args.use_fdt_init else "PEFT_Default",
        "init_alpha": args.fdt_alpha if args.use_fdt_init else None,
        "init_preset": args.init_preset,

        "best_train_loss": float(best_loss),
        "final_test_loss": float(test_loss),
        "final_test_loss_std": float(test_std),

        "early_convergence": early_convergence,
        "auc_metrics": auc_metrics,

        "wall_time_seconds": float(total_time),
        "wall_time_minutes": float(total_time / 60),
        "peak_memory_gb": float((peak_memory - start_memory) / 1e9),
        "throughput_samples_per_sec": float(len(train_dataset) / total_time) if total_time > 0 else None,
        "avg_time_per_step_ms": float(total_time * 1000 / args.max_iters),

        "init_time_seconds": init_info["init_time_seconds"],
        "init_time_ms": init_info["init_time_seconds"] * 1000 if init_info["init_time_seconds"] else None,

        "num_train_samples": len(train_dataset),
        "num_test_samples": len(test_dataset),
        "test_batches_evaluated": len(test_losses),

        "measured_alphas_init": init_info.get("measured_alphas_init", {}),
        "measured_alphas_final": init_info.get("measured_alphas_final", {}),
    }

    results_file = os.path.join(args.out_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(convert_to_json_serializable(results), f, indent=2)
    print(f"[保存] ✓ 结果汇总: {results_file}")

    init_info_file = os.path.join(args.out_dir, "init_info.json")
    with open(init_info_file, "w") as f:
        json.dump(convert_to_json_serializable(init_info), f, indent=2)
    print(f"[保存] ✓ 初始化信息: {init_info_file}")

    final_path = os.path.join(args.out_dir, "final_model")
    try:
        model.save_pretrained(final_path)
        print(f"[保存] ✓ 最终模型: {final_path}")
    except Exception as e:
        print(f"[保存] ⚠️ 最终模型保存失败: {e}")

    try:
        tokenizer.save_pretrained(args.out_dir)
        print(f"[保存] ✓ Tokenizer: {args.out_dir}\n")
    except Exception as e:
        print(f"[保存] ⚠️ Tokenizer 保存失败: {e}\n")

    # ==================== 完成 ====================
    print("=" * 70)
    print("🎉 训练完成!")
    print("=" * 70)
    print(f"\n数据集: {args.dataset.upper()}")
    print(f"模型: {args.model_path.split('/')[-1]}")
    print(f"LoRA 秩: r={args.lora_r}")
    print(f"初始化: {'FDT (α='+str(args.fdt_alpha)+')' if args.use_fdt_init else 'PEFT Default (Kaiming+Zero)'}")

    print(f"\n📊 训练指标:")
    print(f"  • 最佳训练损失: {best_loss:.4f}")
    print(f"  • 最终测试损失: {test_loss:.4f}")

    if early_convergence["loss_at_100"]:
        print(f"\n📈 早期收敛:")
        print(f"  • Loss@100: {early_convergence['loss_at_100']:.4f}")
        if early_convergence["loss_at_500"]:
            print(f"  • Loss@500: {early_convergence['loss_at_500']:.4f}")

    if auc_metrics.get("auc_0_500"):
        print(f"\n📉 AUC 指标:")
        print(f"  • AUC(0-500): {auc_metrics['auc_0_500']:.2f}")
        if auc_metrics.get("auc_0_2500"):
            print(f"  • AUC(0-2500): {auc_metrics['auc_0_2500']:.2f}")

    print(f"\n⏱️ 效率指标:")
    print(f"  • 训练时间: {total_time/60:.2f} 分钟")
    print(f"  • 峰值内存: {(peak_memory - start_memory) / 1e9:.2f} GB")
    print(f"  • 吞吐量: {len(train_dataset) / total_time:.2f} samples/s" if total_time > 0 else "  • 吞吐量: None")

    if init_info["init_time_seconds"]:
        print(f"  • 初始化耗时: {init_info['init_time_seconds']*1000:.2f} ms")

    print(f"\n💾 输出目录: {args.out_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
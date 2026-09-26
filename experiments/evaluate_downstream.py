"""
下游任务评估脚本 - 支持 4 数据集的完整评估

数据集:
  - gsm8k: 数学推理 (准确率)
  - cmmlu: 中文知识 (准确率)
  - sharegpt: 对话交互 (Perplexity)
  - mbpp: 代码生成 (通过率)

运行示例:
  python evaluate_downstream.py \
      --model_path outputs_gsm8k_alpha1.1_r16/best_model \
      --base_model /opt/pangu/openPangu-Embedded-7B-V1.1 \
      --dataset gsm8k \
      --device npu:1 \
      --num_samples 100
"""

import os
import sys
import time
import argparse
import json
import re
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ==================== 数据集路径配置 ====================
DATASET_PATHS = {
    'gsm8k': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/gsm8k',
    'cmmlu': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/cmmlu/processed',
    'sharegpt': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/sharegpt_datasets/computer_en_26k.jsonl',
    'mbpp': '/root/nvme0n1/Noneq_Neural_Network/pretrained_models/mbpp/processed',
}

# ==================== 评估函数 ====================

class GSM8KEvaluator:
    """GSM8K 数学推理评估"""

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def format_prompt(self, question):
        return f"问题：{question}\n解答："

    def extract_answer(self, text):
        """提取 GSM8K 答案中的数字"""
        # GSM8K 答案格式: "#### 42" 或 "The answer is 42"
        patterns = [
            r'####\s*([-+]?\d*\.?\d+)',
            r'答案[是为][:：]\s*([-+]?\d*\.?\d+)',
            r'answer\s+is\s+([-+]?\d*\.?\d+)',
            r'=\s*([-+]?\d*\.?\d+)(?:\s|$|\.|,)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue

        # 最后尝试：提取最后一个数字
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                pass
        return None

    def normalize_answer(self, answer):
        """标准化答案（处理分数、百分数等）"""
        if isinstance(answer, (int, float)):
            return float(answer)

        text = str(answer).strip()

        # 优先处理 GSM8K 的 #### 格式
        hash_match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
        if hash_match:
            try:
                return float(hash_match.group(1))
            except:
                pass

        text = text.lower()

        # 处理百分数
        if '%' in text:
            text = text.replace('%', '').strip()
            try:
                return float(text) / 100
            except:
                pass

        # 处理分数
        frac_match = re.match(r'(\d+)\s*/\s*(\d+)', text)
        if frac_match:
            try:
                return float(frac_match.group(1)) / float(frac_match.group(2))
            except:
                pass

        # 提取数字（返回最后一个，通常是答案）
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                pass

        return None

    def evaluate(self, model, test_dataset, max_samples=None, max_new_tokens=256):
        """评估 GSM8K"""
        correct = 0
        total = 0
        results = []

        indices = list(range(len(test_dataset)))
        if max_samples and max_samples < len(indices):
            indices = indices[:max_samples]

        print(f"\n[GSM8K] 评估 {len(indices)} 个样本...")

        for idx in tqdm(indices, desc="GSM8K"):
            example = test_dataset[idx]
            question = example['question']
            true_answer = example['answer']

            # 构造 prompt
            prompt = self.format_prompt(question)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # 生成答案
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_answer = generated.split("解答：")[-1].strip() if "解答：" in generated else generated

            # 提取答案
            pred = self.extract_answer(generated_answer)
            true = self.normalize_answer(true_answer)

            is_correct = False
            if pred is not None and true is not None:
                # 允许小的浮点误差
                is_correct = abs(pred - true) < 1e-6 or abs(pred - true) / max(abs(true), 1) < 0.01

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'question': question[:100],  # 限制长度
                'true_answer': str(true_answer)[:100],
                'generated': generated_answer[:200],
                'extracted_pred': pred,
                'extracted_true': true,
                'correct': is_correct,
            })

        accuracy = correct / total if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results[:10],  # 只保存前 10 个详细结果
        }


class CMMLUEvaluator:
    """CMMLU 中文知识评估"""

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self.label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    def format_prompt(self, question, choices):
        """格式化 prompt"""
        prompt = f"问题：{question}\n"
        prompt += f"A. {choices[0]}\n"
        prompt += f"B. {choices[1]}\n"
        prompt += f"C. {choices[2]}\n"
        prompt += f"D. {choices[3]}\n"
        prompt += "答案："
        return prompt

    def evaluate_likelihood(self, model, test_dataset, max_samples=None):
        """使用似然方法评估（推荐，更准确）"""
        correct = 0
        total = 0
        results = []

        indices = list(range(len(test_dataset)))
        if max_samples and max_samples < len(indices):
            indices = indices[:max_samples]

        print(f"\n[CMMLU] 评估 {len(indices)} 个样本（似然法）...")

        for idx in tqdm(indices, desc="CMMLU"):
            example = test_dataset[idx]
            question = example['Question']
            choices = [example['A'], example['B'], example['C'], example['D']]
            true_answer = example['Answer']

            # 计算每个选项的似然
            losses = []
            for choice in choices:
                prompt = self.format_prompt(question, choices)
                text = prompt + choice

                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    # 负对数似然（越小越好）
                    loss = outputs.loss.item()
                    losses.append(-loss)  # 转为正数（越大越好）

            predicted_idx = np.argmax(losses)
            predicted_label = ['A', 'B', 'C', 'D'][predicted_idx]
            is_correct = (predicted_label == true_answer)

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'question': question[:100],
                'predicted': predicted_label,
                'true': true_answer,
                'correct': is_correct,
                'losses': losses,
            })

        accuracy = correct / total if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results[:10],
        }

    def evaluate_generate(self, model, test_dataset, max_samples=None, max_new_tokens=32):
        """使用生成方法评估（备选）"""
        correct = 0
        total = 0
        results = []

        indices = list(range(len(test_dataset)))
        if max_samples and max_samples < len(indices):
            indices = indices[:max_samples]

        print(f"\n[CMMLU] 评估 {len(indices)} 个样本（生成法）...")

        for idx in tqdm(indices, desc="CMMLU"):
            example = test_dataset[idx]
            question = example['Question']
            choices = [example['A'], example['B'], example['C'], example['D']]
            true_answer = example['Answer']

            prompt = self.format_prompt(question, choices)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_part = generated.split("答案：")[-1].strip() if "答案：" in generated else generated

            # 提取答案
            predicted_label = None
            for label in ['A', 'B', 'C', 'D']:
                if label in answer_part[:10]:  # 只看前几个字符
                    predicted_label = label
                    break

            if predicted_label is None:
                # 尝试从首字符提取
                first_char = answer_part[0].upper() if answer_part else ''
                if first_char in ['A', 'B', 'C', 'D']:
                    predicted_label = first_char

            is_correct = (predicted_label == true_answer)

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'question': question[:100],
                'predicted': predicted_label,
                'true': true_answer,
                'correct': is_correct,
                'generated': answer_part[:50],
            })

        accuracy = correct / total if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results[:10],
        }


class MBPPEvaluator:
    """MBPP 代码生成评估"""

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def format_prompt(self, problem_text):
        return f"# Problem\n{problem_text}\n\n# Solution\n"

    def extract_code(self, generated):
        """提取生成的代码"""
        if "# Solution\n" in generated:
            code = generated.split("# Solution\n")[-1].strip()
        else:
            code = generated.strip()

        # 移除可能的额外注释
        lines = []
        for line in code.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
            elif lines:  # 已有代码，遇到注释停止
                break

        return '\n'.join(lines)

    def execute_test_cases(self, code, test_list, timeout=5):
        """执行测试用例"""
        import sys
        import io
        import contextlib

        # 准备执行环境
        test_code = code + '\n\n'
        test_code += '\n'.join(test_list)

        # 捕获输出
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            exec_globals = {'__builtins__': __builtins__}
            exec(test_code, exec_globals)

            # 检查是否有断言错误
            output = sys.stdout.getvalue()
            success = True

        except AssertionError:
            success = False
        except Exception as e:
            success = False
        finally:
            sys.stdout = old_stdout

        return success

    def evaluate(self, model, test_dataset, max_samples=None, max_new_tokens=256):
        """评估 MBPP"""
        correct = 0
        total = 0
        results = []

        indices = list(range(len(test_dataset)))
        if max_samples and max_samples < len(indices):
            indices = indices[:max_samples]

        print(f"\n[MBPP] 评估 {len(indices)} 个样本...")

        for idx in tqdm(indices, desc="MBPP"):
            example = test_dataset[idx]
            problem_text = example['text']
            true_code = example['code']
            test_list = example.get('test_list', [])

            prompt = self.format_prompt(problem_text)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_code = self.extract_code(generated)

            # 执行测试
            passed = False
            if test_list:
                try:
                    passed = self.execute_test_cases(generated_code, test_list)
                except Exception as e:
                    passed = False
            else:
                # 没有测试用例，检查代码相似性
                passed = generated_code.strip() == true_code.strip()

            if passed:
                correct += 1
            total += 1

            results.append({
                'problem': problem_text[:100],
                'generated_code': generated_code[:200],
                'passed': passed,
            })

        pass_rate = correct / total if total > 0 else 0

        return {
            'pass_rate': pass_rate,
            'correct': correct,
            'total': total,
            'results': results[:10],
        }


class ShareGPTEvaluator:
    """ShareGPT 对话评估"""

    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def format_conversation(self, conversations):
        """格式化对话

        ShareGPT 数据在不同来源/预处理后可能出现多种结构：
        - str: 已经拼接好的全文
        - list[dict]: {from,value} 或 {role,content}
        - list[str]: 每行一句
        - 其他/混合：尽量降级为可用文本
        """

        if conversations is None:
            return ""

        # 有些数据集直接把 conversations 存成一整段字符串
        if isinstance(conversations, str):
            return conversations.strip()

        # 有些数据集是 dict（例如单轮对话）
        if isinstance(conversations, dict):
            conversations = [conversations]

        if not isinstance(conversations, (list, tuple)):
            return str(conversations).strip()

        lines = []
        for turn in conversations:
            if turn is None:
                continue
            if isinstance(turn, str):
                s = turn.strip()
                if s:
                    lines.append(s)
                continue
            if isinstance(turn, dict):
                role = turn.get('from', turn.get('role', 'unknown'))
                content = turn.get('value', turn.get('content', turn.get('text', '')))
                if content is None:
                    content = ""
                role = str(role)
                content = str(content)
                if content.strip():
                    lines.append(f"{role}: {content}")
                continue

            # 兜底：未知类型
            s = str(turn).strip()
            if s:
                lines.append(s)

        return "\n".join(lines).strip()

    def compute_perplexity(self, model, test_dataset, max_samples=None, max_length=512):
        """计算 Perplexity"""
        total_loss = 0
        total_tokens = 0
        results = []

        indices = list(range(len(test_dataset)))
        if max_samples and max_samples < len(indices):
            indices = indices[:max_samples]

        print(f"\n[ShareGPT] 计算 {len(indices)} 个样本的 Perplexity...")

        for idx in tqdm(indices, desc="ShareGPT"):
            example = test_dataset[idx]
            conversations = example.get('conversations', [])
            text = self.format_conversation(conversations)

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                n_tokens = inputs["input_ids"].size(1)

                total_loss += loss * n_tokens
                total_tokens += n_tokens

                results.append({
                    'loss': loss,
                    'tokens': n_tokens,
                    'ppl': math.exp(loss),
                })

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss)

        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'results': results[:10],
        }


def load_sharegpt_conversations_dataset(path: str):
    from datasets import Dataset, Features, Sequence, Value

    def norm_turn(t):
        if not isinstance(t, dict):
            return None
        # 常见两种格式：{"from","value"} 或 {"role","content"}
        speaker = t.get("from", t.get("role", "unknown"))
        content = t.get("value", t.get("content", t.get("text", "")))
        if content is None:
            content = ""
        speaker = str(speaker)
        content = str(content)
        return {"from": speaker, "value": content}

    def extract_conversations(obj):
        conv = obj.get("conversations", None)
        if conv is None:
            conv = obj.get("messages", None)
        if conv is None:
            conv = obj.get("conversation", None)
        if not isinstance(conv, list):
            return None
        out = []
        for t in conv:
            nt = norm_turn(t)
            if nt is not None and nt["value"] != "":
                out.append(nt)
        return out

    records = []
    skipped = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        # 既兼容 JSONL，也兼容一个大的 JSON 数组
        first_nonspace = ""
        pos = f.tell()
        for line in f:
            s = line.strip()
            if s:
                first_nonspace = s[0]
                break
        f.seek(pos)

        if first_nonspace == "[":
            data = json.load(f)
            for obj in data:
                if not isinstance(obj, dict):
                    skipped += 1
                    continue
                conv = extract_conversations(obj)
                if conv is None:
                    skipped += 1
                    continue
                records.append({"conversations": conv})
        else:
            for line_no, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    skipped += 1
                    continue
                if not isinstance(obj, dict):
                    skipped += 1
                    continue
                conv = extract_conversations(obj)
                if conv is None:
                    skipped += 1
                    continue
                records.append({"conversations": conv})

    features = Features(
        {"conversations": Sequence({"from": Value("string"), "value": Value("string")})}
    )
    ds = Dataset.from_list(records, features=features)

    if skipped:
        print(f"[ShareGPT] ⚠️ 跳过无效样本: {skipped}")
    return ds

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="下游任务评估脚本")

    # 模型配置
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练好的模型路径")
    parser.add_argument("--base_model", type=str,
                       default="/opt/pangu/openPangu-Embedded-7B-V1.1",
                       help="基础模型路径（用于 LoRA）")

    # 数据集配置
    parser.add_argument("--dataset", type=str, required=True,
                       choices=['gsm8k', 'cmmlu', 'sharegpt', 'mbpp'],
                       help="数据集名称")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="数据集路径（覆盖默认路径）")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="评估样本数（None=全部）")

    # 生成配置
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="最大生成 token 数")
    parser.add_argument("--eval_method", type=str, default='auto',
                       choices=['auto', 'likelihood', 'generate'],
                       help="CMMLU 评估方法")

    # 输出配置
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（默认为模型路径下的 evaluation）")
    parser.add_argument("--save_details", action="store_true",
                       help="保存详细结果")

    # 其他
    parser.add_argument("--device", type=str, default="npu:1")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批大小（目前只支持 1）")

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device)
    device_type = args.device.split(':')[0]

    if device_type == 'npu':
        try:
            import torch_npu
            torch_npu.npu.set_device(device)
            print(f"[设备] ✓ NPU 初始化成功: {device}\n")
        except Exception as e:
            print(f"[设备] ❌ NPU 初始化失败: {e}")
            return
    else:
        print(f"[设备] 使用: {device}\n")

    # 获取数据集路径
    dataset_path = args.dataset_path or DATASET_PATHS.get(args.dataset)
    if not dataset_path:
        print(f"❌ 未找到数据集路径: {args.dataset}")
        return

    # 加载 tokenizer
    print("="*70)
    print("📦 加载模型")
    print("="*70)

    print(f"[模型] 路径: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model if os.path.exists(os.path.join(args.model_path, 'adapter_config.json')) else args.model_path,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[模型] ✓ Tokenizer 加载成功")

        # 加载模型
    print(f"[模型] 路径: {args.model_path}")
    
    # 检查是否是 LoRA 模型
    adapter_config_path = os.path.join(args.model_path, 'adapter_config.json')
    
    if os.path.exists(adapter_config_path):
        # ⚡⚡⚡ LoRA 模型加载（NPU 兼容版本）⚡⚡⚡
        from peft import PeftModel, PeftConfig
        import json
        
        print(f"[模型] 检测到 LoRA 模型")
        
        # ⚡⚡⚡ 方案1: 修改 adapter_config.json 临时禁用 safetensors ⚡⚡⚡
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # 检查是否有 safetensors 文件
        safetensors_file = os.path.join(args.model_path, 'adapter_model.safetensors')
        pytorch_file = os.path.join(args.model_path, 'adapter_model.bin')
        
        if os.path.exists(safetensors_file) and not os.path.exists(pytorch_file):
            # 转换 safetensors 为 PyTorch 格式
            print(f"[模型] 检测到 safetensors 格式，正在转换为 PyTorch 格式...")
            
            try:
                from safetensors.torch import load_file
                
                # 强制在 CPU 上加载
                state_dict = load_file(safetensors_file, device='cpu')
                
                # 保存为 PyTorch 格式
                torch.save(state_dict, pytorch_file)
                print(f"[模型] ✓ 已转换为 PyTorch 格式: {pytorch_file}")
                
            except Exception as e:
                print(f"[模型] ⚠️ 转换失败: {e}")
                print(f"[模型] 尝试使用备用加载方案...")
        
        # 加载基础模型
        print(f"[模型] 加载基础模型: {args.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='cpu'  # ⚡ 先在 CPU 上加载
        )
        
        # 加载 LoRA adapter（优先使用 PyTorch 格式）
        print(f"[模型] 加载 LoRA adapter: {args.model_path}")
        
        try:
            # ⚡⚡⚡ 核心修复：设置环境变量禁用 safetensors ⚡⚡⚡
            os.environ['PEFT_USE_SAFETENSORS'] = '0'  # 禁用 safetensors
            
            model = PeftModel.from_pretrained(
                base_model,
                args.model_path,
                device_map='cpu'  # ⚡ 先在 CPU 上加载
            )
            
            print(f"[模型] ✓ LoRA 模型加载成功")
        
        except Exception as e:
            print(f"[模型] ❌ LoRA 加载失败: {e}")
            
            # ⚡⚡⚡ 备用方案：手动加载权重 ⚡⚡⚡
            print(f"[模型] 尝试手动加载权重...")
            
            if os.path.exists(pytorch_file):
                state_dict = torch.load(pytorch_file, map_location='cpu')
                base_model.load_state_dict(state_dict, strict=False)
                model = base_model
                print(f"[模型] ✓ 手动加载成功")
            else:
                raise RuntimeError(f"无法加载 LoRA 权重，请检查模型文件")
    
    else:
        # ⚡⚡⚡ 完整模型加载 ⚡⚡⚡
        print(f"[模型] 加载完整模型: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='cpu'  # ⚡ 先在 CPU 上加载
        )
        print(f"[模型] ✓ 完整模型加载成功")
    
    # ⚡⚡⚡ 统一移动到 NPU 设备 ⚡⚡⚡
    print(f"[模型] 移动模型到设备: {device}")
    model = model.to(device)
    model.eval()
    
    print(f"[模型] ✓ 模型已就绪\n")

    # 加载数据集
    print("="*70)
    print(f"📊 加载数据集 ({args.dataset.upper()})")
    print("="*70)

    try:
        from datasets import load_from_disk, load_dataset

        if os.path.isdir(dataset_path):
            dataset = load_from_disk(dataset_path)

            # load_from_disk 可能返回 DatasetDict 或 Dataset
            if isinstance(dataset, dict) or hasattr(dataset, "keys"):
                if 'test' in dataset:
                    test_dataset = dataset['test']
                elif 'validation' in dataset:
                    test_dataset = dataset['validation']
                else:
                    test_dataset = dataset['train']
            else:
                test_dataset = dataset  # 直接是 Dataset

        elif os.path.isfile(dataset_path):
            ext = os.path.splitext(dataset_path)[1].lower()
            if ext in [".jsonl", ".json"]:
                if args.dataset == "sharegpt":
                    test_dataset = load_sharegpt_conversations_dataset(dataset_path)
                else:
                    test_dataset = load_dataset("json", data_files=dataset_path, split="train")
            else:
                raise ValueError(f"不支持的文件类型: {dataset_path}")
        else:
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        print(f"[数据] ✓ 加载成功: {len(test_dataset)} 样本")

        if args.num_samples and args.num_samples < len(test_dataset):
            print(f"[数据] 限制评估: {args.num_samples} 样本")

        print()


    except Exception as e:
        print(f"[数据] ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 评估
    print("="*70)
    print(f"🎯 开始评估 ({args.dataset.upper()})")
    print("="*70)

    start_time = time.time()
    metrics = {}

    if args.dataset == 'gsm8k':
        evaluator = GSM8KEvaluator(tokenizer, device)
        result = evaluator.evaluate(
            model, test_dataset,
            max_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens
        )
        metrics['accuracy'] = result['accuracy']
        print(f"\n[GSM8K] 准确率: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    elif args.dataset == 'cmmlu':
        evaluator = CMMLUEvaluator(tokenizer, device)

        # 选择评估方法
        method = args.eval_method
        if method == 'auto':
            method = 'likelihood'  # 默认使用似然法

        if method == 'likelihood':
            result = evaluator.evaluate_likelihood(model, test_dataset, max_samples=args.num_samples)
        else:
            result = evaluator.evaluate_generate(
                model, test_dataset,
                max_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens
            )

        metrics['accuracy'] = result['accuracy']
        print(f"\n[CMMLU] 准确率: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    elif args.dataset == 'mbpp':
        evaluator = MBPPEvaluator(tokenizer, device)
        result = evaluator.evaluate(
            model, test_dataset,
            max_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens
        )
        metrics['pass_rate'] = result['pass_rate']
        print(f"\n[MBPP] 通过率: {result['pass_rate']:.2%} ({result['correct']}/{result['total']})")

    elif args.dataset == 'sharegpt':
        evaluator = ShareGPTEvaluator(tokenizer, device)
        result = evaluator.compute_perplexity(model, test_dataset, max_samples=args.num_samples)
        metrics['perplexity'] = result['perplexity']
        metrics['avg_loss'] = result['avg_loss']
        print(f"\n[ShareGPT] Perplexity: {result['perplexity']:.2f}")
        print(f"[ShareGPT] 平均损失: {result['avg_loss']:.4f}")

    eval_time = time.time() - start_time

    # 保存结果
    print("="*70)
    print("💾 保存结果")
    print("="*70)

    output_dir = args.output_dir or os.path.join(args.model_path, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    # 保存指标
    summary = {
        'dataset': args.dataset,
        'model_path': args.model_path,
        'base_model': args.base_model,
        'eval_time_seconds': eval_time,
        # 'num_samples': result.get('total', args.num_samples),
        'num_samples': (min(args.num_samples, len(test_dataset)) if args.num_samples else len(test_dataset)),
        **metrics,
    }

    summary_file = os.path.join(output_dir, f"{args.dataset}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[保存] ✓ 指标汇总: {summary_file}")

    # 保存详细结果
    if args.save_details and 'results' in result:
        details_file = os.path.join(output_dir, f"{args.dataset}_details.json")
        with open(details_file, 'w') as f:
            json.dump(result['results'], f, indent=2, ensure_ascii=False)
        print(f"[保存] ✓ 详细结果: {details_file}")

    print()
    print("="*70)
    print("🎉 评估完成!")
    print("="*70)
    print(f"\n数据集: {args.dataset.upper()}")
    print(f"评估时间: {eval_time:.2f} 秒")
    print(f"\n📊 结果:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['accuracy', 'pass_rate']:
                print(f"  • {key}: {value:.2%}")
            else:
                print(f"  • {key}: {value:.4f}")
        else:
            print(f"  • {key}: {value}")
    print(f"\n💾 输出目录: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

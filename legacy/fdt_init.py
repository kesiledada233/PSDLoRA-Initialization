"""
频域感知初始化器（Frequency-Domain-Aware Initialization）

核心思想：
- 传统初始化（Xavier/He）→ 白噪声（α ≈ 0）
- FDT 初始化 → 粉红噪声（α ≈ 1.2）
- 匹配收敛模型的频域特性，加速早期收敛

实现方法：
1. FFT 合成法：精确生成 P(f) ∝ f^(-α)
2. 自回归法：快速近似，适合大规模部署

适用场景：
- LoRA 微调（主要用途）
- 全参数微调
- 从头预训练
"""

import torch
import numpy as np
from typing import Optional, Tuple
import warnings


def fdt_initialize_(
    tensor: torch.Tensor,
    alpha: float = 1.2,
    temp_ratio: Optional[float] = None,
    method: str = 'fft',
    verbose: bool = False  # ← 移除 fan_mode 和 target_std 参数
) -> torch.Tensor:
    """
    频域感知初始化（只生成功率律噪声，不做方差归一化）
    
    核心原理：
    1. 生成功率谱 P(f) ∝ f^(-α) 的随机信号
    2. 返回**未归一化**的噪声（让调用者决定方差）
    
    Args:
        tensor: 要初始化的参数张量（任意形状）
        alpha: 功率律指数（推荐 1.2，范围 0.8-1.5）
        temp_ratio: 高频/低频能量比（可选）
        method: 'fft'（精确）或 'ar'（快速）
        verbose: 是否打印详细信息
    
    Returns:
        初始化后的张量（原地修改，**未归一化**）
    """
    with torch.no_grad():
        original_shape = tensor.shape
        original_device = tensor.device
        original_dtype = tensor.dtype
        
        # 展平为 1D
        numel = tensor.numel()
        
        if verbose:
            print(f"🔧 FDT 初始化: shape={original_shape}, α={alpha:.2f}, method={method}")
        
        # === 方法 1: FFT 频域合成（精确）===
        if method == 'fft':
            n_freqs = numel // 2 + 1
            freqs = np.arange(1, n_freqs)
            
            if len(freqs) == 0:
                warnings.warn(f"张量太小 (numel={numel})，使用标准正态分布")
                tensor.normal_(0, 1)
                return tensor
            
            # 功率谱 P(f) ∝ f^(-α)
            power_spectrum = freqs.astype(np.float64) ** (-alpha)
            
            # 可选：温度比调整
            if temp_ratio is not None and temp_ratio != 1.0:
                cutoff_idx = max(1, len(freqs) // 16)
                
                energy_low = power_spectrum[:cutoff_idx].sum()
                energy_high = power_spectrum[cutoff_idx:].sum()
                
                if energy_low > 0 and energy_high > 0:
                    current_ratio = energy_high / energy_low
                    scale = np.sqrt(temp_ratio / current_ratio)
                    power_spectrum[cutoff_idx:] *= scale
                    
                    if verbose:
                        print(f"  温度比调整: {current_ratio:.3f} → {temp_ratio:.3f}")
            
            # 振幅 A(f) = √P(f)
            amplitudes = np.sqrt(power_spectrum)
            
            # 随机相位
            phases = np.random.uniform(0, 2 * np.pi, len(freqs))
            
            # 构造复数 FFT 系数
            fft_coeffs = amplitudes * np.exp(1j * phases)
            fft_coeffs_full = np.concatenate([[0.0 + 0.0j], fft_coeffs])
            
            # 逆 FFT
            time_series = np.fft.irfft(fft_coeffs_full, n=numel)
            
            # ⚡⚡⚡ 移除方差归一化！让调用者自己决定 ⚡⚡⚡
            # 只做零均值
            time_series = time_series - time_series.mean()
        
        # === 方法 2: 自回归近似 ===
        elif method == 'ar':
            beta = 0.5 + 0.3 * (alpha - 1.0)
            beta = np.clip(beta, 0.1, 0.9)
            
            time_series = np.zeros(numel)
            time_series[0] = np.random.randn()
            
            noise_scale = np.sqrt(1 - beta**2)
            
            for i in range(1, numel):
                time_series[i] = beta * time_series[i-1] + noise_scale * np.random.randn()
            
            time_series = time_series - time_series.mean()
            
            if verbose:
                print(f"  AR 参数: β={beta:.3f}")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 转换并填充
        tensor_flat = torch.from_numpy(time_series).to(dtype=original_dtype, device=original_device)
        tensor.copy_(tensor_flat.reshape(original_shape))
        
        if verbose:
            actual_std = tensor.std().item()
            print(f"  ✓ 完成: 原始 std={actual_std:.6f}（未归一化）")

    return tensor

def apply_fdt_to_lora(
    model: torch.nn.Module,
    alpha: float = 1.2,
    temp_ratio: Optional[float] = None,
    method: str = 'fft',
    verbose: bool = True
) -> torch.nn.Module:
    """对模型中的 LoRA 层应用 FDT 初始化（带方差归一化）"""
    import math
    
    initialized_count = 0
    skipped_count = 0
    total_trainable = 0
    
    if verbose:
        print("\n" + "="*70)
        print("🔧 应用 FDT 初始化到 LoRA 层（带方差归一化）")
        print("="*70)
        print(f"配置: α={alpha:.2f}", end='')
        if temp_ratio:
            print(f", τ={temp_ratio:.2f}", end='')
        print(f", method={method}")
        print("-"*70)
    
    lora_patterns = ['lora', 'adapter', 'delta', 'ia3']
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        total_trainable += 1
        is_lora = any(pattern in name.lower() for pattern in lora_patterns)
        
        if is_lora:
            try:
                if verbose:
                    print(f"[{initialized_count+1}] {name}")
                    print(f"    形状: {list(param.shape)}")
                
                # ⚡⚡⚡ 步骤 1: 计算 Xavier 目标标准差 ⚡⚡⚡
                dimensions = param.dim()
                if dimensions >= 2:
                    fan_in = param.size(1)
                    fan_out = param.size(0)
                    
                    if dimensions > 2:
                        receptive_field_size = param[0][0].numel()
                        fan_in *= receptive_field_size
                        fan_out *= receptive_field_size
                    
                    # Xavier/Glorot 标准差
                    target_std = math.sqrt(2.0 / (fan_in + fan_out))
                else:
                    # 1D 张量（很少见）
                    target_std = 1.0 / math.sqrt(param.numel())
                
                if verbose:
                    print(f"    Xavier std: {target_std:.6f} (fan_in={fan_in}, fan_out={fan_out})")
                
                # ⚡⚡⚡ 步骤 2: 生成 FDT 噪声（未归一化）⚡⚡⚡
                fdt_initialize_(
                    param.data,
                    alpha=alpha,
                    temp_ratio=temp_ratio,
                    method=method,
                    verbose=False
                )
                
                # ⚡⚡⚡ 步骤 3: 归一化到 Xavier 方差（只做一次！）⚡⚡⚡
                with torch.no_grad():
                    current_std = param.data.std().item()
                    
                    if current_std > 1e-8:
                        scale_factor = target_std / current_std
                        param.data.mul_(scale_factor)
                        
                        if verbose:
                            actual_std = param.data.std().item()
                            print(f"    归一化: {current_std:.6f} × {scale_factor:.4f} = {actual_std:.6f}")
                    else:
                        if verbose:
                            print(f"    ⚠️  标准差过小 ({current_std:.2e})，使用随机初始化")
                        # 退化为 Xavier
                        param.data.normal_(0, target_std)
                
                # ⚡⚡⚡ 步骤 4: 验证功率律指数 ⚡⚡⚡
                try:
                    from measure_alpha import measure_alpha
                    measured_alpha = measure_alpha(param.data)
                    error = abs(measured_alpha - alpha)
                    status = "✓" if error < 0.15 else "⚠️"
                    
                    if verbose:
                        print(f"    测量 α: {measured_alpha:.3f} {status} (误差={error:.3f})")
                except Exception as e:
                    if verbose:
                        print(f"    ⚠️  无法测量 α: {e}")
                
                if verbose:
                    print()
                
                initialized_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"    ✗ 失败: {e}\n")
                import traceback
                traceback.print_exc()
                skipped_count += 1
        else:
            skipped_count += 1
    
    print("-"*70)
    print(f"统计:")
    print(f"  • 总可训练参数: {total_trainable}")
    print(f"  • 已初始化: {initialized_count}")
    print(f"  • 跳过: {skipped_count}")
    
    if initialized_count == 0:
        print("\n❌ 警告：没有参数被初始化！")
        print("可能原因:")
        print("  1. 参数名不包含 'lora' 等关键词")
        print("  2. 所有参数都是 requires_grad=False")
    elif initialized_count < total_trainable * 0.5:
        print(f"\n⚠️  警告：只初始化了 {initialized_count}/{total_trainable} 个参数")
    else:
        print(f"\n✅ 成功初始化 {initialized_count} 个 LoRA 层")
    
    print("="*70)
    
    return model


def apply_fdt_to_all_params(
    model: torch.nn.Module,
    alpha: float = 1.2,
    temp_ratio: Optional[float] = None,
    method: str = 'fft',
    skip_frozen: bool = True,
    verbose: bool = True
) -> torch.nn.Module:
    """
    对模型的所有参数应用 FDT 初始化（用于从头训练）
    
    Args:
        model: 模型
        alpha: 功率律指数
        temp_ratio: 温度比（可选）
        method: 初始化方法
        skip_frozen: 是否跳过冻结参数
        verbose: 是否打印日志
    
    Returns:
        初始化后的模型（原地修改）
    """
    initialized_count = 0
    skipped_count = 0
    total_params = 0
    
    if verbose:
        print("\n" + "="*70)
        print("🔧 FDT 初始化 - 全参数".center(70))
        print("="*70)
    
    for name, param in model.named_parameters():
        if skip_frozen and not param.requires_grad:
            skipped_count += 1
            continue
        
        # 只初始化权重矩阵（跳过 bias 和 LayerNorm）
        if len(param.shape) >= 2:
            if verbose and initialized_count < 5:  # 只打印前 5 个
                print(f"📊 {name:60s} {list(param.shape)}")
            
            fdt_initialize_(
                param,
                alpha=alpha,
                temp_ratio=temp_ratio,
                method=method,
                verbose=False
            )
            
            initialized_count += 1
            total_params += param.numel()
    
    if verbose:
        print("-"*70)
        print(f"✓ 完成: {initialized_count} 个参数, 共 {total_params:,} 元素")
        if skipped_count > 0:
            print(f"  跳过: {skipped_count} 个冻结/1D 参数")
        print("="*70 + "\n")
    
    return model


# ============ 便捷函数 ============

def init_lora_with_pink_noise(
    model: torch.nn.Module,
    alpha: float = 1.2,
    verbose: bool = True
) -> torch.nn.Module:
    """
    便捷函数：用粉红噪声初始化 LoRA（推荐配置）
    
    Example:
        >>> model = init_lora_with_pink_noise(model, alpha=1.2)
    """
    return apply_fdt_to_lora(model, alpha=alpha, method='fft', verbose=verbose)


def init_lora_with_custom_spectrum(
    model: torch.nn.Module,
    alpha: float,
    temp_ratio: float,
    verbose: bool = True
) -> torch.nn.Module:
    """
    便捷函数：用自定义谱初始化 LoRA（高级用法）
    
    Example:
        >>> # 强功率律 + 高温度比（激进探索）
        >>> model = init_lora_with_custom_spectrum(model, alpha=1.5, temp_ratio=2.0)
    """
    return apply_fdt_to_lora(
        model,
        alpha=alpha,
        temp_ratio=temp_ratio,
        method='fft',
        verbose=verbose
    )
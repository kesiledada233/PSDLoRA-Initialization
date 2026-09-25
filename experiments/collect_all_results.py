#!/usr/bin/env python3
"""
收集所有实验结果并生成汇总表格
"""

import json
import os
from pathlib import Path
import pandas as pd

# 所有实验目录
EXPERIMENT_DIRS = [
    # GSM8K 消融实验 - α 值
    "outputs_gsm8k_ablation_alpha0.6_r16",
    "outputs_gsm8k_ablation_alpha0.8_r16",
    "outputs_gsm8k_ablation_baseline_r16",  # 已有
    "outputs_gsm8k_ablation_alpha0.9_r16",   # 已有
    "outputs_gsm8k_ablation_alpha1.1_r16",   # 已有
    "outputs_gsm8k_ablation_alpha1.5_r16",
    
    # GSM8K 消融实验 - LoRA 秩
    "outputs_gsm8k_ablation_r8_baseline",
    "outputs_gsm8k_ablation_r8_alpha1.1",
    "outputs_gsm8k_ablation_r32_baseline",
    "outputs_gsm8k_ablation_r32_alpha1.1",
    
    # 其他数据集
    "outputs_cmmlu_baseline",
    "outputs_cmmlu_alpha1.1",
    "outputs_sharegpt_baseline",
    "outputs_sharegpt_alpha1.1",
    "outputs_mbpp_baseline",
    "outputs_mbpp_alpha1.1",
    
    # DeepSeek 跨模型
    "outputs_deepseek_gsm8k_baseline_r16",
    "outputs_deepseek_gsm8k_alpha1.1_r16",
]

def collect_results():
    """收集所有实验结果"""
    results_table = []
    
    for exp_dir in EXPERIMENT_DIRS:
        results_file = os.path.join(exp_dir, "results.json")
        
        if not os.path.exists(results_file):
            print(f"⚠️  未找到: {results_file}")
            continue
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # 提取模型名称
            model_name = data['model_path'].split('/')[-1]
            if 'deepseek' in model_name.lower():
                model_name = 'DeepSeek-7B'
            elif 'pangu' in model_name.lower():
                model_name = 'OpenPangu-7B'
            
            # 构建表格行
            row = {
                '实验目录': exp_dir,
                '数据集': data['dataset'].upper(),
                '模型': model_name,
                'LoRA秩': data['lora_r'],
                '初始化': data['init_method'],
                'α值': f"{data['init_alpha']:.1f}" if data['init_alpha'] else '-',
                '训练时间(min)': f"{data['wall_time_minutes']:.2f}",
                '峰值内存(GB)': f"{data['peak_memory_gb']:.2f}",
                '吞吐量(样本/s)': f"{data['throughput_samples_per_sec']:.2f}",
                'AUC(0-500)': f"{data['auc_500']:.2f}" if data['auc_500'] else 'N/A',
                '最佳训练损失': f"{data['best_train_loss']:.4f}",
                '测试损失': f"{data['final_test_loss']:.4f}",
                '测试损失标准差': f"{data['final_test_loss_std']:.4f}",
            }
            
            results_table.append(row)
            print(f"✅ 已收集: {exp_dir}")
            
        except Exception as e:
            print(f"❌ 读取失败 {results_file}: {e}")
            continue
    
    return pd.DataFrame(results_table)

def main():
    print("="*70)
    print("📊 收集所有实验结果")
    print("="*70)
    print()
    
    # 收集结果
    df = collect_results()
    
    if df.empty:
        print("\n❌ 未找到任何结果文件")
        return
    
    print()
    print("="*70)
    print("📋 结果汇总表格")
    print("="*70)
    print()
    
    # 打印 Markdown 表格
    print(df.to_markdown(index=False))
    
    # 保存 CSV
    csv_file = "all_experiments_results.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print()
    print(f"✅ 结果保存到: {csv_file}")
    
    # 保存 Excel（如果 openpyxl 可用）
    try:
        excel_file = "all_experiments_results.xlsx"
        df.to_excel(excel_file, index=False)
        print(f"✅ 结果保存到: {excel_file}")
    except ImportError:
        print("⚠️  未安装 openpyxl，跳过 Excel 导出")
    
    # 分组统计
    print()
    print("="*70)
    print("📈 分组统计")
    print("="*70)
    
    # 按数据集分组
    print("\n▶ 按数据集分组:")
    grouped = df.groupby('数据集').agg({
        'AUC(0-500)': lambda x: x.replace('N/A', '0').astype(float).mean(),
        '测试损失': lambda x: x.astype(float).mean(),
        '训练时间(min)': lambda x: x.astype(float).mean(),
    })
    print(grouped)
    
    # 按初始化方法分组
    print("\n▶ 按初始化方法分组:")
    grouped = df.groupby('初始化').agg({
        'AUC(0-500)': lambda x: x.replace('N/A', '0').astype(float).mean(),
        '测试损失': lambda x: x.astype(float).mean(),
        '训练时间(min)': lambda x: x.astype(float).mean(),
    })
    print(grouped)
    
    print()
    print("="*70)
    print("🎉 结果收集完成!")
    print("="*70)

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
光谱相似度评估脚本 - V0.8.8 预测结果
使用多种系数评估预测光谱与真实光谱的相似度

评估指标：
1. Pearson 相关系数
2. Spearman 相关系数
3. Simple Matching Score
4. RMSD (均方根偏差)
5. Euclidean Similarity (Grimme 文献)
6. Spectral Information Similarity (SIS)
7. Cosine Similarity
8. Mean Absolute Error (MAE)
"""

import sys
import math
import numpy as np
import pickle
import gzip
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import scipy.stats
from scipy.spatial.distance import cosine
from pathlib import Path

# ==================== 配置 ====================

class Config:
    # 当前文件所在目录 (flashair/)
    _CURRENT_DIR = Path(__file__).parent
    
    # V0.8.8 预测结果路径 (flashair/V0.8.8/data/)
    WORK_DIR = _CURRENT_DIR / 'V0.8.8'
    PREDICTION_DETAILS_FILE = WORK_DIR / 'data' / 'prediction_details.pkl.gz'
    
    # 输出目录 (flashair/V0.8.8/evaluation/)
    OUTPUT_DIR = WORK_DIR / 'evaluation'
    FIGURE_DIR = OUTPUT_DIR / 'figures'
    
    # 采样数量（设为 None 表示全部，设为数字表示随机采样）
    SAMPLE_SIZE = None  # 例如 10000
    
    # 随机种子
    RANDOM_SEED = 42


# ==================== 相似度函数 ====================
def pearson(u, v) -> float:
    """计算 Pearson 相关系数"""
    u, v = np.array(u), np.array(v)
    try:
        return scipy.stats.pearsonr(u, v)[0]
    except:
        return 0.0


def spearman(u, v) -> float:
    """计算 Spearman 相关系数"""
    u, v = np.array(u), np.array(v)
    try:
        return scipy.stats.spearmanr(u, v)[0]
    except:
        return 0.0


def simple_matching_score(u, v) -> float:
    """计算简单匹配分数"""
    u, v = np.array(u), np.array(v)
    numerator = np.square(np.sum(u * v))
    denominator = np.sum(np.square(u)) * np.sum(np.square(v))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def rmsd(u, v) -> float:
    """计算均方根偏差 (RMSD)"""
    u, v = np.array(u), np.array(v)
    if len(u) != len(v):
        raise ValueError("Input arrays must have same length")
    return math.sqrt(np.mean((u - v) ** 2))


def euclid_similarity(u, v) -> float:
    """计算 Grimme 文献中的欧氏相似度"""
    u, v = np.array(u), np.array(v)
    numerator = np.sum((u - v) ** 2)
    denominator = np.sum(v ** 2)
    if denominator == 0:
        return 0.0
    return (1 + numerator / denominator) ** -1


def spectral_info_similarity(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    计算 Spectral Information Similarity (SIS)
    SIS = 1 / (1 + SID), 其中 SID = D(p||q) + D(q||p)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    if p.shape != q.shape:
        raise ValueError(f"Input spectra must have same shape. Got {p.shape} vs {q.shape}")
    if (p < 0).any() or (q < 0).any():
        raise ValueError("Input spectra must be non-negative")
    
    # 归一化
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    p = p / p.sum()
    q = q / q.sum()
    
    # 计算对称散度
    with np.errstate(divide='ignore', invalid='ignore'):
        D_pq = np.sum(p * np.log(p / q))
        D_qp = np.sum(q * np.log(q / p))
    
    SID = np.nan_to_num(D_pq) + np.nan_to_num(D_qp)
    return 1 / (1 + SID)


def cosine_similarity(u, v) -> float:
    """计算余弦相似度"""
    u, v = np.array(u), np.array(v)
    try:
        return 1 - cosine(u, v)
    except:
        return 0.0


def mae(u, v) -> float:
    """计算平均绝对误差 (MAE)"""
    u, v = np.array(u), np.array(v)
    return np.mean(np.abs(u - v))


# ==================== 创建目录 ====================
def create_directories():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {Config.OUTPUT_DIR}")


# ==================== 加载数据 ====================
def load_predictions():
    """加载 V0.8.8 预测结果"""
    print(f"\n加载预测结果: {Config.PREDICTION_DETAILS_FILE}")
    
    if not Config.PREDICTION_DETAILS_FILE.exists():
        raise FileNotFoundError(f"预测详情文件不存在: {Config.PREDICTION_DETAILS_FILE}")
    
    with gzip.open(Config.PREDICTION_DETAILS_FILE, 'rb') as f:
        predictions = pickle.load(f)
    
    print(f"  加载了 {len(predictions)} 个分子的预测记录")
    
    # 筛选有预测结果且有效的分子
    valid = []
    for p in predictions:
        if p.get('pred_ir') is not None and p.get('true_ir') is not None:
            valid.append(p)
    
    print(f"  有效样本: {len(valid)} 个")
    
    # 采样
    if Config.SAMPLE_SIZE is not None and Config.SAMPLE_SIZE < len(valid):
        np.random.seed(Config.RANDOM_SEED)
        indices = np.random.choice(len(valid), Config.SAMPLE_SIZE, replace=False)
        valid = [valid[i] for i in indices]
        print(f"  随机采样: {len(valid)} 个")
    
    return valid


# ==================== 评估 ====================
def evaluate_predictions(predictions):
    """计算所有相似度指标"""
    print("\n计算相似度指标...")
    
    metric_functions = {
        'pearson': pearson,
        'spearman': spearman,
        'simple_matching': simple_matching_score,
        'rmsd': rmsd,
        'euclid_similarity': euclid_similarity,
        'sis': spectral_info_similarity,
        'cosine': cosine_similarity,
        'mae': mae,
    }
    
    metric_names = list(metric_functions.keys())
    
    # 初始化结果存储
    results = []
    level_metrics = defaultdict(lambda: defaultdict(list))
    
    total = len(predictions)
    
    for i, p in enumerate(predictions):
        if (i + 1) % 5000 == 0:
            print(f"  已处理: {i+1}/{total}")
        
        true_ir = np.array(p['true_ir'], dtype=np.float64)
        pred_ir = np.array(p['pred_ir'], dtype=np.float64)
        
        # 归一化（确保非负，用于 SIS）
        true_min, true_max = np.min(true_ir), np.max(true_ir)
        if true_max > true_min:
            true_norm = (true_ir - true_min) / (true_max - true_min)
        else:
            true_norm = true_ir
        
        pred_min, pred_max = np.min(pred_ir), np.max(pred_ir)
        if pred_max > pred_min:
            pred_norm = (pred_ir - pred_min) / (pred_max - pred_min)
        else:
            pred_norm = pred_ir
        
        # 计算所有指标
        metrics = {}
        for name, func in metric_functions.items():
            try:
                if name in ['rmsd', 'mae']:
                    # RMSD 和 MAE 使用原始值（非归一化）
                    val = func(true_ir, pred_ir)
                else:
                    val = func(true_norm, pred_norm)
                metrics[name] = val
            except Exception as e:
                metrics[name] = 0.0
        
        # 获取匹配级别
        match_level = p.get('match_level', 'Unknown')
        
        # 存储结果
        result = {
            'smiles': p.get('smiles', 'Unknown'),
            'match_level': match_level,
            'matched_smiles': p.get('matched_smiles', 'Unknown'),
            'match_similarity': p.get('match_similarity', 0.0),
            'fold': p.get('fold', 0),
            'num_fragments': len(p.get('fragments', [])),
            'pred_time_ms': p.get('pred_time_ms', 0),
        }
        result.update(metrics)
        results.append(result)
        
        # 按级别统计
        for name, val in metrics.items():
            level_metrics[match_level][name].append(val)
    
    return results, level_metrics, metric_names


# ==================== 统计输出 ====================
def print_statistics(results, level_metrics, metric_names):
    """打印统计结果"""
    print("\n" + "=" * 80)
    print("评估统计 (V0.8.8)")
    print("=" * 80)
    
    # 整体统计
    print("\n整体统计（所有分子）:")
    print("-" * 70)
    
    # 定义显示顺序和格式
    display_metrics = ['pearson', 'spearman', 'simple_matching', 'cosine', 'euclid_similarity', 'sis', 'rmsd', 'mae']
    display_names = {
        'pearson': 'Pearson',
        'spearman': 'Spearman',
        'simple_matching': 'Simple Matching',
        'cosine': 'Cosine Similarity',
        'euclid_similarity': 'Euclidean Similarity',
        'sis': 'SIS',
        'rmsd': 'RMSD',
        'mae': 'MAE',
    }
    
    for metric in display_metrics:
        if metric not in metric_names:
            continue
        values = [r[metric] for r in results if r.get(metric) is not None]
        if values:
            print(f"  {display_names.get(metric, metric)}:")
            print(f"    平均: {np.mean(values):.4f}")
            print(f"    中位数: {np.median(values):.4f}")
            print(f"    标准差: {np.std(values):.4f}")
            print(f"    最小: {np.min(values):.4f}")
            print(f"    最大: {np.max(values):.4f}")
    
    # 按匹配级别统计
    print("\n按匹配级别统计:")
    print("-" * 80)
    
    levels = sorted(level_metrics.keys())
    
    # 表头
    header = f"{'级别':<15}"
    for metric in ['pearson', 'spearman', 'simple_matching', 'sis', 'rmsd']:
        if metric in metric_names:
            header += f" {metric:>14}"
    print(header)
    print("-" * 80)
    
    for level in levels:
        if level not in level_metrics:
            continue
        row = f"{level:<15}"
        for metric in ['pearson', 'spearman', 'simple_matching', 'sis', 'rmsd']:
            if metric not in metric_names:
                continue
            values = level_metrics[level].get(metric, [])
            if values:
                row += f" {np.mean(values):>14.4f}"
            else:
                row += f" {'N/A':>14}"
        print(row)


# ==================== 绘图 ====================
def plot_all_metrics(results, metric_names):
    """绘制所有指标分布图"""
    print("\n绘制指标分布图...")
    
    display_metrics = ['pearson', 'spearman', 'simple_matching', 'cosine', 'euclid_similarity', 'sis', 'rmsd', 'mae']
    display_names = {
        'pearson': 'Pearson',
        'spearman': 'Spearman',
        'simple_matching': 'Simple Matching',
        'cosine': 'Cosine Similarity',
        'euclid_similarity': 'Euclidean Similarity',
        'sis': 'SIS',
        'rmsd': 'RMSD',
        'mae': 'MAE',
    }
    
    # 筛选存在的指标
    plot_metrics = [m for m in display_metrics if m in metric_names]
    n_metrics = len(plot_metrics)
    
    if n_metrics == 0:
        print("  没有可绘制的指标")
        return
    
    # 计算子图布局
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": 'sans-serif',
        "font.sans-serif": ["Arial"],
        "axes.linewidth": 0.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), dpi=300)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_metrics == 1 else axes
    
    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]
        values = [r[metric] for r in results if r.get(metric) is not None]
        
        if not values:
            ax.text(0.5, 0.5, f'No data', ha='center', va='center')
            ax.set_title(display_names.get(metric, metric))
            continue
        
        ax.hist(values, bins=50, color='salmon', edgecolor='w', alpha=0.7)
        ax.axvline(np.mean(values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(values):.3f}', linewidth=1.5)
        ax.axvline(np.median(values), color='blue', linestyle='--', 
                   label=f'Median: {np.median(values):.3f}', linewidth=1.5)
        ax.set_xlabel(display_names.get(metric, metric))
        ax.set_ylabel('Count')
        ax.set_title(f'{display_names.get(metric, metric)} Distribution')
        ax.legend(fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_path = Config.FIGURE_DIR / 'all_metrics_distribution_v088.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  分布图已保存: {save_path}")


def plot_level_comparison(results, level_metrics, metric_names):
    """绘制按匹配级别对比图"""
    print("绘制级别对比图...")
    
    main_metrics = ['pearson', 'spearman', 'simple_matching', 'sis']
    main_metrics = [m for m in main_metrics if m in metric_names]
    
    if not main_metrics:
        print("  没有可绘制的指标")
        return
    
    # 获取级别
    levels = sorted(level_metrics.keys())
    
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": 'sans-serif',
        "font.sans-serif": ["Arial"],
        "axes.linewidth": 0.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    
    n_metrics = len(main_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), dpi=300)
    if n_rows == 1:
        axes = [axes] if n_metrics == 1 else axes
    else:
        axes = axes.flatten()
    
    metric_names_display = {
        'pearson': 'Pearson Correlation',
        'spearman': 'Spearman Correlation',
        'simple_matching': 'Simple Matching Score',
        'sis': 'Spectral Information Similarity',
    }
    
    for idx, metric in enumerate(main_metrics):
        ax = axes[idx]
        
        data = []
        labels = []
        for level in levels:
            values = level_metrics[level].get(metric, [])
            if values:
                data.append(values)
                labels.append(f"{level}\n(n={len(values)})")
        
        if not data:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center')
            ax.set_title(metric_names_display.get(metric, metric))
            continue
        
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, 
                        showfliers=True, widths=0.6)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#F7DC6F']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Match Level')
        ax.set_ylabel(metric_names_display.get(metric, metric))
        ax.set_title(f'{metric_names_display.get(metric, metric)} by Match Level')
        ax.grid(True, alpha=0.3)
        if metric != 'rmsd':
            ax.set_ylim(0, 1)
    
    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_path = Config.FIGURE_DIR / 'metrics_by_level_v088.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  级别对比图已保存: {save_path}")


def plot_correlation_heatmap(results, metric_names):
    """绘制指标相关性热图"""
    print("绘制相关性热图...")
    
    display_metrics = ['pearson', 'spearman', 'simple_matching', 'cosine', 'euclid_similarity', 'sis', 'rmsd', 'mae']
    plot_metrics = [m for m in display_metrics if m in metric_names]
    
    if len(plot_metrics) < 2:
        print("  指标太少，无法绘制热图")
        return
    
    # 提取数据
    data = {}
    for metric in plot_metrics:
        data[metric] = [r[metric] for r in results if r.get(metric) is not None]
    
    # 确保所有指标长度一致
    min_len = min(len(v) for v in data.values())
    for metric in plot_metrics:
        data[metric] = data[metric][:min_len]
    
    # 计算相关性矩阵
    n = len(plot_metrics)
    corr_matrix = np.zeros((n, n))
    for i, m1 in enumerate(plot_metrics):
        for j, m2 in enumerate(plot_metrics):
            corr_matrix[i, j] = np.corrcoef(data[m1], data[m2])[0, 1]
    
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": 'sans-serif',
        "font.sans-serif": ["Arial"],
        "axes.linewidth": 0.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(plot_metrics, rotation=45, ha='right')
    ax.set_yticklabels(plot_metrics)
    
    # 添加数值
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center', 
                          color='black' if abs(corr_matrix[i, j]) < 0.5 else 'white',
                          fontsize=9)
    
    ax.set_title('Correlation Matrix of Similarity Metrics (V0.8.8)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    save_path = Config.FIGURE_DIR / 'metric_correlation_v088.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  相关性热图已保存: {save_path}")


# ==================== 保存结果 ====================
def save_results(results, level_metrics, metric_names):
    """保存评估结果"""
    print("\n保存评估结果...")
    
    # 保存详细结果
    with open(Config.OUTPUT_DIR / 'evaluation_results_v088.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'level_metrics': dict(level_metrics),
            'metric_names': metric_names,
            'config': {
                'sample_size': Config.SAMPLE_SIZE,
                'random_seed': Config.RANDOM_SEED,
            }
        }, f)
    print(f"  评估结果已保存: {Config.OUTPUT_DIR / 'evaluation_results_v088.pkl'}")
    
    # 保存统计摘要
    summary = {'overall': {}, 'by_level': {}}
    
    display_metrics = ['pearson', 'spearman', 'simple_matching', 'cosine', 'euclid_similarity', 'sis', 'rmsd', 'mae']
    for metric in display_metrics:
        if metric not in metric_names:
            continue
        values = [r[metric] for r in results if r.get(metric) is not None]
        if values:
            summary['overall'][metric] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values),
            }
    
    for level, metrics in level_metrics.items():
        summary['by_level'][level] = {}
        for metric, values in metrics.items():
            if values:
                summary['by_level'][level][metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'count': len(values),
                }
    
    with open(Config.OUTPUT_DIR / 'summary_v088.pkl', 'wb') as f:
        pickle.dump(summary, f)
    print(f"  统计摘要已保存: {Config.OUTPUT_DIR / 'summary_v088.pkl'}")
    
    # 导出 CSV
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(Config.OUTPUT_DIR / 'evaluation_results_v088.csv', index=False)
        print(f"  CSV已保存: {Config.OUTPUT_DIR / 'evaluation_results_v088.csv'}")
    except ImportError:
        print("  (pandas未安装，跳过CSV导出)")


# ==================== 主程序 ====================
def main():
    print("=" * 80)
    print("光谱相似度评估 - V0.8.8 预测结果")
    print("=" * 80)
    
    create_directories()
    
    # 1. 加载预测结果
    predictions = load_predictions()
    
    # 2. 计算所有指标
    results, level_metrics, metric_names = evaluate_predictions(predictions)
    
    # 3. 打印统计
    print_statistics(results, level_metrics, metric_names)
    
    # 4. 绘图
    plot_all_metrics(results, metric_names)
    plot_level_comparison(results, level_metrics, metric_names)
    plot_correlation_heatmap(results, metric_names)
    
    # 5. 保存结果
    save_results(results, level_metrics, metric_names)
    
    print("\n✅ 评估完成！")
    print(f"输出目录: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
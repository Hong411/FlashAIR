# -*- coding: utf-8 -*-
"""
FlashAIRx: Validation Engine for FlashAIR
Physics-based IR spectrum correction using GFN2-xTB prior + GPR residual

This module implements the FlashAIRx validation engine described in:
"FlashAIR: Fast and Interpretable IR Spectral Analysis via Physical 
Prior and Residual Correction"

Author: FlashAIR Team
"""

import sqlite3
import json
import numpy as np
import scipy.stats
import tensorflow as tf
import gpflow
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
from pathlib import Path
CURRENT_DIR = Path(__file__).parent
DATABASE_PATH = CURRENT_DIR.parent / 'data' / 'FlashAIR-QM9d.db'

RANDOM_STATE = 23
TEST_SIZE = 0.3
MIN_SAMPLES = 10
GP_ITERATIONS = 1000

# ============ 工具函数 ============

def sqrt_norm(data):
    """平方根归一化"""
    arr = np.array(data, dtype=np.float64)
    sqrt_arr = np.sqrt(np.maximum(arr, 0))
    min_val, max_val = np.min(sqrt_arr), np.max(sqrt_arr)
    if max_val - min_val < 1e-12:
        return np.zeros_like(arr).tolist()
    return ((sqrt_arr - min_val) / (max_val - min_val)).tolist()


def cal_pearson(u, v):
    """Pearson相关系数"""
    try:
        return scipy.stats.pearsonr(u, v)[0]
    except:
        return np.nan


def cal_spearman(u, v):
    """Spearman秩相关系数"""
    try:
        return scipy.stats.spearmanr(u, v)[0]
    except:
        return np.nan


def spectral_info_similarity(p, q, epsilon=1e-10):
    """光谱信息相似度 SIS = 1/(1+SID)"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    p = p / np.sum(p)
    q = q / np.sum(q)
    with np.errstate(divide='ignore', invalid='ignore'):
        D_pq = np.sum(p * np.log(p / q))
        D_qp = np.sum(q * np.log(q / p))
    SID = np.nan_to_num(D_pq) + np.nan_to_num(D_qp)
    return 1 / (1 + SID)


def parse_fg_string(fg_str):
    """
    解析官能团字符串，返回排序后的元组（顺序无关）
    输入: '["Alkane", "Aromatic"]' 或 '["Alkane"]'
    输出: ('Alkane', 'Aromatic')
    """
    try:
        fg_list = json.loads(fg_str)
        if isinstance(fg_list, list):
            return tuple(sorted(fg_list))
    except:
        pass
    return ()


def get_fg_count(fg_tuple):
    """获取官能团数量"""
    return len(fg_tuple)


# ============ 数据库操作 ============

def get_all_molecules(db_path):
    """
    获取所有分子的ID和官能团组合
    返回: dict {mol_id: fg_tuple}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT mol_id, mol_functional_groups FROM molecular_information;")
    rows = cursor.fetchall()
    conn.close()
    
    result = {}
    for mol_id, fg_str in rows:
        fg_tuple = parse_fg_string(fg_str)
        if fg_tuple:  # 只保留有官能团标签的分子
            result[mol_id] = fg_tuple
    
    return result


def get_spectra(db_path, mol_id):
    """
    获取指定分子的DFT和XTB光谱
    返回: (dft_intensity, xtb_intensity) 已归一化
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取DFT光谱 (sim_spectrum)
    cursor.execute("SELECT intensity FROM sim_spectrum WHERE mol_id = ?;", (mol_id,))
    dft_row = cursor.fetchone()
    
    # 获取XTB光谱 (xtb_spectrum, 列名是id)
    cursor.execute("SELECT intensity FROM xtb_spectrum WHERE id = ?;", (mol_id,))
    xtb_row = cursor.fetchone()
    
    conn.close()
    
    if dft_row is None or xtb_row is None:
        return None, None
    
    dft_intensity = json.loads(dft_row[0])
    xtb_intensity = json.loads(xtb_row[0])
    
    # 确保长度一致
    if len(dft_intensity) != len(xtb_intensity):
        return None, None
    
    # sqrt_norm归一化
    dft_norm = sqrt_norm(dft_intensity)
    xtb_norm = sqrt_norm(xtb_intensity)
    
    return dft_norm, xtb_norm


def get_group_data(db_path, fg_tuple, mol_fg_dict):
    """
    获取指定官能团组合的所有分子数据
    返回: (X_list, y_list, mol_ids)
    """
    X_list, y_list, mol_ids = [], [], []
    
    for mol_id, fg in mol_fg_dict.items():
        if fg == fg_tuple:
            dft, xtb = get_spectra(db_path, mol_id)
            if dft is not None and xtb is not None:
                X_list.append(xtb)
                y_list.append(dft)
                mol_ids.append(mol_id)
    
    return X_list, y_list, mol_ids


# ============ GPR模型 (GPflow) ============

def train_gpr_model(X_train, y_train):
    """使用GPflow训练GPR模型"""
    X_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
    y_tf = tf.convert_to_tensor(y_train, dtype=tf.float64)
    
    kernel = gpflow.kernels.SquaredExponential(lengthscales=1.0, variance=1.0)
    gpr = gpflow.models.GPR(data=(X_tf, y_tf), kernel=kernel)
    gpr.likelihood.variance.assign(1e-3)
    gpflow.utilities.set_trainable(gpr.likelihood.variance, False)
    
    optimizer = tf.optimizers.Adam()
    
    @tf.function
    def optimization_step():
        with tf.GradientTape() as tape:
            loss = gpr.training_loss()
        gradients = tape.gradient(loss, gpr.trainable_variables)
        optimizer.apply_gradients(zip(gradients, gpr.trainable_variables))
    
    for _ in range(GP_ITERATIONS):
        optimization_step()
    
    return gpr


def predict_gpr_model(model, X_test):
    """使用GPflow模型预测"""
    X_tf = tf.convert_to_tensor(X_test, dtype=tf.float64)
    mean, _ = model.predict_f(X_tf)
    return mean.numpy()


# ============ 评估函数 ============

def evaluate_predictions(y_true, y_pred):
    """计算PCC, Spearman, SIS"""
    pcc_list, spearman_list, sis_list = [], [], []
    
    for true, pred in zip(y_true, y_pred):
        pcc = cal_pearson(true, pred)
        spearman = cal_spearman(true, pred)
        sis = spectral_info_similarity(true, pred)
        
        if not np.isnan(pcc):
            pcc_list.append(pcc)
            spearman_list.append(spearman)
            sis_list.append(sis)
    
    return {
        'pcc': np.mean(pcc_list) if pcc_list else np.nan,
        'spearman': np.mean(spearman_list) if spearman_list else np.nan,
        'sis': np.mean(sis_list) if sis_list else np.nan,
        'n_samples': len(pcc_list)
    }


# ============ 主流程 ============

def main():
    print("=" * 60)
    print("XTB → DFT 光谱校正 GPR 模型训练")
    print("=" * 60)
    
    # 1. 获取所有分子信息
    print("\n[1] 加载分子数据...")
    mol_fg_dict = get_all_molecules(DATABASE_PATH)
    print(f"    共 {len(mol_fg_dict)} 个有官能团标签的分子")
    
    # 2. 按官能团数量分类
    fg_by_count = {1: {}, 2: {}, 3: {}, '4+': {}}
    
    for mol_id, fg_tuple in mol_fg_dict.items():
        count = len(fg_tuple)
        if count == 1:
            fg_by_count[1][mol_id] = fg_tuple
        elif count == 2:
            fg_by_count[2][mol_id] = fg_tuple
        elif count == 3:
            fg_by_count[3][mol_id] = fg_tuple
        else:
            fg_by_count['4+'][mol_id] = fg_tuple
    
    print(f"    单官能团: {len(fg_by_count[1])} 个分子")
    print(f"    双官能团: {len(fg_by_count[2])} 个分子")
    print(f"    三官能团: {len(fg_by_count[3])} 个分子")
    print(f"    四官能团+: {len(fg_by_count['4+'])} 个分子")
    
    # 3. 按官能团组合分组
    print("\n[2] 按官能团组合分组...")
    
    group_dicts = {1: {}, 2: {}, 3: {}, '4+': {}}
    
    for count_key, mol_dict in fg_by_count.items():
        # 统计各组分子数
        fg_groups = {}
        for mol_id, fg_tuple in mol_dict.items():
            if fg_tuple not in fg_groups:
                fg_groups[fg_tuple] = []
            fg_groups[fg_tuple].append(mol_id)
        
        for fg_tuple, mol_ids in fg_groups.items():
            if len(mol_ids) >= MIN_SAMPLES:
                group_dicts[count_key][fg_tuple] = mol_ids
                print(f"    {fg_tuple}: {len(mol_ids)} 个分子")
    
    # 4. 训练模型 + 评估
    print("\n[3] 训练GPR模型并评估...")
    
    models = {1: {}, 2: {}, 3: {}, '4+': {}}
    test_results = {1: {}, 2: {}, 3: {}, '4+': {}}
    all_metrics = {'pcc': [], 'spearman': [], 'sis': []}
    group_summary = {1: [], 2: [], 3: [], '4+': []}
    
    for count_key, fg_dict in group_dicts.items():
        print(f"\n    --- 官能团数量: {count_key} ---")
        
        for fg_tuple, mol_ids in fg_dict.items():
            # 获取数据
            X, y, ids = [], [], []
            for mol_id in mol_ids:
                dft, xtb = get_spectra(DATABASE_PATH, mol_id)
                if dft is not None and xtb is not None:
                    X.append(xtb)
                    y.append(dft)
                    ids.append(mol_id)
            
            if len(X) < MIN_SAMPLES:
                continue
            
            # 划分训练/测试集
            X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
                X, y, ids, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            
            # 训练模型
            model = train_gpr_model(X_train, y_train)
            models[count_key][fg_tuple] = model
            
            # 预测
            y_pred = predict_gpr_model(model, X_test)
            
            # 评估
            metrics = evaluate_predictions(y_test, y_pred)
            metrics['fg'] = fg_tuple
            metrics['n_train'] = len(X_train)
            metrics['n_test'] = len(X_test)
            
            group_summary[count_key].append(metrics)
            all_metrics['pcc'].append(metrics['pcc'])
            all_metrics['spearman'].append(metrics['spearman'])
            all_metrics['sis'].append(metrics['sis'])
            
            # 保存测试结果
            test_results[count_key][fg_tuple] = {
                'xtb_spectra': X_test,
                'dft_spectra': y_test,
                'pred_spectra': y_pred,
                'mol_ids': ids_test,
                'metrics': metrics
            }
            
            print(f"      {fg_tuple}: PCC={metrics['pcc']:.4f}, Spearman={metrics['spearman']:.4f}, SIS={metrics['sis']:.4f} (n={len(X_test)})")
    
    # 5. 输出统计
    print("\n" + "=" * 60)
    print("[4] 评估结果汇总")
    print("=" * 60)
    
    # 按官能团数量输出平均指标
    print("\n按官能团数量分组统计:")
    for count_key in [1, 2, 3, '4+']:
        if group_summary[count_key]:
            pccs = [m['pcc'] for m in group_summary[count_key] if not np.isnan(m['pcc'])]
            sps = [m['spearman'] for m in group_summary[count_key] if not np.isnan(m['spearman'])]
            siss = [m['sis'] for m in group_summary[count_key] if not np.isnan(m['sis'])]
            print(f"\n  FG={count_key}: {len(group_summary[count_key])} 个官能团组")
            print(f"    PCC 平均: {np.mean(pccs):.4f} ± {np.std(pccs):.4f}")
            print(f"    Spearman 平均: {np.mean(sps):.4f} ± {np.std(sps):.4f}")
            print(f"    SIS 平均: {np.mean(siss):.4f} ± {np.std(siss):.4f}")
    
    # 总体平均
    print("\n总体统计:")
    print(f"  PCC 平均: {np.mean(all_metrics['pcc']):.4f} ± {np.std(all_metrics['pcc']):.4f}")
    print(f"  Spearman 平均: {np.mean(all_metrics['spearman']):.4f} ± {np.std(all_metrics['spearman']):.4f}")
    print(f"  SIS 平均: {np.mean(all_metrics['sis']):.4f} ± {np.std(all_metrics['sis']):.4f}")
    
    # 6. 保存结果
    print("\n[5] 保存结果...")
    
    # 保存模型字典
    with open('gpr_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("    模型已保存: gpr_models.pkl")
    
    # 保存测试结果
    with open('test_results.pkl', 'wb') as f:
        pickle.dump(test_results, f)
    print("    测试结果已保存: test_results.pkl")
    
    # 保存汇总指标
    with open('summary_metrics.pkl', 'wb') as f:
        pickle.dump({
            'by_group': group_summary,
            'overall': {
                'pcc': np.mean(all_metrics['pcc']),
                'spearman': np.mean(all_metrics['spearman']),
                'sis': np.mean(all_metrics['sis'])
            }
        }, f)
    print("    汇总指标已保存: summary_metrics.pkl")
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


def load_models():
    """加载保存的模型字典"""
    with open('gpr_models.pkl', 'rb') as f:
        return pickle.load(f)


def load_test_results():
    """加载测试结果"""
    with open('test_results.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
"""
FlashAIRx 光谱匹配与解析模块
功能:
  1. 构建预测光谱库（最多10组，支持1组最小）
  2. 用户光谱加载与预处理（格式: csv/txt，第一行标题，两列: 波数, 强度）
  3. 单光谱相似度计算 (PCC, SIS, SRC)
  4. 混合光谱相似度计算（仅支持两组混合，需要≥2组预测光谱）
  5. 高置信度匹配判定 (三项指标>0.9)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
from itertools import combinations


# ============ 配置 ============
SPECTRUM_START = 546
SPECTRUM_END = 3846
SPECTRUM_POINTS = 825
HIGH_CONFIDENCE_THRESHOLD = 0.9
MAX_LIBRARY_SIZE = 10
MIXTURE_STEP = 0.1
MIXTURE_REFINE_STEP = 0.01
MIXTURE_REFINE_THRESHOLD = 0.85


# ============ 相似度计算函数 ============

def cal_pearson(u: np.ndarray, v: np.ndarray) -> float:
    """计算Pearson相关系数"""
    try:
        return pearsonr(u, v)[0]
    except:
        return np.nan


def cal_spearman(u: np.ndarray, v: np.ndarray) -> float:
    """计算Spearman秩相关系数"""
    try:
        return spearmanr(u, v)[0]
    except:
        return np.nan


def spectral_info_similarity(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    计算光谱信息相似度 SIS = 1/(1+SID)
    """
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


def sqrt_norm(data: Union[List, np.ndarray]) -> np.ndarray:
    """平方根归一化"""
    arr = np.array(data, dtype=np.float64)
    min_val = np.sqrt(np.min(arr))
    max_val = np.sqrt(np.max(arr))
    if max_val - min_val < 1e-12:
        return np.zeros_like(arr)
    return (np.sqrt(arr) - min_val) / (max_val - min_val)


def normalize_spectrum_to_wavenumber(
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    target_start: float = SPECTRUM_START,
    target_end: float = SPECTRUM_END,
    target_points: int = SPECTRUM_POINTS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将光谱插值到标准波数范围
    
    返回:
        (target_wavenumber, interpolated_intensity)
    """
    if len(wavenumber) == 0 or len(intensity) == 0:
        return np.array([]), np.array([])
    
    target_wavenumber = np.linspace(target_start, target_end, target_points)
    
    # 检查是否已在标准范围
    if (np.isclose(wavenumber[0], target_start) and 
        np.isclose(wavenumber[-1], target_end) and 
        len(wavenumber) == target_points):
        return target_wavenumber, np.array(intensity)
    
    f_interp = interp1d(
        wavenumber,
        intensity,
        kind='linear',
        fill_value=0,
        bounds_error=False
    )
    interpolated = f_interp(target_wavenumber)
    
    return target_wavenumber, interpolated


def calculate_similarity(spectrum1: np.ndarray, spectrum2: np.ndarray) -> Dict:
    """
    计算两个光谱之间的相似度指标
    
    参数:
        spectrum1: 归一化光谱1
        spectrum2: 归一化光谱2
    
    返回:
        {
            'pcc': float,
            'sis': float,
            'src': float
        }
    """
    spec1 = np.array(spectrum1, dtype=np.float64).flatten()
    spec2 = np.array(spectrum2, dtype=np.float64).flatten()
    
    # 确保长度一致
    min_len = min(len(spec1), len(spec2))
    spec1 = spec1[:min_len]
    spec2 = spec2[:min_len]
    
    return {
        'pcc': cal_pearson(spec1, spec2),
        'sis': spectral_info_similarity(spec1, spec2),
        'src': cal_spearman(spec1, spec2)
    }


def is_high_confidence(similarity: Dict, threshold: float = HIGH_CONFIDENCE_THRESHOLD) -> bool:
    """判断是否高置信度匹配 (三项指标均>threshold)"""
    return (similarity.get('pcc', 0) > threshold and 
            similarity.get('sis', 0) > threshold and 
            similarity.get('src', 0) > threshold)


def format_similarity(similarity: Dict, indent: int = 0) -> str:
    """格式化输出相似度"""
    prefix = "  " * indent
    lines = [
        f"{prefix}PCC:      {similarity.get('pcc', 0):.4f}",
        f"{prefix}SIS:      {similarity.get('sis', 0):.4f}",
        f"{prefix}SRC:      {similarity.get('src', 0):.4f}"
    ]
    if is_high_confidence(similarity):
        lines.append(f"{prefix}✓ 高置信度匹配!")
    return "\n".join(lines)


# ============ 光谱库管理 ============

class SpectrumLibrary:
    """
    预测光谱库，最多保存10组光谱
    轮换存储，超过10组时覆盖最旧的
    """
    
    def __init__(self, max_size: int = MAX_LIBRARY_SIZE):
        self.max_size = max_size
        self.entries = []  # 按添加顺序存储
        self.wavenumber = None
    
    def add_entry(self, entry: Dict) -> bool:
        """
        添加一个光谱条目
        
        参数:
            entry: {
                'smiles': str,
                'name': str,
                'spectrum': np.ndarray,  # 归一化光谱
                'dft_spectrum': np.ndarray,  # DFT光谱(可选)
                'exp_spectrum': np.ndarray,  # EXP光谱(可选)
                'functional_groups': list,
                'wavenumber': np.ndarray
            }
        
        返回:
            bool: 添加是否成功
        """
        # 验证必要字段
        if 'spectrum' not in entry:
            return False
        
        # 标准化波数
        if self.wavenumber is None:
            self.wavenumber = entry.get('wavenumber')
        
        # 添加到列表
        self.entries.append(entry)
        
        # 超过最大容量，移除最旧的
        if len(self.entries) > self.max_size:
            self.entries.pop(0)
        
        return True
    
    def add_entries(self, entries: List[Dict]) -> int:
        """批量添加"""
        added = 0
        for entry in entries:
            if self.add_entry(entry):
                added += 1
        return added
    
    def clear(self):
        """清空光谱库"""
        self.entries = []
        self.wavenumber = None
    
    def get_size(self) -> int:
        """获取当前光谱库大小"""
        return len(self.entries)
    
    def get_all_spectra(self) -> List[np.ndarray]:
        """获取所有光谱"""
        return [e['spectrum'] for e in self.entries]
    
    def get_all_info(self) -> List[Dict]:
        """获取所有光谱信息"""
        return self.entries
    
    def get_entry(self, index: int) -> Optional[Dict]:
        """获取指定索引的条目"""
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None
    
    def get_spectrum(self, index: int) -> Optional[np.ndarray]:
        """获取指定索引的光谱"""
        entry = self.get_entry(index)
        return entry['spectrum'] if entry else None
    
    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            'max_size': self.max_size,
            'size': len(self.entries),
            'wavenumber': self.wavenumber.tolist() if self.wavenumber is not None else None,
            'entries': [
                {
                    'smiles': e.get('smiles', ''),
                    'name': e.get('name', f'Entry_{i}'),
                    'spectrum': e['spectrum'].tolist(),
                    'functional_groups': e.get('functional_groups', [])
                }
                for i, e in enumerate(self.entries)
            ]
        }
    
    def save(self, file_path: Union[str, Path]):
        """保存到文件"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load(self, file_path: Union[str, Path]):
        """从文件加载"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.max_size = data.get('max_size', MAX_LIBRARY_SIZE)
        self.entries = []
        
        for e in data.get('entries', []):
            entry = {
                'smiles': e.get('smiles', ''),
                'name': e.get('name', ''),
                'spectrum': np.array(e.get('spectrum', [])),
                'functional_groups': e.get('functional_groups', [])
            }
            if len(entry['spectrum']) > 0:
                self.entries.append(entry)
        
        if data.get('wavenumber'):
            self.wavenumber = np.array(data.get('wavenumber'))


# ============ 用户光谱加载 ============

def load_user_spectrum(file_path: Union[str, Path]) -> Dict:
    """
    加载用户光谱文件
    
    格式要求:
        - txt 或 csv 格式
        - 第一行是标题
        - 两列: 波数, 强度
        - 示例:
            Wavenumber,Intensity
            550.0,0.0012
            554.0,0.0015
            ...
    
    参数:
        file_path: 文件路径
    
    返回:
        {
            'success': bool,
            'error': str or None,
            'wavenumber': np.ndarray,
            'intensity': np.ndarray,
            'intensity_norm': np.ndarray,  # sqrt_norm归一化
        }
    """
    result = {
        'success': False,
        'error': None,
        'wavenumber': None,
        'intensity': None,
        'intensity_norm': None
    }
    
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            result['error'] = f"文件不存在: {file_path}"
            return result
        
        # 读取文件
        df = pd.read_csv(file_path, header=0)
        
        # 检查列数
        if df.shape[1] < 2:
            result['error'] = f"文件格式错误: 需要至少2列 (波数, 强度)，当前有 {df.shape[1]} 列"
            return result
        
        # 提取波数和强度
        wavenumber = df.iloc[:, 0].values
        intensity = df.iloc[:, 1].values
        
        # 检查数据有效性
        if len(wavenumber) == 0:
            result['error'] = "文件为空"
            return result
        
        if np.any(np.isnan(wavenumber)) or np.any(np.isnan(intensity)):
            result['error'] = "数据包含NaN值"
            return result
        
        # 检查波数是否单调递增
        if not np.all(np.diff(wavenumber) > 0):
            result['error'] = "波数应单调递增"
            return result
        
        # 插值到标准波数范围
        target_wavenumber, normalized_intensity = normalize_spectrum_to_wavenumber(
            wavenumber, intensity
        )
        
        if len(target_wavenumber) == 0:
            result['error'] = "光谱插值失败"
            return result
        
        # sqrt_norm归一化
        intensity_norm = sqrt_norm(normalized_intensity)
        
        result['success'] = True
        result['wavenumber'] = target_wavenumber
        result['intensity'] = normalized_intensity
        result['intensity_norm'] = intensity_norm
        
        return result
        
    except pd.errors.EmptyDataError:
        result['error'] = "文件为空"
        return result
    except pd.errors.ParserError as e:
        result['error'] = f"文件解析错误: {str(e)}"
        return result
    except Exception as e:
        result['error'] = f"加载文件时出错: {str(e)}"
        return result


# ============ 混合光谱生成 ============

def generate_mixture_spectrum(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    ratio: float
) -> np.ndarray:
    """
    生成两组光谱的混合光谱
    
    参数:
        spectrum1: 光谱1
        spectrum2: 光谱2
        ratio: 光谱1的比例 (0-1)，光谱2的比例为 1-ratio
    
    返回:
        混合后的光谱 (已重新归一化)
    """
    if ratio < 0 or ratio > 1:
        raise ValueError(f"比例必须在0-1之间，当前: {ratio}")
    
    spec1 = np.array(spectrum1, dtype=np.float64)
    spec2 = np.array(spectrum2, dtype=np.float64)
    
    # 确保长度一致
    min_len = min(len(spec1), len(spec2))
    spec1 = spec1[:min_len]
    spec2 = spec2[:min_len]
    
    # 加权求和
    mixed = spec1 * ratio + spec2 * (1 - ratio)
    
    # 重新归一化
    mixed_norm = sqrt_norm(mixed)
    
    return mixed_norm


def generate_mixture_ratios(step: float = MIXTURE_STEP) -> List[float]:
    """
    生成混合比例列表 (0.1 到 0.9)
    """
    ratios = []
    current = step
    while current < 1.0:
        ratios.append(round(current, 2))
        current += step
    return ratios


def refine_mixture_ratios(center_ratio: float, step: float = MIXTURE_REFINE_STEP) -> List[float]:
    """
    在中心比例附近生成精细比例
    """
    ratios = []
    for offset in [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]:
        r = center_ratio + offset
        if 0.01 <= r <= 0.99:
            ratios.append(round(r, 2))
    return list(set(ratios))


# ============ 光谱匹配器 ============

class SpectralMatcher:
    """
    光谱匹配器
    
    功能:
        1. 管理光谱库
        2. 单光谱匹配
        3. 两组混合光谱匹配
        4. 高置信度判定
    """
    
    def __init__(self, max_library_size: int = MAX_LIBRARY_SIZE):
        self.library = SpectrumLibrary(max_library_size)
        self.predictor = None  # FlashAIRxPredictor实例
    
    def set_predictor(self, predictor):
        """设置预测器"""
        self.predictor = predictor
    
    def add_molecule(
        self,
        smiles: str,
        name: str = None,
        functional_groups: List = None,
        calibrate_to_exp: bool = False
    ) -> Dict:
        """
        添加一个分子到光谱库
        
        参数:
            smiles: SMILES字符串
            name: 分子名称
            functional_groups: 官能团列表
            calibrate_to_exp: 是否使用EXP光谱
        
        返回:
            {
                'success': bool,
                'error': str or None,
                'entry': dict or None
            }
        """
        result = {
            'success': False,
            'error': None,
            'entry': None
        }
        
        if self.predictor is None:
            result['error'] = "未设置预测器"
            return result
        
        # 获取预处理和预测
        preprocess_result = self.predictor.preprocess(smiles)
        if not preprocess_result['success']:
            result['error'] = f"预处理失败: {preprocess_result['error']}"
            return result
        
        # 获取XTB光谱和官能团
        xtb_spectrum = preprocess_result['gpr_input']['X']
        functional_groups = preprocess_result['gpr_input']['functional_groups']
        
        # 预测DFT
        predict_result = self.predictor.predict(
            xtb_spectrum=xtb_spectrum,
            functional_groups=functional_groups,
            calibrate_to_exp=calibrate_to_exp
        )
        
        if not predict_result['success']:
            result['error'] = f"预测失败: {predict_result['error']}"
            return result
        
        # 选择光谱 (优先EXP)
        if calibrate_to_exp and predict_result.get('exp_spectrum') is not None:
            spectrum = predict_result['exp_spectrum']
        else:
            spectrum = predict_result['dft_spectrum']
        
        # 归一化
        spectrum_norm = sqrt_norm(spectrum)
        
        # 构建条目
        entry = {
            'smiles': smiles,
            'name': name or smiles,
            'spectrum': spectrum_norm,
            'dft_spectrum': predict_result.get('dft_spectrum'),
            'exp_spectrum': predict_result.get('exp_spectrum'),
            'functional_groups': functional_groups,
            'wavenumber': predict_result.get('wavenumber')
        }
        
        # 添加到库
        self.library.add_entry(entry)
        
        result['success'] = True
        result['entry'] = entry
        return result
    
    def add_molecules(
        self,
        smiles_list: List[str],
        names: List[str] = None,
        calibrate_to_exp: bool = False
    ) -> Dict:
        """
        批量添加分子到光谱库
        
        返回:
            {
                'success': bool,
                'error': str or None,
                'added_count': int,
                'failed': list
            }
        """
        result = {
            'success': True,
            'error': None,
            'added_count': 0,
            'failed': []
        }
        
        if names is None:
            names = [None] * len(smiles_list)
        
        for smiles, name in zip(smiles_list, names):
            add_result = self.add_molecule(smiles, name, calibrate_to_exp)
            if add_result['success']:
                result['added_count'] += 1
            else:
                result['failed'].append({
                    'smiles': smiles,
                    'error': add_result['error']
                })
        
        if result['added_count'] == 0:
            result['success'] = False
            result['error'] = "所有分子添加失败"
        
        return result
    
    def match_single(self, user_spectrum: np.ndarray) -> List[Dict]:
        """
        单光谱匹配: 用户光谱 vs 库中所有光谱
        
        参数:
            user_spectrum: 用户光谱 (已归一化)
        
        返回:
            list: 按相似度排序的匹配结果
        """
        results = []
        
        for i, entry in enumerate(self.library.entries):
            similarity = calculate_similarity(user_spectrum, entry['spectrum'])
            results.append({
                'index': i,
                'name': entry.get('name', f'Entry_{i}'),
                'smiles': entry.get('smiles', ''),
                'similarity': similarity,
                'is_high_confidence': is_high_confidence(similarity)
            })
        
        # 按PCC降序排序
        results.sort(key=lambda x: x['similarity'].get('pcc', 0), reverse=True)
        
        return results
    
    def match_mixture(self, user_spectrum: np.ndarray) -> Dict:
        """
        两组混合光谱匹配
        
        条件:
            - 需要至少2组预测光谱
            - 仅支持两组混合
            - 比例之和 = 1
        
        参数:
            user_spectrum: 用户光谱 (已归一化)
        
        返回:
            {
                'success': bool,
                'error': str or None,
                'mixtures': list,  # 所有混合匹配结果
                'best_match': dict,  # 最佳匹配
                'is_high_confidence': bool
            }
        """
        result = {
            'success': True,
            'error': None,
            'mixtures': [],
            'best_match': None,
            'is_high_confidence': False
        }
        
        library_size = self.library.get_size()
        
        # 检查是否至少有2组预测光谱
        if library_size < 2:
            result['success'] = False
            result['error'] = f"混合匹配需要至少2组预测光谱，当前仅有 {library_size} 组"
            return result
        
        # 获取所有光谱
        spectra = self.library.get_all_spectra()
        entries = self.library.get_all_info()
        
        all_mixtures = []
        
        # 生成所有两两组合
        for idx1, idx2 in combinations(range(library_size), 2):
            spec1 = spectra[idx1]
            spec2 = spectra[idx2]
            name1 = entries[idx1].get('name', f'Entry_{idx1}')
            name2 = entries[idx2].get('name', f'Entry_{idx2}')
            smiles1 = entries[idx1].get('smiles', '')
            smiles2 = entries[idx2].get('smiles', '')
            
            # 粗搜索
            ratios = generate_mixture_ratios(MIXTURE_STEP)
            
            best_ratio = None
            best_similarity = None
            best_score = -1
            
            for ratio in ratios:
                mixed = generate_mixture_spectrum(spec1, spec2, ratio)
                similarity = calculate_similarity(user_spectrum, mixed)
                # 使用平均分作为评价指标
                avg_score = (similarity['pcc'] + similarity['sis'] + similarity['src']) / 3
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_ratio = ratio
                    best_similarity = similarity
            
            # 精细搜索 (在最佳比例附近)
            if best_ratio is not None and best_score > MIXTURE_REFINE_THRESHOLD:
                refine_ratios = refine_mixture_ratios(best_ratio)
                for ratio in refine_ratios:
                    if ratio == best_ratio:
                        continue
                    mixed = generate_mixture_spectrum(spec1, spec2, ratio)
                    similarity = calculate_similarity(user_spectrum, mixed)
                    avg_score = (similarity['pcc'] + similarity['sis'] + similarity['src']) / 3
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_ratio = ratio
                        best_similarity = similarity
            
            if best_similarity is not None:
                mixture_entry = {
                    'components': [
                        {'index': idx1, 'name': name1, 'smiles': smiles1, 'ratio': best_ratio},
                        {'index': idx2, 'name': name2, 'smiles': smiles2, 'ratio': 1 - best_ratio}
                    ],
                    'ratio': best_ratio,
                    'similarity': best_similarity,
                    'is_high_confidence': is_high_confidence(best_similarity)
                }
                all_mixtures.append(mixture_entry)
        
        # 按PCC排序
        all_mixtures.sort(key=lambda x: x['similarity'].get('pcc', 0), reverse=True)
        
        result['mixtures'] = all_mixtures
        
        if all_mixtures:
            result['best_match'] = all_mixtures[0]
            result['is_high_confidence'] = all_mixtures[0]['is_high_confidence']
        
        return result
    
    def match(
        self,
        user_spectrum: np.ndarray,
        enable_mixture: bool = True
    ) -> Dict:
        """
        完整匹配流程
        
        参数:
            user_spectrum: 用户光谱 (已归一化)
            enable_mixture: 是否启用混合匹配
        
        返回:
            {
                'success': bool,
                'error': str or None,
                'library_size': int,
                'single_matches': list,
                'mixture_matches': dict or None,
                'best_overall': dict,
                'is_high_confidence': bool
            }
        """
        result = {
            'success': True,
            'error': None,
            'library_size': self.library.get_size(),
            'single_matches': [],
            'mixture_matches': None,
            'best_overall': None,
            'is_high_confidence': False
        }
        
        # 1. 单光谱匹配
        single_results = self.match_single(user_spectrum)
        result['single_matches'] = single_results
        
        # 2. 混合匹配 (需要≥2组)
        if enable_mixture and self.library.get_size() >= 2:
            mixture_results = self.match_mixture(user_spectrum)
            result['mixture_matches'] = mixture_results
        
        # 3. 确定最佳整体匹配
        best_candidates = []
        
        # 单光谱最佳
        if single_results:
            best_single = single_results[0]
            best_candidates.append({
                'type': 'single',
                'name': best_single['name'],
                'smiles': best_single['smiles'],
                'similarity': best_single['similarity'],
                'is_high_confidence': best_single['is_high_confidence']
            })
        
        # 混合匹配最佳
        if result['mixture_matches'] and result['mixture_matches'].get('best_match'):
            best_mixture = result['mixture_matches']['best_match']
            best_candidates.append({
                'type': 'mixture',
                'components': best_mixture['components'],
                'similarity': best_mixture['similarity'],
                'is_high_confidence': best_mixture['is_high_confidence']
            })
        
        # 按PCC排序选择最佳
        if best_candidates:
            best_candidates.sort(key=lambda x: x['similarity'].get('pcc', 0), reverse=True)
            result['best_overall'] = best_candidates[0]
            result['is_high_confidence'] = best_candidates[0]['is_high_confidence']
        
        return result
    
    def match_from_file(
        self,
        user_spectrum_file: Union[str, Path],
        enable_mixture: bool = True
    ) -> Dict:
        """
        从文件加载用户光谱并匹配
        
        参数:
            user_spectrum_file: 用户光谱文件路径
            enable_mixture: 是否启用混合匹配
        
        返回:
            dict: 匹配结果
        """
        # 加载用户光谱
        load_result = load_user_spectrum(user_spectrum_file)
        
        if not load_result['success']:
            return {
                'success': False,
                'error': load_result['error'],
                'library_size': self.library.get_size()
            }
        
        # 执行匹配
        match_result = self.match(
            user_spectrum=load_result['intensity_norm'],
            enable_mixture=enable_mixture
        )
        
        # 添加用户光谱信息
        match_result['user_spectrum'] = {
            'wavenumber': load_result['wavenumber'].tolist(),
            'intensity': load_result['intensity'].tolist()
        }
        
        return match_result


# ============ 结果输出格式化 ============

def format_match_result(match_result: Dict) -> str:
    """
    格式化输出匹配结果
    """
    lines = []
    lines.append("=" * 70)
    lines.append("光谱匹配结果")
    lines.append("=" * 70)
    
    lines.append(f"\n光谱库大小: {match_result.get('library_size', 0)} 组")
    
    # 单光谱匹配结果
    lines.append("\n" + "-" * 70)
    lines.append("单光谱匹配排名:")
    lines.append("-" * 70)
    
    single_matches = match_result.get('single_matches', [])
    for i, match in enumerate(single_matches[:5]):  # 显示前5名
        lines.append(f"\n  #{i+1}: {match['name']}")
        lines.append(f"    SMILES: {match['smiles']}")
        lines.append(f"    {format_similarity(match['similarity'], 2)}")
    
    if len(single_matches) > 5:
        lines.append(f"\n  ... 共 {len(single_matches)} 组")
    
    # 混合匹配结果
    mixture_matches = match_result.get('mixture_matches')
    if mixture_matches and mixture_matches.get('success', True):
        lines.append("\n" + "-" * 70)
        lines.append("两组混合匹配结果:")
        lines.append("-" * 70)
        
        mixtures = mixture_matches.get('mixtures', [])
        for i, match in enumerate(mixtures[:5]):
            comp1 = match['components'][0]
            comp2 = match['components'][1]
            lines.append(f"\n  #{i+1}: {comp1['name']} ({comp1['ratio']:.2f}) + {comp2['name']} ({comp2['ratio']:.2f})")
            lines.append(f"    {format_similarity(match['similarity'], 2)}")
        
        if len(mixtures) > 5:
            lines.append(f"\n  ... 共 {len(mixtures)} 种混合组合")
    
    # 最佳整体匹配
    best = match_result.get('best_overall')
    if best:
        lines.append("\n" + "=" * 70)
        lines.append("最佳匹配结果:")
        lines.append("=" * 70)
        
        if best['type'] == 'single':
            lines.append(f"\n  类型: 单分子匹配")
            lines.append(f"  分子: {best['name']}")
            lines.append(f"  SMILES: {best['smiles']}")
            lines.append(f"  {format_similarity(best['similarity'], 1)}")
        else:
            comp1 = best['components'][0]
            comp2 = best['components'][1]
            lines.append(f"\n  类型: 两组混合匹配")
            lines.append(f"  组分1: {comp1['name']} ({comp1['ratio']:.2f})")
            lines.append(f"  组分2: {comp2['name']} ({comp2['ratio']:.2f})")
            lines.append(f"  {format_similarity(best['similarity'], 1)}")
        
        if best.get('is_high_confidence'):
            lines.append("\n" + "!" * 70)
            lines.append("✓ 解析结构高度接近！")
            lines.append("!" * 70)
        else:
            lines.append("\n" + "-" * 70)
            lines.append("未达到高置信度阈值，建议进一步验证")
            lines.append("-" * 70)
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


def save_match_result(match_result: Dict, output_path: Union[str, Path]):
    """
    保存匹配结果到文件
    """
    output_path = Path(output_path)
    
    # 生成可序列化的结果
    serializable = {
        'success': match_result.get('success', False),
        'error': match_result.get('error'),
        'library_size': match_result.get('library_size', 0),
        'is_high_confidence': match_result.get('is_high_confidence', False),
        'single_matches': [
            {
                'index': m['index'],
                'name': m['name'],
                'smiles': m['smiles'],
                'similarity': m['similarity'],
                'is_high_confidence': m['is_high_confidence']
            }
            for m in match_result.get('single_matches', [])
        ]
    }
    
    # 混合匹配结果
    mixture = match_result.get('mixture_matches')
    if mixture:
        serializable['mixture_matches'] = {
            'success': mixture.get('success', True),
            'error': mixture.get('error'),
            'mixtures': [
                {
                    'components': [
                        {'index': c['index'], 'name': c['name'], 'smiles': c['smiles'], 'ratio': c['ratio']}
                        for c in m['components']
                    ],
                    'similarity': m['similarity'],
                    'is_high_confidence': m['is_high_confidence']
                }
                for m in mixture.get('mixtures', [])
            ]
        }
    
    # 最佳匹配
    best = match_result.get('best_overall')
    if best:
        serializable['best_overall'] = {
            'type': best['type'],
            'similarity': best['similarity'],
            'is_high_confidence': best['is_high_confidence']
        }
        if best['type'] == 'single':
            serializable['best_overall']['name'] = best['name']
            serializable['best_overall']['smiles'] = best['smiles']
        else:
            serializable['best_overall']['components'] = [
                {'name': c['name'], 'smiles': c['smiles'], 'ratio': c['ratio']}
                for c in best['components']
            ]
    
    # 保存JSON
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    # 同时保存文本格式
    text_path = output_path.with_suffix('.txt')
    with open(text_path, 'w') as f:
        f.write(format_match_result(match_result))


# ============ 导出 ============

__all__ = [
    'SpectrumLibrary',
    'SpectralMatcher',
    'load_user_spectrum',
    'calculate_similarity',
    'is_high_confidence',
    'format_similarity',
    'format_match_result',
    'save_match_result',
    'sqrt_norm',
    'normalize_spectrum_to_wavenumber',
    'generate_mixture_spectrum',
    'HIGH_CONFIDENCE_THRESHOLD',
    'MAX_LIBRARY_SIZE',
]
"""
FlashAIRx 光谱预测模块
功能: 接收XTB IR光谱和官能团信息，通过GPR模型进行两级校准
第一步: XTB→DFT (从字典中按官能团选择模型)
第二步: DFT→EXP
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from scipy.interpolate import interp1d
import joblib


# ============ 配置 ============
CURRENT_DIR = Path(__file__).parent
DEFAULT_MODEL_DIR = CURRENT_DIR / "models"

SPECTRUM_START = 546
SPECTRUM_END = 3846
SPECTRUM_POINTS = 825


# ============ 归一化函数 ============

def sqrt_norm(data: Union[List, np.ndarray]) -> np.ndarray:
    """平方根归一化"""
    arr = np.array(data, dtype=np.float64)
    min_val = np.sqrt(np.min(arr))
    max_val = np.sqrt(np.max(arr))
    if max_val - min_val < 1e-12:
        return np.zeros_like(arr)
    return (np.sqrt(arr) - min_val) / (max_val - min_val)


def inv_sqrt(data: Union[List, np.ndarray]) -> np.ndarray:
    """
    去除平方根效果后归一化 (反归一化)
    """
    arr = np.array(data, dtype=np.float64)
    inv_sqrt_data = np.square(arr)
    # 归一化到[0,1]
    min_val = np.min(inv_sqrt_data)
    max_val = np.max(inv_sqrt_data)
    if max_val - min_val < 1e-12:
        return np.zeros_like(inv_sqrt_data)
    return (inv_sqrt_data - min_val) / (max_val - min_val)


def normalize_spectrum_to_wavenumber(
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    target_start: float = SPECTRUM_START,
    target_end: float = SPECTRUM_END,
    target_points: int = SPECTRUM_POINTS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将光谱插值到目标波数范围
    
    返回:
        (target_wavenumber, interpolated_intensity)
    """
    if len(wavenumber) == 0 or len(intensity) == 0:
        return np.array([]), np.array([])
    
    target_wavenumber = np.linspace(target_start, target_end, target_points)
    
    # 检查是否已经在目标范围
    if (np.isclose(wavenumber[0], target_start) and 
        np.isclose(wavenumber[-1], target_end) and 
        len(wavenumber) == target_points):
        return target_wavenumber, np.array(intensity)
    
    # 插值
    f_interp = interp1d(
        wavenumber,
        intensity,
        kind='linear',
        fill_value=0,
        bounds_error=False
    )
    interpolated = f_interp(target_wavenumber)
    
    return target_wavenumber, interpolated

# ==================== 通用加载函数 ====================

def load_model_file(file_path: Path):
    """
    通用模型加载函数，自动检测格式
    支持: joblib, pickle
    """
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        return joblib.load(file_path, mmap_mode=None)
    except Exception:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    # 所有方法都失败
    raise ValueError(f"无法加载模型文件: {file_path}，请确认文件格式正确")

# ============ GPR预测 ============

def gpr_predict_tf(model, X_test: np.ndarray) -> np.ndarray:
    """
    使用GPflow模型进行预测
    
    参数:
        model: GPflow GPR模型
        X_test: 输入特征 (归一化光谱)
    
    返回:
        预测结果 (1D数组)
    """
    X_tf = tf.convert_to_tensor(X_test, dtype=tf.float64)
    if len(X_tf.shape) == 1:
        X_tf = tf.expand_dims(X_tf, 0)
    mean, _ = model.predict_f(X_tf)
    return mean.numpy().flatten()


def select_model_by_functional_groups(
    functional_groups: Tuple[str, ...],
    model_dict: Dict,
    exact_match_only: bool = False
) -> Tuple[Optional[object], Optional[Tuple[str, ...]]]:
    """
    根据官能团从字典中选择匹配的GPR模型
    
    参数:
        functional_groups: 官能团元组 (已排序)
        model_dict: 模型字典 {fg_key: model}
        exact_match_only: 是否只接受精确匹配
    
    返回:
        (model, matched_key) 或 (None, None)
    """
    if not functional_groups or not model_dict:
        return None, None
    
    fg_set = set(functional_groups)
    
    # 1. 精确匹配 (官能团种类和数量完全相同)
    for key, model in model_dict.items():
        if isinstance(key, tuple) and set(key) == fg_set:
            return model, key
        elif isinstance(key, str) and len(functional_groups) == 1 and key == functional_groups[0]:
            return model, key
    
    if exact_match_only:
        return None, None
    
    # 2. 部分匹配 (选择匹配官能团数量最多的)
    best_model = None
    best_key = None
    best_match_count = -1
    best_ratio = -1
    
    for key, model in model_dict.items():
        key_set = set(key) if isinstance(key, tuple) else {key}
        common = key_set & fg_set
        
        # 匹配数量优先，其次考虑匹配比例
        match_count = len(common)
        if match_count > best_match_count:
            best_match_count = match_count
            best_model = model
            best_key = key
        elif match_count == best_match_count and best_match_count > 0:
            # 相同匹配数量时，选择官能团数量更接近的
            key_size = len(key_set)
            ratio = match_count / max(key_size, len(fg_set))
            if ratio > best_ratio:
                best_ratio = ratio
                best_model = model
                best_key = key
    
    # 至少匹配1个官能团
    if best_match_count >= 1:
        return best_model, best_key
    
    return None, None


# ============ 第一步: XTB → DFT ============

def predict_dft_spectrum(
    xtb_spectrum: Union[List, np.ndarray],
    functional_groups: Union[List, Tuple[str, ...]],
    model_dict: Dict,
    wavenumber: Optional[np.ndarray] = None,
    exact_match_only: bool = False,
    return_metadata: bool = False
) -> Dict:
    """
    第一步: XTB → DFT 预测
    
    流程:
        1. sqrt_norm(XTB) → 输入模型
        2. GPR预测 → sqrt_norm(DFT) 预测值
        3. inv_sqrt → DFT光谱
    
    参数:
        xtb_spectrum: XTB光谱强度 (原始值)
        functional_groups: 官能团列表
        model_dict: GPR模型字典 {fg_key: model}
        wavenumber: 对应的波数 (用于插值)
        exact_match_only: 是否只接受精确官能团匹配
        return_metadata: 是否返回额外元数据
    
    返回:
        {
            'success': bool,
            'error': str or None,
            'dft_spectrum': np.ndarray,      # DFT光谱 (反归一化)
            'dft_spectrum_norm': np.ndarray, # DFT光谱 (sqrt_norm归一化)
            'matched_key': tuple or str or None,
            'functional_groups': tuple,
            'wavenumber': np.ndarray,
        }
    """
    result = {
        'success': False,
        'error': None,
        'dft_spectrum': None,
        'dft_spectrum_norm': None,
        'matched_key': None,
        'functional_groups': tuple(sorted(functional_groups)),
        'wavenumber': None
    }
    
    try:
        # 1. 准备输入光谱: XTB → sqrt_norm
        xtb_arr = np.array(xtb_spectrum, dtype=np.float64)
        X_input = sqrt_norm(xtb_arr)
        
        # 确保是1D数组
        if len(X_input.shape) > 1:
            X_input = X_input.flatten()
        
        # 2. 选择模型
        fg_tuple = tuple(sorted(functional_groups))
        model, matched_key = select_model_by_functional_groups(
            fg_tuple, 
            model_dict, 
            exact_match_only
        )
        
        if model is None:
            result['error'] = f"未找到匹配的模型，官能团: {fg_tuple}"
            return result
        
        result['matched_key'] = matched_key
        
        # 3. GPR预测: sqrt_norm(XTB) → sqrt_norm(DFT) 预测值
        y_pred_norm = gpr_predict_tf(model, X_input)
        
        # 4. 反归一化: sqrt_norm(DFT) → DFT
        dft_spectrum = inv_sqrt(y_pred_norm)
        result['dft_spectrum'] = dft_spectrum
        result['dft_spectrum_norm'] = y_pred_norm
        
        # 5. 波数处理
        if wavenumber is not None:
            _, interpolated = normalize_spectrum_to_wavenumber(
                wavenumber, 
                dft_spectrum
            )
            result['dft_spectrum'] = interpolated
            
            _, interpolated_norm = normalize_spectrum_to_wavenumber(
                wavenumber,
                y_pred_norm
            )
            result['dft_spectrum_norm'] = interpolated_norm
            result['wavenumber'] = np.linspace(SPECTRUM_START, SPECTRUM_END, SPECTRUM_POINTS)
        else:
            result['wavenumber'] = np.linspace(SPECTRUM_START, SPECTRUM_END, SPECTRUM_POINTS)
        
        if return_metadata:
            result['_input_spectrum'] = X_input
            result['_raw_prediction'] = y_pred_norm
        
        result['success'] = True
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result


# ============ 第二步: DFT → EXP ============

def predict_exp_spectrum(
    dft_spectrum_norm: Union[List, np.ndarray],
    gpr_model: object,
    wavenumber: Optional[np.ndarray] = None,
    return_metadata: bool = False
) -> Dict:
    """
    第二步: DFT → EXP 预测 (直接调用GPR模型)
    
    流程:
        1. 输入已经是 sqrt_norm(DFT)，直接输入GPR模型
        2. GPR预测 → sqrt_norm(EXP) 预测值
        3. inv_sqrt → EXP光谱
    
    参数:
        dft_spectrum_norm: DFT光谱 (sqrt_norm归一化后)
        gpr_model: 单个GPR模型 (直接加载的模型对象)
        wavenumber: 对应的波数
        return_metadata: 是否返回额外元数据
    
    返回:
        {
            'success': bool,
            'error': str or None,
            'exp_spectrum': np.ndarray,       # EXP光谱 (反归一化)
            'exp_spectrum_norm': np.ndarray,  # EXP光谱 (sqrt_norm归一化)
            'wavenumber': np.ndarray,
        }
    """
    result = {
        'success': False,
        'error': None,
        'exp_spectrum': None,
        'exp_spectrum_norm': None,
        'wavenumber': None
    }
    
    try:
        # 1. 输入已经是 sqrt_norm(DFT)，直接使用
        X_input = np.array(dft_spectrum_norm, dtype=np.float64)
        
        # 确保是1D数组
        if len(X_input.shape) > 1:
            X_input = X_input.flatten()
        
        # 2. 直接调用GPR模型预测: sqrt_norm(DFT) → sqrt_norm(EXP)
        y_pred_norm = gpr_predict_tf(gpr_model, X_input)
        
        # 3. 反归一化: sqrt_norm(EXP) → EXP
        exp_spectrum = inv_sqrt(y_pred_norm)
        result['exp_spectrum'] = exp_spectrum
        result['exp_spectrum_norm'] = y_pred_norm
        
        # 4. 波数处理
        if wavenumber is not None:
            _, interpolated = normalize_spectrum_to_wavenumber(
                wavenumber, 
                exp_spectrum
            )
            result['exp_spectrum'] = interpolated
            
            _, interpolated_norm = normalize_spectrum_to_wavenumber(
                wavenumber,
                y_pred_norm
            )
            result['exp_spectrum_norm'] = interpolated_norm
            result['wavenumber'] = np.linspace(SPECTRUM_START, SPECTRUM_END, SPECTRUM_POINTS)
        else:
            result['wavenumber'] = np.linspace(SPECTRUM_START, SPECTRUM_END, SPECTRUM_POINTS)
        
        if return_metadata:
            result['_input_spectrum'] = X_input
            result['_raw_prediction'] = y_pred_norm
        
        result['success'] = True
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result


# ============ 两级校准 ============

def calibrate_spectrum(
    xtb_spectrum: Union[List, np.ndarray],
    functional_groups: Union[List, Tuple[str, ...]],
    model_dict: Dict,
    gpr_model_exp: Optional[object] = None,
    wavenumber: Optional[np.ndarray] = None,
    calibrate_to_exp: bool = False,
    exact_match_only: bool = False,
    return_all_steps: bool = False
) -> Dict:
    """
    两级光谱校准: XTB → DFT → EXP (可选)
    
    数据流:
        XTB (原始)
          → sqrt_norm → sqrt_norm(XTB)
          → GPR (字典模型) → sqrt_norm(DFT) 预测
          → inv_sqrt → DFT (反归一化)
          → [如果启用EXP] 
          → sqrt_norm → sqrt_norm(DFT)
          → GPR (单个模型) → sqrt_norm(EXP) 预测
          → inv_sqrt → EXP (反归一化)
    
    参数:
        xtb_spectrum: XTB光谱强度 (原始值)
        functional_groups: 官能团列表
        model_dict: XTB→DFT 模型字典 {fg_key: model}
        gpr_model_exp: DFT→EXP 单个GPR模型
        wavenumber: 波数
        calibrate_to_exp: 是否校准到EXP
        exact_match_only: 是否只接受精确官能团匹配
        return_all_steps: 是否返回所有中间步骤
    
    返回:
        {
            'success': bool,
            'error': str or None,
            'dft_spectrum': np.ndarray,       # DFT光谱 (反归一化)
            'dft_spectrum_norm': np.ndarray,  # DFT光谱 (sqrt_norm)
            'exp_spectrum': np.ndarray,       # EXP光谱 (反归一化)
            'exp_spectrum_norm': np.ndarray,  # EXP光谱 (sqrt_norm)
            'matched_key': tuple or str or None,
            'functional_groups': tuple,
            'wavenumber': np.ndarray,
        }
    """
    result = {
        'success': False,
        'error': None,
        'dft_spectrum': None,
        'dft_spectrum_norm': None,
        'exp_spectrum': None,
        'exp_spectrum_norm': None,
        'matched_key': None,
        'functional_groups': tuple(sorted(functional_groups)),
        'wavenumber': None
    }
    
    try:
        # ===== 第一步: XTB → DFT =====
        dft_result = predict_dft_spectrum(
            xtb_spectrum=xtb_spectrum,
            functional_groups=functional_groups,
            model_dict=model_dict,
            wavenumber=wavenumber,
            exact_match_only=exact_match_only,
            return_metadata=return_all_steps
        )
        
        if not dft_result['success']:
            result['error'] = f"DFT预测失败: {dft_result['error']}"
            return result
        
        result['dft_spectrum'] = dft_result['dft_spectrum']
        result['dft_spectrum_norm'] = dft_result['dft_spectrum_norm']
        result['matched_key'] = dft_result['matched_key']
        result['wavenumber'] = dft_result['wavenumber']
        
        if return_all_steps:
            result['_dft_input'] = dft_result.get('_input_spectrum')
            result['_dft_raw'] = dft_result.get('_raw_prediction')
        
        # ===== 第二步: DFT → EXP (可选) =====
        if calibrate_to_exp and gpr_model_exp is not None:
            # 输入: dft_spectrum_norm (已经是sqrt_norm后的DFT)
            exp_result = predict_exp_spectrum(
                dft_spectrum_norm=result['dft_spectrum_norm'],
                gpr_model=gpr_model_exp,
                wavenumber=result['wavenumber'],
                return_metadata=return_all_steps
            )
            
            if exp_result['success']:
                result['exp_spectrum'] = exp_result['exp_spectrum']
                result['exp_spectrum_norm'] = exp_result['exp_spectrum_norm']
                if return_all_steps:
                    result['_exp_input'] = exp_result.get('_input_spectrum')
                    result['_exp_raw'] = exp_result.get('_raw_prediction')
            else:
                # DFT→EXP失败不影响整体结果，但记录错误
                result['error'] = f"EXP预测失败: {exp_result['error']}"
        
        result['success'] = True
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result


# ============ 批量处理 ============

def batch_calibrate(
    spectra_list: List[Union[List, np.ndarray]],
    functional_groups_list: List[Union[List, Tuple[str, ...]]],
    model_dict: Dict,
    gpr_model_exp: Optional[object] = None,
    calibrate_to_exp: bool = False,
    exact_match_only: bool = False,
    skip_errors: bool = False
) -> List[Dict]:
    """
    批量光谱校准
    """
    results = []
    
    for i, (spectrum, fgs) in enumerate(zip(spectra_list, functional_groups_list)):
        result = calibrate_spectrum(
            xtb_spectrum=spectrum,
            functional_groups=fgs,
            model_dict=model_dict,
            gpr_model_exp=gpr_model_exp,
            calibrate_to_exp=calibrate_to_exp,
            exact_match_only=exact_match_only
        )
        results.append(result)
        
        if not result['success'] and not skip_errors:
            break
    
    return results


# ============ 模型管理 ============

class FlashAIRxPredictor:
    """
    FlashAIRx预测器类
    管理模型加载和光谱预测
    """
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        model_dict_path: Optional[Path] = None,
        gpr_exp_path: Optional[Path] = None,
        auto_load: bool = True
    ):
        """
        初始化预测器
        
        参数:
            model_dir: 模型目录
            model_dict_path: XTB→DFT模型字典文件路径
            gpr_exp_path: DFT→EXP单个GPR模型文件路径
            auto_load: 是否自动加载模型
        """
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        
        # 第一步模型: 字典 {fg_key: model}
        if model_dict_path:
            self.model_dict_path = Path(model_dict_path)
        else:
            # 尝试多个可能的文件名
            possible_names = [
                "xtb2dft_all_models.pkl",
                "xtb2dft_models_dict.pkl", 
                "xtb2dft_model1_tf.pkl",
                "xtb2dft_model1_tf_integrated.pkl"
            ]
            self.model_dict_path = None
            for name in possible_names:
                test_path = self.model_dir / name
                if test_path.exists():
                    self.model_dict_path = test_path
                    break
            
            if self.model_dict_path is None:
                self.model_dict_path = self.model_dir / "xtb2dft_models_dict.pkl"
        
        # 第二步模型: 单个GPR模型
        if gpr_exp_path:
            self.gpr_exp_path = Path(gpr_exp_path)
        else:
            self.gpr_exp_path = self.model_dir / "dft2exp_model.pkl"
        
        self.model_dict = {}
        self.gpr_model_exp = None
        
        if auto_load:
            self.load_models()
    
    def load_models(self) -> Dict:
        """加载模型 (支持 joblib 和 pickle)"""
        result = {
            'model_dict': {},
            'gpr_model_exp': None
        }
        
        # 加载XTB→DFT模型字典
        if self.model_dict_path and self.model_dict_path.exists():
            print(f"加载模型字典: {self.model_dict_path.name}")
            try:
                # ✅ 使用通用加载函数
                self.model_dict = load_model_file(self.model_dict_path)
                
                if isinstance(self.model_dict, dict):
                    result['model_dict'] = self.model_dict
                    print(f"  ✅ 已加载 {len(self.model_dict)} 个模型")
                    
                    # 统计模型类型
                    gpflow_count = 0
                    sklearn_count = 0
                    for model in self.model_dict.values():
                        if hasattr(model, 'predict_f'):
                            gpflow_count += 1
                        elif hasattr(model, 'predict') and hasattr(model, 'kernel_'):
                            sklearn_count += 1
                    
                    print(f"  GPflow: {gpflow_count}, sklearn: {sklearn_count}")
                    
                    # 显示键示例
                    keys = list(self.model_dict.keys())
                    if keys:
                        print(f"  键示例: {keys[:5]}")
                        if len(keys) > 5:
                            print(f"         ... 共 {len(keys)} 个键")
                else:
                    # 单个模型，用文件名作为键
                    key = self.model_dict_path.stem
                    self.model_dict = {key: self.model_dict}
                    result['model_dict'] = self.model_dict
                    print(f"  加载单个模型: {key}")
                    
            except Exception as e:
                print(f"  ❌ 加载失败: {e}")
                raise
        
        # 加载DFT→EXP单个GPR模型
        if self.gpr_exp_path and self.gpr_exp_path.exists():
            print(f"加载EXP模型: {self.gpr_exp_path.name}")
            try:
                self.gpr_model_exp = load_model_file(self.gpr_exp_path)
                result['gpr_model_exp'] = self.gpr_model_exp
                print(f"  ✅ EXP模型已加载")
            except Exception as e:
                print(f"  ⚠️ EXP模型加载失败: {e}")
                self.gpr_model_exp = None
        
        return result
    
    def reload_models(self):
        """重新加载模型"""
        self.load_models()
    
    def predict(
        self,
        xtb_spectrum: Union[List, np.ndarray],
        functional_groups: Union[List, Tuple[str, ...]],
        calibrate_to_exp: bool = False,
        exact_match_only: bool = False,
        return_all_steps: bool = False
    ) -> Dict:
        """
        预测光谱
        
        参数:
            xtb_spectrum: XTB光谱强度 (原始值)
            functional_groups: 官能团列表
            calibrate_to_exp: 是否校准到EXP
            exact_match_only: 是否只接受精确匹配
            return_all_steps: 是否返回所有中间步骤
        
        返回:
            dict: 预测结果
        """
        return calibrate_spectrum(
            xtb_spectrum=xtb_spectrum,
            functional_groups=functional_groups,
            model_dict=self.model_dict,
            gpr_model_exp=self.gpr_model_exp if calibrate_to_exp else None,
            calibrate_to_exp=calibrate_to_exp,
            exact_match_only=exact_match_only,
            return_all_steps=return_all_steps
        )
    
    def predict_dft_only(
        self,
        xtb_spectrum: Union[List, np.ndarray],
        functional_groups: Union[List, Tuple[str, ...]],
        exact_match_only: bool = False
    ) -> Dict:
        """仅预测DFT光谱"""
        return self.predict(
            xtb_spectrum=xtb_spectrum,
            functional_groups=functional_groups,
            calibrate_to_exp=False,
            exact_match_only=exact_match_only
        )
    
    def predict_full(
        self,
        xtb_spectrum: Union[List, np.ndarray],
        functional_groups: Union[List, Tuple[str, ...]],
        exact_match_only: bool = False
    ) -> Dict:
        """完整预测 (XTB→DFT→EXP)"""
        return self.predict(
            xtb_spectrum=xtb_spectrum,
            functional_groups=functional_groups,
            calibrate_to_exp=True,
            exact_match_only=exact_match_only
        )
    
    def predict_batch(
        self,
        spectra_list: List[Union[List, np.ndarray]],
        functional_groups_list: List[Union[List, Tuple[str, ...]]],
        calibrate_to_exp: bool = False,
        exact_match_only: bool = False,
        skip_errors: bool = False
    ) -> List[Dict]:
        """批量预测"""
        return batch_calibrate(
            spectra_list=spectra_list,
            functional_groups_list=functional_groups_list,
            model_dict=self.model_dict,
            gpr_model_exp=self.gpr_model_exp if calibrate_to_exp else None,
            calibrate_to_exp=calibrate_to_exp,
            exact_match_only=exact_match_only,
            skip_errors=skip_errors
        )


# ============ 导出 ============

__all__ = [
    'predict_dft_spectrum',
    'predict_exp_spectrum',
    'calibrate_spectrum',
    'batch_calibrate',
    'FlashAIRxPredictor',
    'sqrt_norm',
    'inv_sqrt',
    'normalize_spectrum_to_wavenumber',
    'select_model_by_functional_groups',
]
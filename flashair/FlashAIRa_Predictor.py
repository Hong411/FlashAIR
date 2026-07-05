"""
FlashAIRa 光谱预测模块（Jupyter调用版）
直接复用 FlashAIRa_prediction_and_evaluate.py 中的核心类

使用方式:
    from flashaira_predictor import FlashAIRaPredictor
    
    predictor = FlashAIRaPredictor(db_path, test_mode=True)
    result = predictor.predict_from_smiles("CCO")
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Union, List
import numpy as np

warnings.filterwarnings('ignore')

from FlashAIRa_prediction_and_evaluate import (
    Config,
    DatabaseLoader,
    FragmentLibrary,
    FragmentMatcher,
    MoleculeSplitter,
    FunctionalGroupUtils,
    create_directories
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class FlashAIRaPredictor:
    """
    FlashAIRa 光谱预测器（Jupyter调用版）
    """
    
    def __init__(
        self, 
        db_path: Optional[Union[str, Path]] = None,
        test_mode: bool = False,
        library_cache_path: Optional[Union[str, Path]] = None
    ):
        """
        初始化预测器
        
        参数:
            db_path: 数据库路径，默认使用 Config.DB_PATH
            test_mode: 是否测试模式（只使用基础库）
            library_cache_path: 碎片库缓存路径
        """
        self.db_path = Path(db_path) if db_path else Path(Config.DB_PATH)
        self.test_mode = test_mode
        self.library_cache_path = Path(library_cache_path) if library_cache_path else None
        
        create_directories()
        
        if self.library_cache_path and self.library_cache_path.exists():
            self.library = FragmentLibrary()
            self.library.load(self.library_cache_path)
        else:
            self.library = self._build_library()
            if self.library_cache_path:
                self.library.save(self.library_cache_path)
        
        self.splitter = MoleculeSplitter()
        self.matcher = FragmentMatcher(self.library)
        self._cache = {}
    
    def _build_library(self) -> FragmentLibrary:
        """构建碎片库"""
        loader = DatabaseLoader(str(self.db_path))
        
        if self.test_mode:
            return self._build_base_library(loader)
        else:
            return self._build_full_library(loader)
    
    def _build_base_library(self, loader) -> FragmentLibrary:
        """构建基础碎片库（测试模式）"""
        import sqlite3
    
        # ✅ 直接创建连接，不依赖 loader.conn
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT mol_id, intensity FROM sim_spectrum WHERE intensity IS NOT NULL")
        ir_spectra = {}
        for mol_id, intensity_str in cursor.fetchall():
            try:
                import json
                data = json.loads(intensity_str)
                spectrum = np.array(data, dtype=np.float32)
                if len(spectrum) == Config.IR_POINTS:
                    ir_spectra[mol_id] = spectrum
            except:
                continue
        
        cursor.execute("""
            SELECT m.mol_id, m.mol_smiles
            FROM molecular_information m
            INNER JOIN sim_spectrum s ON m.mol_id = s.mol_id
            WHERE m.mol_smiles IS NOT NULL
            GROUP BY m.mol_id
        """)
        
        library = FragmentLibrary()
        fg_utils = FunctionalGroupUtils()
        
        for mol_id, smiles in cursor.fetchall():
            ir = ir_spectra.get(mol_id)
            if ir is None:
                continue
            
            from rdkit import Chem
            from rdkit.Chem import rdmolops
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                Chem.SanitizeMol(mol)
                rdmolops.FastFindRings(mol)
            except:
                continue
            
            heavy_atoms = mol.GetNumHeavyAtoms()
            fg_types = fg_utils.get_fg_types(mol)
            has_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
            
            is_base = False
            if heavy_atoms <= Config.SMALL_MOLECULE_MAX_HEAVY_ATOMS:
                is_base = True
            if has_aromatic and heavy_atoms <= Config.AROMATIC_SMALL_MAX_HEAVY_ATOMS:
                is_base = True
            
            if is_base:
                library.add_molecule(smiles, ir, heavy_atoms, fg_types)
        
        library.finalize()
        conn.close()
        return library
    
    def _build_full_library(self, loader) -> FragmentLibrary:
        """构建完整碎片库"""
        conn = loader.conn
        cursor = conn.cursor()
        
        cursor.execute("SELECT mol_id, intensity FROM sim_spectrum WHERE intensity IS NOT NULL")
        ir_spectra = {}
        for mol_id, intensity_str in cursor.fetchall():
            try:
                import json
                data = json.loads(intensity_str)
                spectrum = np.array(data, dtype=np.float32)
                if len(spectrum) == Config.IR_POINTS:
                    ir_spectra[mol_id] = spectrum
            except:
                continue
        
        cursor.execute("""
            SELECT m.mol_id, m.mol_smiles
            FROM molecular_information m
            INNER JOIN sim_spectrum s ON m.mol_id = s.mol_id
            WHERE m.mol_smiles IS NOT NULL
            GROUP BY m.mol_id
        """)
        
        library = FragmentLibrary()
        fg_utils = FunctionalGroupUtils()
        
        from rdkit import Chem
        from rdkit.Chem import rdmolops
        
        for mol_id, smiles in cursor.fetchall():
            ir = ir_spectra.get(mol_id)
            if ir is None:
                continue
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                Chem.SanitizeMol(mol)
                rdmolops.FastFindRings(mol)
            except:
                continue
            
            heavy_atoms = mol.GetNumHeavyAtoms()
            fg_types = fg_utils.get_fg_types(mol)
            
            library.add_molecule(smiles, ir, heavy_atoms, fg_types)
        
        library.finalize()
        conn.close()
        return library
    
    def predict_from_smiles(self, smiles: str) -> Dict:
        """
        从SMILES预测IR光谱
        
        返回:
            {
                'success': bool,
                'error': str or None,
                'smiles': str,
                'predicted_ir': np.ndarray,
                'wavenumber': np.ndarray,
                'fragments': list,
                'match_level': str,
                'matched_smiles': str,
                'match_similarity': float,
                'raw_ir': np.ndarray,
            }
        """
        result = {
            'success': False,
            'error': None,
            'smiles': smiles,
            'predicted_ir': None,
            'wavenumber': Config.load_wavenumbers(),
            'fragments': [],
            'match_level': None,
            'matched_smiles': None,
            'match_similarity': 0.0,
            'raw_ir': None
        }
        
        try:
            from rdkit import Chem
            from rdkit.Chem import rdmolops
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result['error'] = f"无效的SMILES: {smiles}"
                return result
            
            try:
                Chem.SanitizeMol(mol)
                rdmolops.FastFindRings(mol)
            except Exception as e:
                result['error'] = f"分子清理失败: {str(e)}"
                return result
            
            try:
                cache_key = Chem.MolToSmiles(mol)
                if cache_key in self._cache:
                    cached = self._cache[cache_key]
                    result['predicted_ir'] = cached['predicted_ir']
                    result['fragments'] = cached['fragments']
                    result['match_level'] = cached.get('match_level')
                    result['matched_smiles'] = cached.get('matched_smiles')
                    result['match_similarity'] = cached.get('match_similarity', 0.0)
                    result['raw_ir'] = cached.get('raw_ir')
                    result['success'] = True
                    return result
            except:
                pass
            
            fragments = self.splitter.split(mol)
            
            if not fragments:
                result['error'] = "分子拆分失败"
                return result
            
            fragment_smiles = []
            for frag in fragments:
                try:
                    frag_smiles = Chem.MolToSmiles(frag)
                    fragment_smiles.append(frag_smiles)
                except:
                    fragment_smiles.append("None")
            result['fragments'] = fragment_smiles
            
            ir_list = []
            match_levels = []
            match_smiles_list = []
            match_similarities = []
            
            for frag in fragments:
                ir = self.matcher.match(frag)
                if ir is not None:
                    ir_list.append(ir)
                    info = self.matcher.get_last_match_info()
                    if info:
                        match_levels.append(info.get('level', 'Unknown'))
                        match_smiles_list.append(info.get('matched_smiles', 'Unknown'))
                        match_similarities.append(info.get('similarity', 0.0))
            
            if not ir_list:
                result['error'] = "所有片段匹配失败"
                return result
            
            combined = np.zeros(Config.IR_POINTS, dtype=np.float32)
            for ir in ir_list:
                combined += ir
            
            result['raw_ir'] = combined.copy()
            
            min_val = np.min(combined)
            max_val = np.max(combined)
            if max_val > min_val:
                combined = (combined - min_val) / (max_val - min_val)
            else:
                combined = np.zeros(Config.IR_POINTS, dtype=np.float32)
            
            result['predicted_ir'] = combined
            
            if match_levels:
                priority = {'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4}
                best_idx = 0
                best_priority = 99
                for i, level in enumerate(match_levels):
                    p = priority.get(level, 99)
                    if p < best_priority:
                        best_priority = p
                        best_idx = i
                
                result['match_level'] = match_levels[best_idx] if best_idx < len(match_levels) else None
                result['matched_smiles'] = match_smiles_list[best_idx] if best_idx < len(match_smiles_list) else None
                result['match_similarity'] = match_similarities[best_idx] if best_idx < len(match_similarities) else 0.0
            
            try:
                self._cache[cache_key] = {
                    'predicted_ir': combined,
                    'fragments': fragment_smiles,
                    'match_level': result['match_level'],
                    'matched_smiles': result['matched_smiles'],
                    'match_similarity': result['match_similarity'],
                    'raw_ir': result['raw_ir']
                }
            except:
                pass
            
            result['success'] = True
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
    
    def predict_from_smiles_batch(self, smiles_list: List[str]) -> List[Dict]:
        """批量预测"""
        results = []
        for smiles in smiles_list:
            results.append(self.predict_from_smiles(smiles))
        return results
    
    def clear_cache(self):
        """清空缓存"""
        self._cache = {}
    
    def get_library_info(self) -> Dict:
        """获取碎片库信息"""
        return {
            'size': len(self.library.smiles_to_ir),
            'test_mode': self.test_mode,
            'fg_types': list(self.library.fg_to_smiles.keys()),
            'fg_combo_types': len(self.library.fg_combo_to_smiles)
        }


def create_predictor(
    db_path: Optional[Union[str, Path]] = None,
    test_mode: bool = False,
    library_cache_path: Optional[Union[str, Path]] = None
) -> FlashAIRaPredictor:
    """创建预测器实例"""
    return FlashAIRaPredictor(db_path, test_mode, library_cache_path)


__all__ = ['FlashAIRaPredictor', 'create_predictor']
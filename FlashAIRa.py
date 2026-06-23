# -*- coding: utf-8 -*-
"""
分子碎片化 IR 谱预测系统 V0.8.8

核心逻辑：
1. BFS 从官能团向外扩展提取片段（芳香环视为节点，脂肪环按普通原子）
2. 四级匹配：L1精确匹配 → L3/L4降级匹配 → L2相似匹配
3. 深度按重原子计数，氢原子不计入深度
4. 支持单侧/双侧官能团分类
5. 多官能团无组合时，分别匹配并组合光谱
6. L2 按官能团分类 + 指纹分桶加速

新增功能：
1. 保存预测光谱 (pred_ir) 和真实光谱 (true_ir)
2. 保存每个分子的拆分片段 (fragments)
3. 保存匹配信息 (匹配级别、匹配分子、相似度)

"""

import sqlite3
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdmolops
from collections import defaultdict
import pickle
import gzip
import json
from pathlib import Path
import time
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装 tqdm 可获得更好进度显示 (pip install tqdm)")

# ==================== 配置参数 ====================
class Config:
    # 数据库路径
    DB_PATH = r'D:\chemdata\Database\SOMIR0907d.db'
    
    # 工作目录
    WORK_DIR = Path('V0.8.8')
    INDEX_DIR = WORK_DIR / 'index'
    FIGURE_DIR = WORK_DIR / 'figures'
    DATA_DIR = WORK_DIR / 'data'
    
    # 索引文件
    FRAGMENT_LIBRARY_FILE = INDEX_DIR / 'fragment_library.pkl.gz'
    PREDICTION_RESULTS_FILE = DATA_DIR / 'prediction_results.pkl'
    PREDICTION_DETAILS_FILE = DATA_DIR / 'prediction_details.pkl.gz'
    
    # IR 谱参数
    WAVENUMBER_START = 550
    WAVENUMBER_END = 3846
    WAVENUMBER_STEP = 4
    IR_POINTS = 825
    
    # 波数文件路径
    WAVENUMBERS_PATH = Path(r'D:\chemdata\Database\npy\wavenumber_550-3846-4.npy')
    
    # 拆分参数
    MAX_DEPTH_SINGLE = 4      # 单侧最大深度（重原子数）
    MAX_DEPTH_BILATERAL = 4   # 双侧每侧最大深度
    SIMILARITY_THRESHOLD = 0.7
    FINGERPRINT_RADIUS = 2
    FINGERPRINT_BITS = 2048
    
    # 分子分类
    SMALL_MOLECULE_MAX_HEAVY_ATOMS = 6
    AROMATIC_SMALL_MAX_HEAVY_ATOMS = 9
    
    # 交叉验证
    CV_FOLDS = 3
    
    # L2 相似匹配参数
    L2_TOP_N = 500            # 最大候选数量
    FP_BUCKET_BITS = 8        # 指纹分桶位数
    
    # 官能团 SMARTS 模式
    FUNCTIONAL_GROUP_PATTERNS = {
        "Alkene": "[CX3]=[C]",
        "Alkyne": "[CX2]#C",
        "Aromatic": "[a]",
        "Alcohol": "[CX4][OX2H]",
        "Ester": "[CX3](=O)[OX2H0][#6]",
        "Aldehyde": "[CX3H1](=O)[#6]",
        "Ketone": "[#6][CX3](=O)[#6]",
        "Carboxylic Acid": "[CX3](=O)[OX1H0-,X2H1]",
        "Ether": "[OX2;!$(OC=O)]([#6])[#6]",
        "Amide": "[CX3](=[OX1])[NX3H2,NX3H1,NX3H0,NX4H]",
        "Amine": "[NX3;H2,H1,H0;!$(NC=O)]",
        "Nitrile": "[NX1]#[CX2]",
        "Nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "Imine": "[CX3]=[NX2]",
        "Halide": "[#6][F,Cl,Br,I]",
    }
    
    # 官能团重要性分层（L3 匹配用）
    FG_IMPORTANCE_HIGH = ["Carboxylic Acid", "Amide", "Ester", "Nitro", "Ketone", "Aldehyde"]
    FG_IMPORTANCE_MEDIUM = ["Alcohol", "Amine", "Alkene", "Imine", "Nitrile"]
    FG_IMPORTANCE_WEAK = ["Ether", "Halide", "Alkyne"]
    
    # 完整删除顺序（从最弱到最强）
    FG_DELETE_ORDER = [
        "Alkyne", "Halide", "Ether",           # Weak
        "Nitrile", "Imine", "Alkene", "Amine", "Alcohol",  # Medium
        "Aldehyde", "Ketone", "Nitro", "Ester", "Amide", "Carboxylic Acid"  # High
    ]
    
    # 单侧扩展官能团
    SINGLE_SIDE_FGS = {"Alcohol", "Aldehyde", "Carboxylic Acid", "Halide", 
                       "Nitrile", "Nitro"}
    
    # 烷烃代表分子
    ALKANE_REPRESENTATIVES = {
        1: "C", 2: "CC", 3: "CCC", 4: "CCCC", 5: "CCCCC", 6: "CCCCCC",
        7: "CCCCCCC", 8: "CCCCCCCC"
    }
    
    # 芳香族母体（同模式烷基芳烃）
    AROMATIC_PARENT = {
        "single": "Cc1ccccc1",           # 甲苯
        "ortho": "Cc1ccccc1C",           # 邻二甲苯
        "meta": "Cc1cc(C)cc1",           # 间二甲苯
        "para": "Cc1ccc(C)cc1",          # 对二甲苯
        "tri_123": "Cc1c(C)cccc1C",      # 1,2,3-三甲苯
        "tri_124": "Cc1cc(C)c(C)c1",     # 1,2,4-三甲苯
        "tri_135": "Cc1cc(C)cc(C)c1",    # 1,3,5-三甲苯
        "tetra": "Cc1c(C)c(C)cc1C",      # 四甲苯
        "penta": "Cc1c(C)c(C)c(C)cc1",   # 五甲苯
        "hexa": "Cc1c(C)c(C)c(C)c(C)c1"  # 六甲苯
    }
    
    @staticmethod
    def load_wavenumbers():
        """从 npy 文件加载波数坐标"""
        if Config.WAVENUMBERS_PATH.exists():
            return np.load(Config.WAVENUMBERS_PATH)
        else:
            return np.arange(Config.WAVENUMBER_START, Config.WAVENUMBER_END + Config.WAVENUMBER_STEP, Config.WAVENUMBER_STEP)


# ==================== 创建目录 ====================
def create_directories():
    Config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    Config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    Config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"工作目录: {Config.WORK_DIR}")


# ==================== 指纹生成器 ====================
class FingerprintGenerator:
    def __init__(self, radius=2, nbits=2048):
        self.radius = radius
        self.nbits = nbits
        self._generator = None
        self._init_generator()
    
    def _init_generator(self):
        try:
            self._generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.radius, fpSize=self.nbits)
        except TypeError:
            try:
                self._generator = rdFingerprintGenerator.GetMorganGenerator(
                    radius=self.radius, nBits=self.nbits)
            except:
                self._generator = None
    
    def get_fingerprint(self, mol):
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
            rdmolops.FastFindRings(mol)
        except:
            pass
        
        if self._generator is not None:
            return self._generator.GetFingerprint(mol)
        else:
            return AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.nbits)


# ==================== 碎片库类 ====================
class FragmentLibrary:
    """碎片库 - 四级匹配的数据源"""
    
    def __init__(self):
        # L1: 精确匹配索引
        self.smiles_to_ir = {}
        self.inchikey_to_ir = {}
        
        # L2/L3: 索引
        self.smiles_to_fp = {}
        self.smiles_to_heavy = {}
        self.all_smiles = []
        
        # L3: 官能团索引
        self.fg_to_smiles = defaultdict(list)           # 单官能团 -> smiles列表
        self.fg_combo_to_smiles = defaultdict(list)    # 官能团组合 -> smiles列表
        
        # L2: 指纹分桶索引
        self.fp_buckets = defaultdict(list)             # 指纹前N位 -> smiles列表
        self.fp_cache = {}                              # smiles -> fingerprint
        
        # L4: 芳香族索引
        self.aromatic_by_pattern = defaultdict(list)
        
        self.fp_gen = FingerprintGenerator()
    
    def add_molecule(self, smiles, ir_spectrum, heavy_atoms, fg_types, aromatic_pattern=None):
        """添加分子到碎片库"""
        canon_smiles = Chem.CanonSmiles(smiles)
        
        # L1
        self.smiles_to_ir[canon_smiles] = ir_spectrum
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            inchikey = Chem.MolToInchiKey(mol)
            self.inchikey_to_ir[inchikey] = ir_spectrum
            fp = self.fp_gen.get_fingerprint(mol)
            self.smiles_to_fp[canon_smiles] = fp
            self.smiles_to_heavy[canon_smiles] = heavy_atoms
            
            # 指纹分桶
            bucket_key = fp.ToBitString()[:Config.FP_BUCKET_BITS]
            self.fp_buckets[bucket_key].append(canon_smiles)
            self.fp_cache[canon_smiles] = fp
        
        self.all_smiles.append(canon_smiles)
        
        # L3: 官能团索引
        for fg in fg_types:
            self.fg_to_smiles[fg].append(canon_smiles)
        
        if len(fg_types) > 1:
            combo = "|".join(sorted(fg_types))
            self.fg_combo_to_smiles[combo].append(canon_smiles)
        
        # L4: 芳香族索引
        if aromatic_pattern:
            self.aromatic_by_pattern[aromatic_pattern].append({
                'smiles': canon_smiles, 'heavy_atoms': heavy_atoms, 'ir': ir_spectrum
            })
    
    def finalize(self):
        """索引排序（按重原子数）"""
        for fg in self.fg_to_smiles:
            self.fg_to_smiles[fg].sort(key=lambda x: self.smiles_to_heavy.get(x, 0))
        for combo in self.fg_combo_to_smiles:
            self.fg_combo_to_smiles[combo].sort(key=lambda x: self.smiles_to_heavy.get(x, 0))
        for pattern in self.aromatic_by_pattern:
            self.aromatic_by_pattern[pattern].sort(key=lambda x: x['heavy_atoms'])
    
    def save(self, filepath):
        with gzip.open(filepath, 'wb') as f:
            pickle.dump({
                'smiles_to_ir': self.smiles_to_ir,
                'inchikey_to_ir': self.inchikey_to_ir,
                'all_smiles': self.all_smiles,
                'fg_to_smiles': dict(self.fg_to_smiles),
                'fg_combo_to_smiles': dict(self.fg_combo_to_smiles),
                'fp_buckets': dict(self.fp_buckets),
                'fp_cache': self.fp_cache,
                'smiles_to_heavy': self.smiles_to_heavy,
                'aromatic_by_pattern': dict(self.aromatic_by_pattern)
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"碎片库已保存: {filepath}")
    
    def load(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.smiles_to_ir = data['smiles_to_ir']
        self.inchikey_to_ir = data.get('inchikey_to_ir', {})
        self.all_smiles = data['all_smiles']
        self.fg_to_smiles = defaultdict(list, data['fg_to_smiles'])
        self.fg_combo_to_smiles = defaultdict(list, data.get('fg_combo_to_smiles', {}))
        self.fp_buckets = defaultdict(list, data.get('fp_buckets', {}))
        self.fp_cache = data.get('fp_cache', {})
        self.smiles_to_heavy = data.get('smiles_to_heavy', {})
        self.aromatic_by_pattern = defaultdict(list, data.get('aromatic_by_pattern', {}))
        print(f"碎片库已加载: {len(self.smiles_to_ir)} 个分子")
    
    def get_similar_candidates_by_fg(self, fg_type, top_n=None):
        """根据官能团类型获取候选分子"""
        candidates = self.fg_to_smiles.get(fg_type, [])
        if top_n and len(candidates) > top_n:
            return candidates[:top_n]
        return candidates
    
    def get_similar_candidates_by_fingerprint(self, query_fp, top_n=Config.L2_TOP_N):
        """根据指纹分桶获取候选分子"""
        bucket_key = query_fp.ToBitString()[:Config.FP_BUCKET_BITS]
        candidates = self.fp_buckets.get(bucket_key, [])
        if len(candidates) > top_n:
            return candidates[:top_n]
        return candidates


# ==================== 数据库加载器 ====================
class DatabaseLoader:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def load_all_molecules(self):
        """加载所有分子和 IR 谱"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 加载 IR 谱（使用 intensity 列）
        cursor.execute("SELECT mol_id, intensity FROM sim_spectrum WHERE intensity IS NOT NULL")
        ir_spectra = {}
        for mol_id, intensity_str in cursor.fetchall():
            try:
                data = json.loads(intensity_str)
                spectrum = np.array(data, dtype=np.float32)
                if len(spectrum) == Config.IR_POINTS:
                    ir_spectra[mol_id] = spectrum
            except:
                continue
        
        # 加载分子信息
        cursor.execute("""
            SELECT m.mol_id, m.mol_smiles
            FROM molecular_information m
            INNER JOIN sim_spectrum s ON m.mol_id = s.mol_id
            WHERE m.mol_smiles IS NOT NULL
            GROUP BY m.mol_id
        """)
        
        molecules = []
        smiles_to_ir = {}
        
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
            
            molecules.append({
                'mol': mol,
                'smiles': smiles,
                'ir': ir,
                'heavy_atoms': mol.GetNumHeavyAtoms()
            })
            smiles_to_ir[smiles] = ir
        
        conn.close()
        print(f"加载了 {len(molecules)} 个分子")
        return molecules, smiles_to_ir


# ==================== 官能团处理工具 ====================
class FunctionalGroupUtils:
    @staticmethod
    def match_fgs(mol):
        """匹配分子中的所有官能团，返回 [(fg_name, matched_atoms), ...]"""
        results = []
        for fg_name, smarts in Config.FUNCTIONAL_GROUP_PATTERNS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                results.append((fg_name, set(match)))
        
        # 去重合并
        merged = []
        used_atoms = set()
        for fg_name, atoms in sorted(results, key=lambda x: -len(x[1])):
            if not atoms.intersection(used_atoms):
                merged.append((fg_name, atoms))
                used_atoms.update(atoms)
        return merged
    
    @staticmethod
    def get_fg_types(mol):
        """获取分子中的官能团类型列表"""
        fg_matches = FunctionalGroupUtils.match_fgs(mol)
        return list(set([fg for fg, _ in fg_matches]))
    
    @staticmethod
    def get_ring_exit_atoms(mol, ring_atoms, visited):
        """获取芳香环的出口原子（仅重原子）"""
        exits = []
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                if neighbor.GetAtomicNum() == 1:
                    continue
                if n_idx in ring_atoms:
                    continue
                if n_idx in visited:
                    continue
                exits.append(n_idx)
        return exits
    
    @staticmethod
    def get_substitution_pattern(mol, aromatic_rings):
        """获取芳香环的取代模式"""
        if not aromatic_rings:
            return None
        
        ring_atoms = aromatic_rings[0]
        substituents = []
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    continue
                if neighbor.GetIdx() not in ring_atoms:
                    substituents.append(idx)
                    break
        
        num_sub = len(substituents)
        if num_sub <= 1:
            return "single"
        elif num_sub == 2:
            return "para"
        elif num_sub == 3:
            return "tri_135"
        else:
            return "tetra"


# ==================== BFS 扩展模块 ====================
class BFSExtractor:
    def __init__(self, aromatic_rings):
        self.aromatic_rings = aromatic_rings
        self.processed_rings = set()
    
    def bfs_one_direction(self, mol, start_atom, max_depth):
        """单方向 BFS 扩展（深度按重原子计数）"""
        from collections import deque
        
        visited = set()
        queue = deque([(start_atom, 0)])
        visited.add(start_atom)
        self.processed_rings.clear()
        
        while queue:
            idx, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # 检查是否属于芳香环
            ring_id = self._find_aromatic_ring(idx)
            if ring_id is not None and ring_id not in self.processed_rings:
                ring_atoms = self.aromatic_rings[ring_id]
                
                # 确保添加整个环的所有原子
                for ring_idx in ring_atoms:
                    if ring_idx not in visited:
                        visited.add(ring_idx)
                
                # 获取出口原子
                exit_atoms = FunctionalGroupUtils.get_ring_exit_atoms(mol, ring_atoms, visited)
                for exit_idx in exit_atoms:
                    if exit_idx not in visited:
                        visited.add(exit_idx)
                        queue.append((exit_idx, depth + 2))
                
                self.processed_rings.add(ring_id)
                continue
            
            # 普通原子扩展（跳过氢）
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                if neighbor.GetAtomicNum() == 1:
                    continue
                if n_idx in visited:
                    continue
                visited.add(n_idx)
                queue.append((n_idx, depth + 1))
        
        return visited
    
    def _find_aromatic_ring(self, atom_idx):
        for i, ring in enumerate(self.aromatic_rings):
            if atom_idx in ring:
                return i
        return None


# ==================== 分子拆分模块 ====================
class MoleculeSplitter:
    def __init__(self):
        self.fg_utils = FunctionalGroupUtils()
    
    def split(self, mol):
        """拆分分子，返回片段列表"""
        # 预处理
        try:
            Chem.SanitizeMol(mol)
            rdmolops.FastFindRings(mol)
            Chem.SetAromaticity(mol)
        except Exception as e:
            # 如果原分子就有问题，直接返回整个分子
            return [mol]
        
        # 获取芳香环（只保留完整的芳香环）
        ring_info = mol.GetRingInfo()
        aromatic_rings = []
        for ring in ring_info.AtomRings():
            # 检查环中所有原子都是芳香性的
            is_aromatic = True
            for i in ring:
                if not mol.GetAtomWithIdx(i).GetIsAromatic():
                    is_aromatic = False
                    break
            if is_aromatic and len(ring) <= 7:  # 只处理5-7元环
                aromatic_rings.append(set(ring))
        
        # 匹配官能团
        fg_matches = self.fg_utils.match_fgs(mol)
        
        if not fg_matches:
            # 无官能团，整个分子作为一个片段
            return [mol]
        
        # 确定 BFS 起点
        start_points = self._get_bfs_start_points(mol, fg_matches)
        
        # BFS 提取片段
        extractor = BFSExtractor(aromatic_rings)
        fragments = []
        
        for fg_name, start_atoms, is_bilateral in start_points:
            if is_bilateral:
                center = start_atoms[0]
                left_visited = extractor.bfs_one_direction(mol, center, Config.MAX_DEPTH_BILATERAL)
                right_visited = extractor.bfs_one_direction(mol, center, Config.MAX_DEPTH_BILATERAL)
                visited = left_visited.union(right_visited)
            else:
                visited = extractor.bfs_one_direction(mol, start_atoms[0], Config.MAX_DEPTH_SINGLE)
            
            # 提取片段
            frag_mol = self._extract_fragment(mol, visited)
            if frag_mol and frag_mol.GetNumHeavyAtoms() > 0:
                fragments.append(frag_mol)
        
        # 去重
        fragments = self._deduplicate(fragments)
        return fragments
    
    def _deduplicate(self, fragments):
        """片段去重（基于规范 SMILES）"""
        unique = {}
        for frag in fragments:
            if frag is None:
                continue
            try:
                smiles = Chem.MolToSmiles(frag)
                canon = Chem.CanonSmiles(smiles)
                if canon not in unique:
                    unique[canon] = frag
            except:
                continue
        return list(unique.values())
    
    def _get_bfs_start_points(self, mol, fg_matches):
        """确定 BFS 起点和方向"""
        start_points = []
        
        for fg_name, atoms in fg_matches:
            if fg_name in Config.SINGLE_SIDE_FGS:
                feature_atom = self._find_feature_atom(mol, atoms, fg_name)
                if feature_atom is not None:
                    start_points.append((fg_name, [feature_atom], False))
            else:
                center_atom = self._find_center_atom(mol, atoms, fg_name)
                if center_atom is not None:
                    start_points.append((fg_name, [center_atom], True))
        
        return start_points
    
    def _find_feature_atom(self, mol, atoms, fg_name):
        """找特征原子（O、N、卤素等）"""
        for idx in atoms:
            symbol = mol.GetAtomWithIdx(idx).GetSymbol()
            if symbol in ['O', 'N', 'F', 'Cl', 'Br', 'I']:
                return idx
        return next(iter(atoms)) if atoms else None
    
    def _find_center_atom(self, mol, atoms, fg_name):
        """找中心原子"""
        if fg_name == "Ether":
            for idx in atoms:
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'O':
                    return idx
        elif fg_name in ["Ester", "Ketone", "Aldehyde", "Carboxylic Acid"]:
            for idx in atoms:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'C':
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'O':
                            return idx
        elif fg_name in ["Alkene", "Alkyne"]:
            return next(iter(atoms))
        return next(iter(atoms)) if atoms else None
    
    def _extract_fragment(self, mol, atom_indices):
        """提取子结构"""
        if not atom_indices:
            return None
        
        rw_mol = Chem.RWMol(mol)
        all_atoms = set(range(mol.GetNumAtoms()))
        to_remove = all_atoms - atom_indices
        for idx in sorted(to_remove, reverse=True):
            try:
                rw_mol.RemoveAtom(idx)
            except:
                continue
        
        try:
            frag = rw_mol.GetMol()
            
            # 强制重置所有原子的芳香性标记
            for atom in frag.GetAtoms():
                atom.SetIsAromatic(False)
            
            # 尝试清理，如果失败则跳过
            try:
                Chem.SanitizeMol(frag)
                Chem.SetAromaticity(frag)
            except Exception as e:
                # 清理失败，尝试修复
                try:
                    # 移除所有环信息，重新计算
                    for bond in frag.GetBonds():
                        bond.SetIsAromatic(False)
                    Chem.SanitizeMol(frag)
                except:
                    # 如果仍然失败，返回 None
                    return None
            
            frag = Chem.AddHs(frag)
            
            try:
                Chem.SanitizeMol(frag)
            except:
                # 补氢后清理失败，尝试不加氢
                frag = rw_mol.GetMol()
                try:
                    Chem.SanitizeMol(frag)
                except:
                    return None
            
            return frag
        except Exception as e:
            return None


# ==================== 匹配模块 ====================
class FragmentMatcher:
    def __init__(self, library):
        self.library = library
        self.fp_gen = FingerprintGenerator()
        self.fg_utils = FunctionalGroupUtils()
        self.last_match_info = None  # 记录最后一次匹配的信息
    
    def match(self, fragment_mol):
        """四级匹配，返回 IR 谱"""
        # 重置匹配信息
        self.last_match_info = None
        
        # L1 精确匹配
        ir, info = self._exact_match_with_info(fragment_mol)
        if ir is not None:
            self.last_match_info = info
            return ir
        
        # 判断片段类型
        has_aromatic = self._has_aromatic_ring(fragment_mol)
        
        if has_aromatic:
            # L4 芳香族降级匹配
            ir, info = self._aromatic_fallback_with_info(fragment_mol)
            if ir is not None:
                self.last_match_info = info
                return ir
        else:
            # L3 官能团降级匹配
            ir, info = self._functional_group_fallback_with_info(fragment_mol)
            if ir is not None:
                self.last_match_info = info
                return ir
        
        # L2 相似匹配（最后兜底）
        ir, info = self._similarity_match_with_info(fragment_mol)
        self.last_match_info = info
        return ir
    
    def get_last_match_info(self):
        """获取最后一次匹配的信息"""
        return self.last_match_info
    
    def _exact_match_with_info(self, mol):
        """L1 精确匹配，返回 (IR, info_dict)"""
        try:
            smiles = Chem.MolToSmiles(mol)
            canon = Chem.CanonSmiles(smiles)
            ir = self.library.smiles_to_ir.get(canon)
            if ir is not None:
                return ir, {'level': 'L1', 'matched_smiles': canon, 'similarity': 1.0}
            return None, None
        except:
            return None, None
    
    def _functional_group_fallback_with_info(self, mol):
        """L3 官能团降级匹配（脂肪族）- 按相似度优先，芳香性约束"""
        fg_types = self.fg_utils.get_fg_types(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        query_fp = self.fp_gen.get_fingerprint(mol)
        
        # 检查当前片段是否含芳香环
        has_aromatic = self._has_aromatic_ring(mol)
        
        # 情况A：纯烷烃
        if not fg_types:
            carbon_count = heavy_atoms
            for c in range(carbon_count, carbon_count + 10):
                rep_smiles = Config.ALKANE_REPRESENTATIVES.get(c)
                if rep_smiles and rep_smiles in self.library.smiles_to_ir:
                    ir = self.library.smiles_to_ir[rep_smiles]
                    return ir, {'level': 'L3', 'matched_smiles': rep_smiles, 'similarity': 0.5, 'type': 'alkane'}
            return None, None
        
        # 情况B：单官能团 - 按相似度优先，芳香性约束
        if len(fg_types) == 1:
            fg = fg_types[0]
            candidates = self.library.fg_to_smiles.get(fg, [])
            
            if not candidates:
                return None, None
            
            # 过滤候选：芳香性必须一致
            filtered_candidates = []
            for s in candidates:
                s_has_aromatic = self._smiles_has_aromatic(s)
                if has_aromatic == s_has_aromatic:
                    filtered_candidates.append(s)
            
            # 如果过滤后为空，使用全部候选（但记录警告）
            if not filtered_candidates:
                filtered_candidates = candidates
            
            # 按指纹相似度排序
            best_smiles = None
            best_sim = -1.0
            best_ir = None
            
            for smiles in filtered_candidates:
                cand_fp = self.library.fp_cache.get(smiles)
                if cand_fp is None:
                    continue
                
                sim = DataStructs.TanimotoSimilarity(query_fp, cand_fp)
                if sim > best_sim:
                    best_sim = sim
                    best_smiles = smiles
                    best_ir = self.library.smiles_to_ir.get(smiles)
                    if sim > 0.9:
                        break
            
            if best_ir is not None:
                return best_ir, {'level': 'L3', 'matched_smiles': best_smiles, 'similarity': best_sim, 'type': 'single_fg', 'fg': fg, 'has_aromatic': has_aromatic}
            return None, None
        
        # 情况C：多官能团 - 优先找相同组合
        combo = "|".join(sorted(fg_types))
        candidates = self.library.fg_combo_to_smiles.get(combo, [])
        
        if candidates:
            # 过滤候选：芳香性必须一致
            filtered_candidates = []
            for s in candidates:
                s_has_aromatic = self._smiles_has_aromatic(s)
                if has_aromatic == s_has_aromatic:
                    filtered_candidates.append(s)
            
            if not filtered_candidates:
                filtered_candidates = candidates
            
            # 按指纹相似度排序
            best_smiles = None
            best_sim = -1.0
            best_ir = None
            
            for smiles in filtered_candidates:
                cand_fp = self.library.fp_cache.get(smiles)
                if cand_fp is None:
                    continue
                
                sim = DataStructs.TanimotoSimilarity(query_fp, cand_fp)
                if sim > best_sim:
                    best_sim = sim
                    best_smiles = smiles
                    best_ir = self.library.smiles_to_ir.get(smiles)
                    if sim > 0.9:
                        break
            
            if best_ir is not None:
                return best_ir, {'level': 'L3', 'matched_smiles': best_smiles, 'similarity': best_sim, 'type': 'combo', 'fg_combo': combo, 'has_aromatic': has_aromatic}
        
        # 无相同组合：分别在每个单官能团区域匹配，叠加
        result_ir = np.zeros(Config.IR_POINTS, dtype=np.float32)
        matched_list = []
        similarity_sum = 0.0
        
        for fg in fg_types:
            candidates = self.library.fg_to_smiles.get(fg, [])
            if not candidates:
                continue
            
            # 过滤候选：芳香性必须一致
            filtered_candidates = []
            for s in candidates:
                s_has_aromatic = self._smiles_has_aromatic(s)
                if has_aromatic == s_has_aromatic:
                    filtered_candidates.append(s)
            
            if not filtered_candidates:
                filtered_candidates = candidates
            
            # 按指纹相似度排序
            best_smiles = None
            best_sim = -1.0
            best_ir = None
            
            for smiles in filtered_candidates:
                cand_fp = self.library.fp_cache.get(smiles)
                if cand_fp is None:
                    continue
                
                sim = DataStructs.TanimotoSimilarity(query_fp, cand_fp)
                if sim > best_sim:
                    best_sim = sim
                    best_smiles = smiles
                    best_ir = self.library.smiles_to_ir.get(smiles)
            
            if best_ir is not None:
                result_ir += best_ir
                matched_list.append(best_smiles)
                similarity_sum += best_sim
        
        if matched_list:
            avg_similarity = similarity_sum / len(matched_list)
            return result_ir, {'level': 'L3', 'matched_smiles': ' + '.join(matched_list), 'similarity': avg_similarity, 'type': 'multi_fg_叠加', 'fg_list': fg_types, 'has_aromatic': has_aromatic}
        
        return None, None
    
    def _aromatic_fallback_with_info(self, mol):
        """L4 芳香族降级匹配，返回 (IR, info_dict)"""
        # 获取芳香环信息
        try:
            Chem.SanitizeMol(mol)
            rdmolops.FastFindRings(mol)
        except:
            return None, None
        
        ring_info = mol.GetRingInfo()
        aromatic_rings = []
        for ring in ring_info.AtomRings():
            try:
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    aromatic_rings.append(set(ring))
            except:
                continue
        
        if not aromatic_rings:
            return None, None
        
        # 获取取代模式
        pattern = self.fg_utils.get_substitution_pattern(mol, aromatic_rings)
        if pattern is None:
            return None, None
        
        # 获取母体 IR
        parent_smiles = Config.AROMATIC_PARENT.get(pattern, "c1ccccc1")
        parent_ir = self.library.smiles_to_ir.get(parent_smiles)
        if parent_ir is None:
            return None, None
        
        parent_ir = np.array(parent_ir)
        
        # 提取取代基
        ring_atoms = aromatic_rings[0]
        substituent_mols = self._extract_substituents(mol, ring_atoms)
        
        # 叠加取代基 IR
        result_ir = parent_ir.copy()
        matched_substituents = []
        for sub_mol in substituent_mols:
            sub_ir, sub_info = self._functional_group_fallback_with_info(sub_mol)
            if sub_ir is None:
                return None, None
            result_ir += sub_ir
            if sub_info:
                matched_substituents.append(sub_info.get('matched_smiles', 'unknown'))
        
        return result_ir, {'level': 'L4', 'matched_smiles': f"{parent_smiles} + {' + '.join(matched_substituents)}", 'similarity': 0.6, 'pattern': pattern}
    
    def _similarity_match_with_info(self, mol):
        """L2 相似匹配，返回 (IR, info_dict)"""
        fg_types = self.fg_utils.get_fg_types(mol)
        query_fp = self.fp_gen.get_fingerprint(mol)
        query_size = mol.GetNumHeavyAtoms()
        
        # 确定候选集
        if not fg_types:
            candidates = self.library.all_smiles
        elif len(fg_types) == 1:
            candidates = self.library.fg_to_smiles.get(fg_types[0], self.library.all_smiles)
        else:
            combo = "|".join(sorted(fg_types))
            candidates = self.library.fg_combo_to_smiles.get(combo, self.library.all_smiles)
        
        bucket_key = query_fp.ToBitString()[:Config.FP_BUCKET_BITS]
        
        best_ir = None
        best_sim = -1.0
        best_smiles = None
        
        for smiles in candidates[:Config.L2_TOP_N]:
            cand_size = self.library.smiles_to_heavy.get(smiles, 0)
            if abs(cand_size - query_size) > max(5, query_size * 0.3):
                continue
            
            cand_fp = self.library.fp_cache.get(smiles)
            if cand_fp is None:
                continue
            
            if cand_fp.ToBitString()[:Config.FP_BUCKET_BITS] != bucket_key:
                continue
            
            sim = DataStructs.TanimotoSimilarity(query_fp, cand_fp)
            if sim > best_sim:
                best_sim = sim
                best_ir = self.library.smiles_to_ir.get(smiles)
                best_smiles = smiles
                if sim > 0.9:
                    break
        
        if best_ir is not None:
            return best_ir, {'level': 'L2', 'matched_smiles': best_smiles, 'similarity': best_sim}
        
        return None, None
    
    def _similarity_match(self, mol):
        """L2 相似匹配（兼容旧接口）"""
        ir, _ = self._similarity_match_with_info(mol)
        return ir
    
    def _exact_match(self, mol):
        """L1 精确匹配（兼容旧接口）"""
        ir, _ = self._exact_match_with_info(mol)
        return ir
    
    def _functional_group_fallback(self, mol):
        """L3 官能团降级匹配（兼容旧接口）"""
        ir, _ = self._functional_group_fallback_with_info(mol)
        return ir
    
    def _aromatic_fallback(self, mol):
        """L4 芳香族降级匹配（兼容旧接口）"""
        ir, _ = self._aromatic_fallback_with_info(mol)
        return ir
    
    def _has_aromatic_ring(self, mol):
        """检查分子是否含芳香环"""
        try:
            for atom in mol.GetAtoms():
                if atom.GetIsAromatic():
                    return True
        except:
            pass
        return False
    
    def _smiles_has_aromatic(self, smiles):
        """检查 SMILES 是否含芳香环"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            for atom in mol.GetAtoms():
                if atom.GetIsAromatic():
                    return True
        except:
            pass
        return False
    
    def _extract_substituents(self, mol, ring_atoms):
        """提取取代基（不含芳香环）"""
        from collections import deque
        
        substituents = []
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    continue
                if neighbor.GetIdx() not in ring_atoms:
                    sub_mol = self._extract_branch(mol, neighbor.GetIdx(), ring_atoms)
                    if sub_mol:
                        substituents.append(sub_mol)
                    break
        return substituents
    
    def _extract_branch(self, mol, start_idx, exclude_atoms):
        """提取分支"""
        from collections import deque
        
        visited = set()
        queue = deque([start_idx])
        visited.add(start_idx)
        
        while queue:
            idx = queue.popleft()
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                if neighbor.GetAtomicNum() == 1:
                    continue
                if n_idx in exclude_atoms:
                    continue
                if n_idx not in visited:
                    visited.add(n_idx)
                    queue.append(n_idx)
        
        if visited:
            rw_mol = Chem.RWMol(mol)
            all_atoms = set(range(mol.GetNumAtoms()))
            to_remove = all_atoms - visited
            for ridx in sorted(to_remove, reverse=True):
                try:
                    rw_mol.RemoveAtom(ridx)
                except:
                    continue
            try:
                frag = rw_mol.GetMol()
                for atom in frag.GetAtoms():
                    atom.SetIsAromatic(False)
                
                try:
                    Chem.SanitizeMol(frag)
                    Chem.SetAromaticity(frag)
                except:
                    return None
                
                frag = Chem.AddHs(frag)
                try:
                    Chem.SanitizeMol(frag)
                except:
                    return None
                return frag
            except:
                pass
        return None


# ==================== IR 预测器 ====================
class IRPredictor:
    def __init__(self, library):
        self.splitter = MoleculeSplitter()
        self.matcher = FragmentMatcher(library)
        self._cache = {}
    
    def predict(self, mol):
        """预测 IR 谱"""
        try:
            smiles = Chem.MolToSmiles(mol)
            if smiles in self._cache:
                return self._cache[smiles]
        except:
            pass
        
        # 拆分
        fragments = self.splitter.split(mol)
        
        if not fragments:
            return None
        
        # 匹配每个片段
        ir_list = []
        for frag in fragments:
            ir = self.matcher.match(frag)
            if ir is not None:
                ir_list.append(ir)
        
        if not ir_list:
            return None
        
        # 叠加
        combined = np.zeros(Config.IR_POINTS, dtype=np.float32)
        for ir in ir_list:
            combined += ir
        
        # 归一化
        min_val = np.min(combined)
        max_val = np.max(combined)
        if max_val > min_val:
            combined = (combined - min_val) / (max_val - min_val)
        else:
            combined = np.zeros(Config.IR_POINTS, dtype=np.float32)
        
        try:
            self._cache[Chem.MolToSmiles(mol)] = combined
        except:
            pass
        
        return combined


# ==================== 交叉验证 ====================
class CrossValidator:
    def __init__(self, molecules, smiles_to_ir):
        self.molecules = molecules
        self.smiles_to_ir = smiles_to_ir
        self.sample_count = 0
        self.max_samples = 20
    
    def run(self):
        """运行 3 折交叉验证"""
        # 分类：小分子 + 含芳香环小分子 入碎片库，其余大分子测试
        small_molecules = []      # ≤6重原子（无论是否含芳香环）
        aromatic_small = []       # 含芳香环且7-9重原子
        large_molecules = []      # 其他大分子（>9重原子，或不含芳香环的>6重原子）
        
        for m in self.molecules:
            heavy = m['heavy_atoms']
            has_aromatic = self._has_aromatic_ring(m['mol'])
            
            if heavy <= Config.SMALL_MOLECULE_MAX_HEAVY_ATOMS:
                # 6重原子以内，全部加入碎片库
                small_molecules.append(m)
            elif has_aromatic and heavy <= Config.AROMATIC_SMALL_MAX_HEAVY_ATOMS:
                # 含芳香环且7-9重原子，加入碎片库
                aromatic_small.append(m)
            else:
                # 其余大分子，用于交叉验证
                large_molecules.append(m)
        
        # 基础碎片库（始终保留）
        base_small_molecules = small_molecules + aromatic_small
        
        print(f"\n分子分类:")
        print(f"  小分子 (≤{Config.SMALL_MOLECULE_MAX_HEAVY_ATOMS}): {len(small_molecules)}")
        print(f"  含芳香环小分子 (7-{Config.AROMATIC_SMALL_MAX_HEAVY_ATOMS}): {len(aromatic_small)}")
        print(f"  碎片库基础分子: {len(base_small_molecules)}")
        print(f"  大分子 (待交叉验证): {len(large_molecules)}")
        
        if not large_molecules:
            print("无大分子，退出")
            return None
        
        # 3 折交叉验证
        kf = KFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=42)
        
        all_pearson = []
        cv_results = {}
        
        # 用于存储前20个分子的对比图
        self.sample_count = 0
        self.max_samples = 20
        
        # 尝试导入 tqdm
        try:
            from tqdm import tqdm
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False
        
        print(f"\n开始 {Config.CV_FOLDS} 折交叉验证...")
        overall_start = time.time()
        all_predictions = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(large_molecules)):
            fold_start = time.time()
            print(f"\n{'='*50}")
            print(f"Fold {fold+1}/{Config.CV_FOLDS}")
            print(f"{'='*50}")
            
            train_mols = [large_molecules[i] for i in train_idx]
            test_mols = [large_molecules[i] for i in test_idx]
            
            # 构建当前折的碎片库
            fold_library = FragmentLibrary()
            
            # 添加基础小分子（≤6重原子 + 含芳香环7-9重原子）
            for m in base_small_molecules:
                fg_types = FunctionalGroupUtils.get_fg_types(m['mol'])
                aromatic_pattern = self._get_aromatic_pattern(m['mol']) if self._has_aromatic_ring(m['mol']) else None
                fold_library.add_molecule(m['smiles'], m['ir'], m['heavy_atoms'], fg_types, aromatic_pattern)
            
            # 添加训练集大分子
            for m in train_mols:
                fg_types = FunctionalGroupUtils.get_fg_types(m['mol'])
                aromatic_pattern = self._get_aromatic_pattern(m['mol']) if self._has_aromatic_ring(m['mol']) else None
                fold_library.add_molecule(m['smiles'], m['ir'], m['heavy_atoms'], fg_types, aromatic_pattern)
            
            fold_library.finalize()
            
            # 预测
            predictor = IRPredictor(fold_library)
            fold_pearson = []
            fold_prediction_times = []
            
            test_total = len(test_mols)
            print(f"\n  预测进度:")
            
            if HAS_TQDM:
                iterator = tqdm(enumerate(test_mols), total=test_total, 
                               desc=f"    Fold {fold+1}", unit="mol")
            else:
                iterator = enumerate(test_mols)
            
            for i, m in iterator:
                if not HAS_TQDM and (i + 1) % 50 == 0:
                    print(f"    已处理: {i+1}/{test_total}")
                
                # 记录预测时间
                pred_start = time.time()
                pred_ir = predictor.predict(m['mol'])
                pred_time = time.time() - pred_start
                fold_prediction_times.append(pred_time)
                
                # 获取拆分片段
                fragments = predictor.splitter.split(m['mol'])
                fragment_smiles = []
                for frag in fragments[:10]:  # 最多保存10个片段
                    try:
                        frag_smiles = Chem.MolToSmiles(frag)
                        fragment_smiles.append(frag_smiles)
                    except:
                        fragment_smiles.append("None")
                
                # 获取匹配信息
                match_info = predictor.matcher.get_last_match_info()
                
                pred_record = {
                    'smiles': m['smiles'],
                    'fold': fold + 1,
                    'true_ir': m['ir'].tolist() if hasattr(m['ir'], 'tolist') else m['ir'],
                    'pred_ir': pred_ir.tolist() if pred_ir is not None and hasattr(pred_ir, 'tolist') else None,
                    'fragments': fragment_smiles,
                    'match_level': match_info.get('level', 'None') if match_info else 'None',
                    'matched_smiles': match_info.get('matched_smiles', 'None') if match_info else 'None',
                    'match_similarity': match_info.get('similarity', 0.0) if match_info else 0.0,
                    'pearson': None,
                    'pred_time_ms': pred_time * 1000
                }
                
                if pred_ir is not None:
                    # 归一化 true_ir
                    true_min = np.min(m['ir'])
                    true_max = np.max(m['ir'])
                    if true_max > true_min:
                        true_norm = (m['ir'] - true_min) / (true_max - true_min)
                    else:
                        true_norm = m['ir']
                    
                    pred_min = np.min(pred_ir)
                    pred_max = np.max(pred_ir)
                    if pred_max > pred_min:
                        pred_norm = (pred_ir - pred_min) / (pred_max - pred_min)
                    else:
                        pred_norm = pred_ir
                    
                    pearson = pearsonr(pred_norm, true_norm)[0]
                    fold_pearson.append(pearson)
                    pred_record['pearson'] = pearson
                    
                    # 低相似度诊断
                    if pearson < 0.5 and self.sample_count < self.max_samples:
                        # ... 诊断代码保持不变 ...
                        pass
                    
                    # 保存前20个对比图
                    if self.sample_count < self.max_samples:
                        self._save_comparison_plot(m['mol'], m['ir'], pred_ir, pearson, fold, self.sample_count)
                        self.sample_count += 1
                
                all_predictions.append(pred_record)
            
            if not HAS_TQDM:
                print(f"    完成: {test_total}/{test_total}")
            
            fold_time = time.time() - fold_start
            mean_pearson = np.mean(fold_pearson) if fold_pearson else 0
            mean_pred_time = np.mean(fold_prediction_times) * 1000 if fold_prediction_times else 0
            
            cv_results[f'fold_{fold+1}'] = {
                'test_size': len(test_mols),
                'success': len(fold_pearson),
                'mean_pearson': mean_pearson,
                'median_pearson': np.median(fold_pearson) if fold_pearson else 0,
                'std_pearson': np.std(fold_pearson) if fold_pearson else 0,
                'pearson_scores': fold_pearson,
                'mean_prediction_time_ms': mean_pred_time
            }
            all_pearson.extend(fold_pearson)
            
            print(f"\n  Fold {fold+1} 结果:")
            print(f"    测试集大小: {len(test_mols)}")
            print(f"    成功预测: {len(fold_pearson)}")
            print(f"    平均 Pearson: {mean_pearson:.4f}")
            print(f"    平均预测时间: {mean_pred_time:.1f}ms")
            print(f"    耗时: {fold_time:.2f}s")
        
        overall_time = time.time() - overall_start
        
        # 总体统计
        overall = {
            'total_tested': len(all_pearson),
            'mean_pearson': np.mean(all_pearson) if all_pearson else 0,
            'median_pearson': np.median(all_pearson) if all_pearson else 0,
            'std_pearson': np.std(all_pearson) if all_pearson else 0,
            'total_time_seconds': overall_time
        }
        
        print(f"\n{'='*50}")
        print(f"交叉验证完成")
        print(f"总耗时: {overall_time:.2f}s")
        print(f"{'='*50}")
        
        return {'cv_results': cv_results, 
                'overall': overall, 
                'all_scores': all_pearson, 
                'predictions': all_predictions
        }
    
    
    def _has_aromatic_ring(self, mol):
        """检查分子是否含芳香环"""
        try:
            for atom in mol.GetAtoms():
                if atom.GetIsAromatic():
                    return True
        except:
            pass
        return False
    
    
    def _get_aromatic_pattern(self, mol):
        """获取芳香环的取代模式"""
        try:
            Chem.SanitizeMol(mol)
            rdmolops.FastFindRings(mol)
            ring_info = mol.GetRingInfo()
            aromatic_rings = []
            for ring in ring_info.AtomRings():
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    aromatic_rings.append(set(ring))
            
            if not aromatic_rings:
                return None
            
            return FunctionalGroupUtils.get_substitution_pattern(mol, aromatic_rings)
        except:
            return None
        
    def _save_comparison_plot(self, mol, true_ir, pred_ir, pearson, fold, sample_idx):
        """保存预测与真实的 IR 对比图"""
        wavenumbers = Config.load_wavenumbers()
        smiles = Chem.MolToSmiles(mol)
        
        # 归一化 true_ir
        true_min = np.min(true_ir)
        true_max = np.max(true_ir)
        if true_max > true_min:
            true_norm = (true_ir - true_min) / (true_max - true_min)
        else:
            true_norm = true_ir
        
        # 归一化 pred_ir
        pred_min = np.min(pred_ir)
        pred_max = np.max(pred_ir)
        if pred_max > pred_min:
            pred_norm = (pred_ir - pred_min) / (pred_max - pred_min)
        else:
            pred_norm = pred_ir
        
        plt.rcParams.update({
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": 'sans-serif',
            "font.sans-serif": ["Arial"],
            "axes.linewidth": 0.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        })
        
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.plot(wavenumbers, true_norm, 'b-', label='True IR', linewidth=1, alpha=0.8)
        ax.plot(wavenumbers, pred_norm, 'r--', label='Predicted IR', linewidth=1, alpha=0.8)
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'Fold {fold+1}, Sample {sample_idx+1}: {smiles[:50]}\nPearson = {pearson:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = Config.FIGURE_DIR / f'fold{fold+1}_sample{sample_idx+1}_{pearson:.3f}.png'
        plt.savefig(save_path, dpi=150)
        plt.close()


# ==================== 绘图函数 ====================
def plot_hist(scores, filename='pearson_distribution.png'):
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": 'sans-serif',
        "font.sans-serif": ["Arial"],
        "axes.linewidth": 0.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.hist(scores, bins=50, color='salmon', edgecolor='w', alpha=0.7, range=(0, 1))
    ax.set_xlim(0, 1)
    ax.set_xlabel('Pearson Correlation Coefficient', fontweight='bold')
    ax.set_ylabel('Number of Molecules', fontweight='bold')
    ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
    ax.axvline(np.median(scores), color='blue', linestyle='--', label=f'Median: {np.median(scores):.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(Config.FIGURE_DIR / filename, dpi=300)
    plt.close()
    print(f"图已保存: {Config.FIGURE_DIR / filename}")


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("分子碎片化 IR 谱预测系统 V0.8.8")
    print("=" * 60)
    
    create_directories()
    
    # 检查缓存
    if Config.FRAGMENT_LIBRARY_FILE.exists() and Config.PREDICTION_RESULTS_FILE.exists():
        print("\n加载已保存的结果...")
        with gzip.open(Config.PREDICTION_RESULTS_FILE, 'rb') as f:
            results = pickle.load(f)
        print(f"总体平均 Pearson: {results['overall']['mean_pearson']:.4f}")
        plot_hist(results['all_scores'], 'pearson_distribution_v088.png')
        return results
    
    # 加载数据
    print("\n加载数据库...")
    loader = DatabaseLoader(Config.DB_PATH)
    molecules, smiles_to_ir = loader.load_all_molecules()
    
    # 交叉验证
    cv = CrossValidator(molecules, smiles_to_ir)
    results = cv.run()
    
    if results is None:
        print("无结果")
        return None
    
    # 保存结果（统计信息）
    with gzip.open(Config.PREDICTION_RESULTS_FILE, 'wb') as f:
        pickle.dump({
            'cv_results': results['cv_results'],
            'overall': results['overall'],
            'all_scores': results['all_scores']
        }, f)
    print(f"\n统计结果已保存: {Config.PREDICTION_RESULTS_FILE}")
    
    # 保存预测详情（包含光谱、片段、匹配信息）
    if 'predictions' in results:
        with gzip.open(Config.PREDICTION_DETAILS_FILE, 'wb') as f:
            pickle.dump(results['predictions'], f)
        print(f"预测详情已保存: {Config.PREDICTION_DETAILS_FILE}")
        print(f"  包含 {len(results['predictions'])} 个分子的详细信息")
        print(f"  每个分子包含: SMILES, 真实IR, 预测IR, 拆分片段, 匹配级别, 匹配分子, Pearson")
    
    # 输出统计
    print("\n" + "=" * 60)
    print("最终统计")
    print("=" * 60)
    overall = results['overall']
    print(f"\n测试分子数: {overall['total_tested']}")
    print(f"平均 Pearson: {overall['mean_pearson']:.4f}")
    print(f"中位数 Pearson: {overall['median_pearson']:.4f}")
    print(f"标准差: {overall['std_pearson']:.4f}")
    
    for fold, res in results['cv_results'].items():
        print(f"\n{fold}: 平均={res['mean_pearson']:.4f}, 成功={res['success']}/{res['test_size']}")
    
    # 绘图
    if results['all_scores']:
        plot_hist(results['all_scores'], 'pearson_distribution_v088.png')
    
    print("\n完成！")
    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
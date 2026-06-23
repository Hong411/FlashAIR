# -*- coding: utf-8 -*-
"""
FlashAIRx: Validation Engine for FlashAIR
Physics-based IR spectrum correction using GFN2-xTB prior + GPR residual

This module implements the FlashAIRx validation engine described in:
"FlashAIR: Fast and Interpretable IR Spectral Analysis via Physical 
Prior and Residual Correction"

Core workflow:
1. Generate GFN2-xTB prior spectrum from 3D molecular geometry
2. Apply functional-group-stratified GPR residual correction (Phase I)
3. Optional: DFT-to-experiment calibration (Phase II)

Author: FlashAIR Team
"""

import os
import json
import pickle
import sqlite3
import math
import time
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import gpflow

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Global configuration for FlashAIRx"""
    
    # Data paths
    DATA_DIR = Path('D:/chemdata/FlashAIR_Proj/Data')
    MODEL_DIR = Path('D:/chemdata/FlashAIR_Proj/Model')
    DATABASE_PATH = Path('D:/chemdata/Database/SOMIR0907d.db')
    
    # Spectral parameters
    WAVENUMBER_START = 550
    WAVENUMBER_END = 3846
    WAVENUMBER_STEP = 4
    IR_POINTS = 825
    LORENTZIAN_WIDTH = 24  # FWHM in cm^-1
    SCALING_FACTOR = 0.97  # Frequency scaling for xTB
    
    # GPR training parameters
    TRAIN_RATIO = 0.7
    TEST_RATIO = 0.3
    RANDOM_STATE = 23
    N_TRAIN_STEPS = 1000
    
    # Functional group definitions (15 groups)
    FG_SMARTS = {
        "Alkane": "[#6]",
        "Alkene": "[CX3]=[C]",
        "Alkyne": "[CX2]#C",
        "Aromatic": "[a]",
        "Alcohol": "[#6][OX2H]",
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
    
    # Functional groups used for stratification (excluding Alkane)
    FG_STRATIFY = [
        "Alcohol", "Alkene", "Alkyne", "Amide", "Amine", "Aromatic",
        "Carboxylic Acid", "Ester", "Ether", "Aldehyde", "Ketone", 
        "Nitrile", "Nitro", "Imine", "Halide"
    ]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_wavenumbers():
    """Load standardized wavenumber grid"""
    return np.arange(Config.WAVENUMBER_START, Config.WAVENUMBER_END + Config.WAVENUMBER_STEP, Config.WAVENUMBER_STEP)


def normalize_spectrum(data):
    """Min-max normalization to [0, 1]"""
    arr = np.array(data, dtype=np.float64)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    return arr


def lorentzian(x, peak, height, width):
    """Lorentzian line shape function"""
    a = (width / 2) ** 2
    return height * a / ((peak - x) ** 2 + a)


class SpectrumGenerator:
    """Generate broadened IR spectrum from peaks"""
    
    def __init__(self, start=550, end=3846, num_pts=825, width=24):
        self.start = start
        self.end = end
        self.num_pts = num_pts
        self.width = width
        self.xvalues = np.linspace(start, end, num_pts)
    
    def generate(self, peaks, scaling=1.0):
        """
        Generate spectrum from peak list
        
        Args:
            peaks: list of (frequency, intensity) tuples
            scaling: frequency scaling factor
        
        Returns:
            tuple: (wavenumbers, raw_intensity, normalized_intensity)
        """
        spectrum = np.zeros(self.num_pts)
        
        for freq, intensity in peaks:
            freq_scaled = freq * scaling
            for i, x in enumerate(self.xvalues):
                spectrum[i] += lorentzian(x, freq_scaled, intensity, self.width)
        
        # Clip small values
        spectrum = np.clip(spectrum, 1e-20, None)
        
        return self.xvalues, spectrum, normalize_spectrum(spectrum)


# ============================================================================
# METRICS
# ============================================================================

def pearson_correlation(u, v):
    """Pearson correlation coefficient"""
    u, v = np.array(u), np.array(v)
    try:
        return pearsonr(u, v)[0]
    except:
        return 0.0


def spearman_correlation(u, v):
    """Spearman rank correlation coefficient"""
    u, v = np.array(u), np.array(v)
    try:
        return spearmanr(u, v)[0]
    except:
        return 0.0


def spectral_info_similarity(p, q, epsilon=1e-10):
    """
    Spectral Information Similarity (SIS)
    SIS = 1 / (1 + SID), SID = D(p||q) + D(q||p)
    """
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: {p.shape} vs {q.shape}")
    
    # Normalize to probability distributions
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    p = p / p.sum()
    q = q / q.sum()
    
    # Calculate symmetric divergence
    with np.errstate(divide='ignore', invalid='ignore'):
        D_pq = np.sum(p * np.log(p / q))
        D_qp = np.sum(q * np.log(q / p))
    
    SID = np.nan_to_num(D_pq) + np.nan_to_num(D_qp)
    return 1 / (1 + SID)


def simple_matching_score(u, v):
    """Simple matching score (dot product squared / product of norms)"""
    u, v = np.array(u), np.array(v)
    numerator = np.square(np.sum(u * v))
    denominator = np.sum(np.square(u)) * np.sum(np.square(v))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def rmsd(u, v):
    """Root mean square deviation"""
    u, v = np.array(u), np.array(v)
    return math.sqrt(np.mean((u - v) ** 2))


def euclidean_similarity(u, v):
    """Euclidean similarity from Grimme literature"""
    u, v = np.array(u), np.array(v)
    numerator = np.sum((u - v) ** 2)
    denominator = np.sum(v ** 2)
    if denominator == 0:
        return 0.0
    return (1 + numerator / denominator) ** -1


def cosine_similarity(u, v):
    """Cosine similarity"""
    u, v = np.array(u), np.array(v)
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
        return 0.0
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# ============================================================================
# FUNCTIONAL GROUP UTILITIES
# ============================================================================

from rdkit import Chem
from rdkit.Chem import rdmolops


def detect_functional_groups(mol):
    """
    Detect functional groups in a molecule using SMARTS patterns
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        list: functional group names present in the molecule
    """
    fg_list = []
    
    for fg_name, smarts in Config.FG_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        if mol.HasSubstructMatch(pattern):
            fg_list.append(fg_name)
    
    # Check for alkane (only C and H, no other functional groups)
    if len(fg_list) == 0:
        # Check if molecule contains only C and H
        has_hetero = False
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in [1, 6]:
                has_hetero = True
                break
        if not has_hetero:
            fg_list.append("Alkane")
    
    return fg_list


def get_element_composition(mol):
    """Get element counts for a molecule"""
    counts = {'O': 0, 'N': 0, 'halogen': 0, 'C': 0, 'H': 0}
    halogen_elements = {9, 17, 35, 53}
    
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 8:
            counts['O'] += 1
        elif atomic_num == 7:
            counts['N'] += 1
        elif atomic_num in halogen_elements:
            counts['halogen'] += 1
        elif atomic_num == 6:
            counts['C'] += 1
        elif atomic_num == 1:
            counts['H'] += 1
    
    return counts


# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoader:
    """Load molecular data and spectra from database"""
    
    def __init__(self, db_path):
        self.db_path = db_path
    
    def load_spectrum(self, mol_id):
        """Load IR spectrum for a molecule from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT wavenumber, intensity FROM sim_spectrum WHERE mol_id = ?',
            (mol_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result is not None:
            wavenumber = json.loads(result[0])
            intensity = json.loads(result[1])
            return wavenumber, intensity
        return None, None
    
    def load_molecule_info(self, mol_id):
        """Load molecular information from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT mol_smiles, mol_functional_groups FROM molecular_information WHERE mol_id = ?',
            (mol_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result is not None:
            return {'smiles': result[0], 'functional_groups': result[1]}
        return None


# ============================================================================
# XTB SPECTRUM GENERATION
# ============================================================================

class XTBInterface:
    """Interface for GFN2-xTB spectrum generation"""
    
    def __init__(self, xtb_path=None):
        self.xtb_path = xtb_path or 'xtb'
    
    def parse_output(self, file_path):
        """
        Parse xTB output file to extract frequencies and intensities
        
        Args:
            file_path: path to xTB output file
            
        Returns:
            tuple: (frequencies, intensities)
        """
        frequencies = []
        intensities = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for i, line in enumerate(lines):
                if 'Frequencies --' in line:
                    freq_line = lines[i].strip().split()[2:]
                    # IR intensities are typically 3 lines below
                    if i + 3 < len(lines):
                        inten_line = lines[i + 3].strip().split()[3:]
                        for freq, inten in zip(freq_line, inten_line):
                            frequencies.append(float(freq))
                            intensities.append(float(inten))
        
        return frequencies, intensities
    
    def generate_spectrum(self, output_file):
        """
        Generate IR spectrum from xTB output file
        
        Args:
            output_file: path to xTB output file
            
        Returns:
            dict: spectrum data with wavenumbers and intensities
        """
        freq, intensity = self.parse_output(output_file)
        spec_gen = SpectrumGenerator(
            start=Config.WAVENUMBER_START,
            end=Config.WAVENUMBER_END,
            num_pts=Config.IR_POINTS,
            width=Config.LORENTZIAN_WIDTH
        )
        wavenumber, raw_intensity, norm_intensity = spec_gen.generate(
            list(zip(freq, intensity)),
            scaling=Config.SCALING_FACTOR
        )
        
        return {
            'wavenumber': wavenumber,
            'raw_intensity': raw_intensity,
            'normalized_intensity': norm_intensity,
            'frequencies': freq,
            'intensities': intensity
        }


# ============================================================================
# GPR TRAINING
# ============================================================================

class GPRTrainer:
    """Train Gaussian Process Regression models for spectral correction"""
    
    def __init__(self, dtype=tf.float64):
        self.dtype = dtype
        self.models = {}
        self.scalers = {}
    
    def train_tf(self, X_train, y_train, n_steps=1000, lr=0.01):
        """
        Train GPR model using TensorFlow/GPflow
        
        Args:
            X_train: input spectra (n_samples, n_features)
            y_train: target spectra (n_samples, n_features)
            n_steps: number of training steps
            lr: learning rate
        
        Returns:
            gpflow.models.GPR: trained model
        """
        # Standardize
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X_train)
        y_scaled = scaler_y.fit_transform(y_train)
        
        # Convert to TensorFlow tensors
        X_tf = tf.convert_to_tensor(X_scaled, dtype=self.dtype)
        y_tf = tf.convert_to_tensor(y_scaled, dtype=self.dtype)
        
        # RBF kernel
        kernel = gpflow.kernels.SquaredExponential(
            lengthscales=np.float64(1.0),
            variance=np.float64(1.0)
        )
        
        # Create GPR model
        gpr = gpflow.models.GPR(data=(X_tf, y_tf), kernel=kernel)
        
        # Fix likelihood variance (jitter)
        gpr.likelihood.variance.assign(np.float64(1e-4))
        gpflow.utilities.set_trainable(gpr.likelihood.variance, True)
        
        # Adam optimizer
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        
        @tf.function
        def optimization_step():
            with tf.GradientTape() as tape:
                loss = gpr.training_loss()
            gradients = tape.gradient(loss, gpr.trainable_variables)
            optimizer.apply_gradients(zip(gradients, gpr.trainable_variables))
            return loss
        
        # Training loop
        for step in range(n_steps):
            loss = optimization_step()
            if (step + 1) % 500 == 0:
                print(f"    Step {step+1}/{n_steps}, Loss: {loss:.4f}")
        
        return gpr, scaler_X, scaler_y
    
    def predict_tf(self, gpr, X_test, scaler_X, scaler_y):
        """Predict using trained GPR model"""
        X_scaled = scaler_X.transform(X_test)
        X_tf = tf.convert_to_tensor(X_scaled, dtype=self.dtype)
        
        mean_scaled, _ = gpr.predict_f(X_tf)
        mean = scaler_y.inverse_transform(mean_scaled.numpy())
        
        return mean
    
    def train_stratified(self, xtb_dict, dft_dict, fg_groups, n_steps=1000):
        """
        Train stratified GPR models by functional group composition
        
        Args:
            xtb_dict: dict mapping functional group string -> list of xTB spectra
            dft_dict: dict mapping functional group string -> list of DFT spectra
            fg_groups: dict mapping complexity level -> list of functional group keys
            n_steps: training steps per model
        
        Returns:
            dict: trained models
        """
        models = {}
        
        for group_name, group_keys in fg_groups.items():
            print(f"\nTraining group: {group_name}")
            
            for fg_key in group_keys:
                if fg_key not in xtb_dict:
                    continue
                    
                X = np.array(xtb_dict[fg_key])
                y = np.array(dft_dict[fg_key])
                
                if len(X) < 20:
                    print(f"  {fg_key}: insufficient data ({len(X)} samples), skipping")
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=Config.TEST_RATIO,
                    random_state=Config.RANDOM_STATE
                )
                
                print(f"  {fg_key}: {len(X_train)} train, {len(X_test)} test")
                
                # Train GPR
                gpr, scaler_X, scaler_y = self.train_tf(
                    X_train, y_train, n_steps=n_steps
                )
                
                models[fg_key] = {
                    'model': gpr,
                    'scaler_X': scaler_X,
                    'scaler_y': scaler_y
                }
        
        self.models = models
        return models


# ============================================================================
# FlashAIRx MAIN CLASS
# ============================================================================

class FlashAIRx:
    """
    FlashAIRx: Validation Engine
    
    Phase I: GFN2-xTB + GPR residual correction (DFT level)
    Phase II: DFT-to-experiment calibration (optional)
    
    Usage:
        flashairx = FlashAIRx()
        flashairx.load_models('model_dir')
        
        # Predict DFT-level spectrum
        result = flashairx.predict_phase_I(xTB_spectrum, functional_groups)
        
        # Predict experimental-level spectrum
        result = flashairx.predict_phase_II(xTB_spectrum, functional_groups)
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize FlashAIRx
        
        Args:
            model_dir: directory containing trained GPR models
        """
        self.model_dir = Path(model_dir) if model_dir else Config.MODEL_DIR
        self.models = {}
        self.scalers = {}
        self.gpr_trainer = GPRTrainer()
        self.xtb_interface = XTBInterface()
        self.spec_gen = SpectrumGenerator(
            start=Config.WAVENUMBER_START,
            end=Config.WAVENUMBER_END,
            num_pts=Config.IR_POINTS,
            width=Config.LORENTZIAN_WIDTH
        )
    
    def load_models(self, model_dir=None):
        """
        Load pre-trained GPR models
        
        Args:
            model_dir: directory containing model files
        """
        if model_dir:
            self.model_dir = Path(model_dir)
        
        # Load models using pickle
        model_files = list(self.model_dir.glob('*.pkl'))
        for model_file in model_files:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            group_name = model_file.stem
            self.models[group_name] = model_data.get('model')
            self.scalers[group_name] = {
                'X': model_data.get('scaler_X'),
                'y': model_data.get('scaler_y')
            }
        
        print(f"Loaded {len(self.models)} GPR models from {self.model_dir}")
        return self.models
    
    def save_models(self, model_dir=None):
        """Save trained models"""
        if model_dir:
            self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        for group_name, model_data in self.models.items():
            model_file = self.model_dir / f"{group_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
        
        print(f"Saved {len(self.models)} models to {self.model_dir}")
    
    def get_functional_group_key(self, fg_list):
        """
        Get stratification key for functional group list
        
        Args:
            fg_list: list of functional group names
        
        Returns:
            str: sorted, comma-separated functional group string
        """
        # Remove alkane if present (for stratification)
        fg_filtered = [fg for fg in fg_list if fg != 'Alkane']
        
        if not fg_filtered:
            return 'alkane'
        
        return ','.join(sorted(fg_filtered))
    
    def find_model(self, fg_list):
        """
        Find the best matching GPR model for given functional groups
        
        Args:
            fg_list: list of functional group names
        
        Returns:
            tuple: (model, scaler_X, scaler_y, match_type)
        """
        fg_key = self.get_functional_group_key(fg_list)
        
        # Try exact match
        if fg_key in self.models:
            return (
                self.models[fg_key]['model'],
                self.models[fg_key]['scaler_X'],
                self.models[fg_key]['scaler_y'],
                'exact'
            )
        
        # Try sub-combination (remove one functional group at a time)
        fg_list_filtered = [fg for fg in fg_list if fg != 'Alkane']
        
        for i in range(len(fg_list_filtered) - 1, 0, -1):
            # Try all combinations of size i
            from itertools import combinations
            for combo in combinations(fg_list_filtered, i):
                combo_key = ','.join(sorted(combo))
                if combo_key in self.models:
                    return (
                        self.models[combo_key]['model'],
                        self.models[combo_key]['scaler_X'],
                        self.models[combo_key]['scaler_y'],
                        f'subset_{i}'
                    )
        
        # Try single functional group
        for fg in fg_list_filtered:
            if fg in self.models:
                return (
                    self.models[fg]['model'],
                    self.models[fg]['scaler_X'],
                    self.models[fg]['scaler_y'],
                    'single'
                )
        
        # Fallback: use alkane model or None
        if 'alkane' in self.models:
            return (
                self.models['alkane']['model'],
                self.models['alkane']['scaler_X'],
                self.models['alkane']['scaler_y'],
                'fallback_alkane'
            )
        
        return None, None, None, 'none'
    
    def predict_phase_I(self, xtb_spectrum, fg_list):
        """
        Phase I: Correct xTB spectrum to DFT level
        
        Args:
            xtb_spectrum: xTB spectrum (numpy array)
            fg_list: list of functional groups in the molecule
        
        Returns:
            dict: prediction results including corrected spectrum
        """
        # Find matching model
        model, scaler_X, scaler_y, match_type = self.find_model(fg_list)
        
        if model is None:
            return {
                'success': False,
                'message': f'No model found for functional groups: {fg_list}',
                'corrected_spectrum': xtb_spectrum,
                'match_type': 'none'
            }
        
        # Prepare input (reshape to 2D)
        X_test = np.array(xtb_spectrum).reshape(1, -1)
        
        # Predict residual
        pred = self.gpr_trainer.predict_tf(model, X_test, scaler_X, scaler_y)
        residual = pred[0]
        
        # Corrected spectrum
        corrected = np.array(xtb_spectrum) + residual
        
        return {
            'success': True,
            'xtb_spectrum': np.array(xtb_spectrum),
            'corrected_spectrum': normalize_spectrum(corrected),
            'residual': residual,
            'match_type': match_type,
            'matched_groups': fg_list if match_type == 'exact' else None
        }
    
    def predict_phase_II(self, xtb_spectrum, fg_list, calibration_model=None):
        """
        Phase II: Correct DFT spectrum to experimental level
        
        Args:
            xtb_spectrum: xTB spectrum (numpy array)
            fg_list: list of functional groups in the molecule
            calibration_model: DFT-to-experiment calibration model
        
        Returns:
            dict: prediction results including experimental-level spectrum
        """
        # First, get DFT-level correction
        result_I = self.predict_phase_I(xtb_spectrum, fg_list)
        
        if not result_I['success']:
            return result_I
        
        # Apply DFT-to-experiment calibration if available
        if calibration_model is not None:
            dft_spectrum = result_I['corrected_spectrum']
            # Apply calibration (placeholder - implement actual calibration)
            # experimental_spectrum = calibration_model.predict(dft_spectrum)
            experimental_spectrum = dft_spectrum  # Placeholder
        else:
            experimental_spectrum = result_I['corrected_spectrum']
        
        result_I['experimental_spectrum'] = normalize_spectrum(experimental_spectrum)
        result_I['phase'] = 'II'
        
        return result_I
    
    def predict_from_mol(self, mol, xtb_output_file=None, optimize=True):
        """
        Predict spectrum directly from molecular structure
        
        Args:
            mol: RDKit Mol object or SMILES string
            xtb_output_file: optional pre-computed xTB output file
            optimize: whether to run xTB optimization
        
        Returns:
            dict: prediction results
        """
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                return {'success': False, 'message': 'Invalid SMILES'}
        
        # Detect functional groups
        fg_list = detect_functional_groups(mol)
        
        # Generate xTB spectrum
        if xtb_output_file is not None:
            # Use existing xTB output
            xtb_result = self.xtb_interface.generate_spectrum(xtb_output_file)
            xtb_spectrum = xtb_result['normalized_intensity']
        else:
            # TODO: Run xTB calculation
            # For now, return placeholder
            return {
                'success': False,
                'message': 'xTB calculation not implemented in this version'
            }
        
        # Phase I correction
        result = self.predict_phase_I(xtb_spectrum, fg_list)
        
        return result
    
    def evaluate(self, xtb_list, dft_list, fg_list):
        """
        Evaluate FlashAIRx performance on a dataset
        
        Args:
            xtb_list: list of xTB spectra
            dft_list: list of DFT reference spectra
            fg_list: list of functional group lists for each molecule
        
        Returns:
            dict: evaluation metrics
        """
        corrected_list = []
        match_types = []
        
        for xtb, fg in zip(xtb_list, fg_list):
            result = self.predict_phase_I(xtb, fg)
            if result['success']:
                corrected_list.append(result['corrected_spectrum'])
                match_types.append(result['match_type'])
        
        # Calculate metrics
        metrics = {
            'pearson': [],
            'spearman': [],
            'sis': [],
            'match_types': match_types
        }
        
        for corrected, dft in zip(corrected_list, dft_list):
            metrics['pearson'].append(pearson_correlation(corrected, dft))
            metrics['spearman'].append(spearman_correlation(corrected, dft))
            metrics['sis'].append(spectral_info_similarity(corrected, dft))
        
        metrics['mean_pearson'] = np.mean(metrics['pearson'])
        metrics['mean_spearman'] = np.mean(metrics['spearman'])
        metrics['mean_sis'] = np.mean(metrics['sis'])
        
        return metrics


# ============================================================================
# TRAINING SCRIPT
# ============================================================================

def train_flashairx_models(data_dict, output_dir):
    """
    Train FlashAIRx GPR models from data dictionary
    
    Args:
        data_dict: dict containing xtb and dft spectra organized by functional group
        output_dir: directory to save trained models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = GPRTrainer()
    
    # Group data by complexity (single, dual, triple, 4+ functional groups)
    # Data should be pre-organized by functional group combination
    fg_groups = {
        'single': [],  # Single functional group
        'dual': [],    # Two functional groups
        'triple': [],  # Three functional groups
        'quad': []     # Four or more functional groups
    }
    
    # Example training loop (actual implementation depends on data format)
    for complexity, fg_list in fg_groups.items():
        print(f"\nTraining {complexity} functional group models...")
        
        for fg_key in fg_list:
            if fg_key not in data_dict:
                continue
                
            X_train = np.array(data_dict[fg_key]['xtb'])
            y_train = np.array(data_dict[fg_key]['dft'])
            
            if len(X_train) < 20:
                continue
            
            gpr, scaler_X, scaler_y = trainer.train_tf(X_train, y_train)
            
            # Save model
            model_data = {
                'model': gpr,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y
            }
            
            with open(output_dir / f'{fg_key}.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"  {fg_key}: {len(X_train)} samples, model saved")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("FlashAIRx: Validation Engine Training")
    print("=" * 60)
    
    # Example usage
    flashairx = FlashAIRx()
    
    # Load pre-trained models if available
    model_path = Path('D:/chemdata/FlashAIR_Proj/Model/flashairx_models')
    if model_path.exists():
        flashairx.load_models(model_path)
    
    print("\nFlashAIRx ready!")

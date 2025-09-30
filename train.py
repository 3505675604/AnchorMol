import os
import sys
import json
import random
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import pickle
from collections import defaultdict
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from embeding import MultiModalFeatureExtractor 

# Classification metrics
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
# Regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

torch.autograd.set_detect_anomaly(True)

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Using random split instead of scaffold split.")

# Transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append('/root/autodl-tmp')



# ================== JSON serialization helper functions ==================

def convert_to_json_serializable(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    else:
        return obj



# ================== Fixed pseudo-pair generator - supports configurable k ==================

class AdvancedPseudoPairGenerator(nn.Module):
    """Fixed pseudo-pair generator - supports configurable number of hard negatives k"""
    
    def __init__(
        self, 
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        projection_dim: int = 128,
        temperature: float = 0.07,
        learnable_temp: bool = True,
        use_projection_head: bool = True,
        use_feature_alignment: bool = True,
        hard_negative_mining: bool = True,
        hard_negative_k: int = 10,  # New: configurable number of hard negatives
        hard_negative_ratio: float = 0.25,  # New: upper limit of k as a ratio of batch_size
        momentum: float = 0.999,
        queue_size: int = 4096,
        use_memory_bank: bool = False,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.similarity_metric = similarity_metric
        self.hard_negative_mining = hard_negative_mining
        self.hard_negative_k = hard_negative_k  # Save k value
        self.hard_negative_ratio = hard_negative_ratio  # Save ratio
        self.momentum = momentum
        self.use_memory_bank = use_memory_bank
        self.queue_size = queue_size
        
        # Temperature parameter
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # Projection heads for each modality
        if use_projection_head:
            self.proj_head_1 = self._build_projection_head(embedding_dim, hidden_dim, projection_dim, dropout_rate)
            self.proj_head_2 = self._build_projection_head(embedding_dim, hidden_dim, projection_dim, dropout_rate)
            self.proj_head_3 = self._build_projection_head(embedding_dim, hidden_dim, projection_dim, dropout_rate)
        else:
            self.proj_head_1 = nn.Identity()
            self.proj_head_2 = nn.Identity()
            self.proj_head_3 = nn.Identity()
        
        # Feature alignment layers
        self.use_feature_alignment = use_feature_alignment
        if use_feature_alignment:
            feat_dim = projection_dim if use_projection_head else embedding_dim
            self.align_1_to_2 = nn.Linear(feat_dim, feat_dim)
            self.align_1_to_3 = nn.Linear(feat_dim, feat_dim)
            self.align_2_to_3 = nn.Linear(feat_dim, feat_dim)
        
        # Memory bank for negative samples
        if use_memory_bank:
            feat_dim = projection_dim if use_projection_head else embedding_dim
            self.register_buffer("queue_1", torch.randn(feat_dim, queue_size))
            self.register_buffer("queue_2", torch.randn(feat_dim, queue_size))
            self.register_buffer("queue_3", torch.randn(feat_dim, queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            
            # Normalize queues
            self.queue_1 = F.normalize(self.queue_1, dim=0)
            self.queue_2 = F.normalize(self.queue_2, dim=0)
            self.queue_3 = F.normalize(self.queue_3, dim=0)
        
        # Momentum encoders
        if momentum > 0:
            self.momentum_proj_1 = self._build_projection_head(embedding_dim, hidden_dim, projection_dim, 0)
            self.momentum_proj_2 = self._build_projection_head(embedding_dim, hidden_dim, projection_dim, 0)
            self.momentum_proj_3 = self._build_projection_head(embedding_dim, hidden_dim, projection_dim, 0)
            
            # No gradients needed
            for param in self.momentum_proj_1.parameters():
                param.requires_grad = False
            for param in self.momentum_proj_2.parameters():
                param.requires_grad = False
            for param in self.momentum_proj_3.parameters():
                param.requires_grad = False
    
    def _build_projection_head(self, input_dim, hidden_dim, output_dim, dropout_rate):
        """Build projection head - fixed version, no inplace ops"""
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())  # Not using inplace=True
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Second layer
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())  # Not using inplace=True
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        return nn.Sequential(*layers)
    
    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update for momentum encoders"""
        if hasattr(self, 'momentum_proj_1'):
            for param_q, param_k in zip(self.proj_head_1.parameters(), self.momentum_proj_1.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            for param_q, param_k in zip(self.proj_head_2.parameters(), self.momentum_proj_2.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            for param_q, param_k in zip(self.proj_head_3.parameters(), self.momentum_proj_3.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_1, keys_2, keys_3):
        """Update memory bank"""
        if not self.use_memory_bank:
            return

        batch_size = keys_1.shape[0]
        ptr = int(self.queue_ptr)

        # Ensure no overflow
        if ptr + batch_size > self.queue_size:
            batch_size = self.queue_size - ptr
            keys_1 = keys_1[:batch_size]
            keys_2 = keys_2[:batch_size]
            keys_3 = keys_3[:batch_size]

        # Update queues
        self.queue_1[:, ptr:ptr + batch_size] = keys_1.T
        self.queue_2[:, ptr:ptr + batch_size] = keys_2.T
        self.queue_3[:, ptr:ptr + batch_size] = keys_3.T

        # Move pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def compute_similarity(self, z1, z2):
        """Compute similarity matrix - fixed version"""
        z1 = z1.clone()
        z2 = z2.clone()
        
        if self.similarity_metric == 'cosine':
            z1_norm = F.normalize(z1, p=2, dim=1)
            z2_norm = F.normalize(z2, p=2, dim=1)
            sim = torch.mm(z1_norm, z2_norm.t())
        elif self.similarity_metric == 'euclidean':
            sim = -torch.cdist(z1, z2, p=2)
        elif self.similarity_metric == 'dot':
            sim = torch.mm(z1, z2.t())
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        if isinstance(self.temperature, torch.Tensor):
            temp = self.temperature.clone()
        else:
            temp = self.temperature
        
        return sim / temp
    
    def hard_negative_sampling(self, sim_matrix, labels):
        """Hard negative mining - supports configurable k"""
        if not self.hard_negative_mining:
            return sim_matrix

        batch_size = sim_matrix.size(0)
        device = sim_matrix.device

        pos_mask = torch.eye(batch_size, device=device).bool()

        neg_sim = sim_matrix.clone()
        neg_sim = neg_sim.masked_fill(pos_mask, -float('inf'))

        # Use configurable k
        # Option 1: use fixed k, but not more than batch_size-1
        k = min(self.hard_negative_k, batch_size - 1)

        # Option 2: also consider ratio limit
        k_from_ratio = int(batch_size * self.hard_negative_ratio)
        k = min(k, k_from_ratio)

        # Ensure k is at least 1 (if batch_size > 1)
        if batch_size > 1:
            k = max(1, k)
        else:
            return sim_matrix

        # Record actual k used (for debugging)
        if hasattr(self, 'training') and self.training:
            if not hasattr(self, '_k_stats'):
                self._k_stats = []
            self._k_stats.append(k)
            if len(self._k_stats) >= 100:  # Print stats every 100 batches
                avg_k = np.mean(self._k_stats)
                logger.debug(f"Average k value in last 100 batches: {avg_k:.2f}")
                self._k_stats = []

        hard_negatives, _ = torch.topk(neg_sim, k, dim=1)

        if isinstance(self.temperature, torch.Tensor):
            temp = self.temperature.clone()
        else:
            temp = self.temperature
        weights = F.softmax(hard_negatives / temp, dim=1)

        weighted_sim = sim_matrix.clone()
        weight_factor = 1 + weights.sum(dim=1, keepdim=True) * 0.1
        weighted_sim = weighted_sim * weight_factor

        return weighted_sim
    
    def nt_xent_loss(self, z1, z2, use_queue=False):
        """NT-Xent loss - fixed version (label loss removed)"""
        batch_size = z1.size(0)
        device = z1.device

        sim_matrix = self.compute_similarity(z1, z2)

        if use_queue and self.use_memory_bank:
            sim_1_queue = self.compute_similarity(z1, self.queue_2.T.clone())
            sim_2_queue = self.compute_similarity(z2, self.queue_1.T.clone())
            sim_matrix = torch.cat([sim_matrix, sim_1_queue], dim=1)

        # Remove label loss: do not use cross-entropy with labels
        # Optionally, you can return the similarity matrix or a dummy loss
        # Here, just return the mean negative similarity as a placeholder
        sim_matrix = self.hard_negative_sampling(sim_matrix, torch.arange(batch_size, device=device))
        loss = -sim_matrix.mean()
        return loss
    
    def alignment_loss(self, z1, z2, z3):
        """Feature alignment loss - fixed version"""
        if not self.use_feature_alignment:
            return torch.tensor(0.0, device=z1.device)

        z1_aligned_to_2 = self.align_1_to_2(z1)
        align_loss_12 = F.mse_loss(z1_aligned_to_2, z2.detach())

        z1_aligned_to_3 = self.align_1_to_3(z1)
        align_loss_13 = F.mse_loss(z1_aligned_to_3, z3.detach())

        z2_aligned_to_3 = self.align_2_to_3(z2)
        align_loss_23 = F.mse_loss(z2_aligned_to_3, z3.detach())

        return (align_loss_12 + align_loss_13 + align_loss_23) / 3
    
    def compute_contrastive_loss(self, z1, z2, z3, return_details=False):
        """Compute full contrastive loss - fixed version"""
        z1 = z1.clone()
        z2 = z2.clone()  
        z3 = z3.clone()

        p1 = self.proj_head_1(z1)
        p2 = self.proj_head_2(z2)
        p3 = self.proj_head_3(z3)

        if self.similarity_metric == 'cosine':
            p1_norm = F.normalize(p1, p=2, dim=1)
            p2_norm = F.normalize(p2, p=2, dim=1)
            p3_norm = F.normalize(p3, p=2, dim=1)
        else:
            p1_norm = p1
            p2_norm = p2
            p3_norm = p3

        if hasattr(self, 'momentum_proj_1'):
            with torch.no_grad():
                self._momentum_update()
                k1 = self.momentum_proj_1(z1.detach())
                k2 = self.momentum_proj_2(z2.detach())
                k3 = self.momentum_proj_3(z3.detach())
                if self.similarity_metric == 'cosine':
                    k1 = F.normalize(k1, p=2, dim=1)
                    k2 = F.normalize(k2, p=2, dim=1)
                    k3 = F.normalize(k3, p=2, dim=1)
        else:
            k1 = p1_norm.detach()
            k2 = p2_norm.detach()
            k3 = p3_norm.detach()

        loss_12 = self.nt_xent_loss(p1_norm, p2_norm, use_queue=self.use_memory_bank)
        loss_13 = self.nt_xent_loss(p1_norm, p3_norm, use_queue=self.use_memory_bank)
        loss_23 = self.nt_xent_loss(p2_norm, p3_norm, use_queue=self.use_memory_bank)

        contrastive_loss = (loss_12 + loss_13 + loss_23) / 3

        align_loss = self.alignment_loss(p1_norm, p2_norm, p3_norm)

        if self.use_memory_bank:
            with torch.no_grad():
                self._dequeue_and_enqueue(k1, k2, k3)

        total_loss = contrastive_loss + 0.1 * align_loss

        if return_details:
            return {
                'total_loss': total_loss,
                'contrastive_loss': contrastive_loss.detach(),
                'align_loss': align_loss.detach(),
                'loss_12': loss_12.detach(),
                'loss_13': loss_13.detach(),
                'loss_23': loss_23.detach()
            }

        return total_loss



# ================== Dataset class ==================

class UnifiedDataset(Dataset):
    """Unified dataset class, supports classification and regression tasks"""
    
    def __init__(self, smiles: List[str], targets: np.ndarray, 
                 task_type: str = 'classification', normalize: bool = False):
        self.smiles = smiles
        self.task_type = task_type
        
        if task_type == 'classification':
            self.targets = torch.FloatTensor(targets)
        else:  # regression
            self.targets = torch.FloatTensor(targets)
            
            # Optional normalization (regression only)
            if normalize:
                self.mean = self.targets.mean()
                self.std = self.targets.std()
                self.targets = (self.targets - self.mean) / (self.std + 1e-8)
            else:
                self.mean = 0
                self.std = 1
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return self.smiles[idx], self.targets[idx]
    
    def denormalize(self, values):
        """Denormalize (regression only)"""
        if self.task_type == 'regression':
            return values * self.std + self.mean
        return values


def collate_fn(batch):
    """Custom collate function"""
    smiles_list = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    return smiles_list, targets



# ================== Data splitting ==================

def scaffold_split_unified(smiles_list: List[str], targets: np.ndarray,
                          task_type: str = 'classification',
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1,
                          test_ratio: float = 0.1,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified scaffold split, supports classification and regression"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    np.random.seed(seed)
    random.seed(seed)
    
    if not RDKIT_AVAILABLE:
        return random_split_unified(smiles_list, targets, task_type, 
                                   train_ratio, val_ratio, test_ratio, seed)
    
    # Compute scaffolds
    scaffold_to_indices = {}
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scaffold = f"invalid_{idx}"
        else:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except:
                scaffold = f"error_{idx}"
        
        if scaffold not in scaffold_to_indices:
            scaffold_to_indices[scaffold] = []
        scaffold_to_indices[scaffold].append(idx)
    
    # Analyze each scaffold
    scaffold_info = []
    for scaffold, indices in scaffold_to_indices.items():
        scaffold_targets = targets[indices]
        
        if task_type == 'classification':
            n_pos = (scaffold_targets == 1).sum()
            n_neg = (scaffold_targets == 0).sum()
            scaffold_info.append({
                'scaffold': scaffold,
                'indices': indices,
                'n_samples': len(indices),
                'n_pos': n_pos,
                'n_neg': n_neg,
                'pos_ratio': n_pos / len(indices) if len(indices) > 0 else 0
            })
        else:  # regression
            scaffold_info.append({
                'scaffold': scaffold,
                'indices': indices,
                'n_samples': len(indices),
                'mean_value': np.mean(scaffold_targets),
                'std_value': np.std(scaffold_targets)
            })
    
    # Sort by sample count
    scaffold_info = sorted(scaffold_info, key=lambda x: x['n_samples'], reverse=True)
    
    # Initialize datasets
    train_idx = []
    val_idx = []
    test_idx = []
    
    # Target sample counts
    n_total = len(smiles_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    logger.info(f"Target split sizes - Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Assign scaffolds
    for info in scaffold_info:
        indices = info['indices']
        
        if len(train_idx) < n_train:
            train_idx.extend(indices)
        elif len(val_idx) < n_val:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)
    
    # For classification, check class balance
    if task_type == 'classification':
    # Check class distribution for each set
        for name, idx_list in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            if len(idx_list) > 0:
                unique_classes = np.unique(targets[idx_list])
                if len(unique_classes) < 2:
                    logger.warning(f"{name} set missing a class, falling back to stratified split")
                    return random_split_unified(smiles_list, targets, task_type,
                                              train_ratio, val_ratio, test_ratio, seed)
    
    # Log distribution info
    if task_type == 'classification':
        logger.info(f"Class distributions:")
        for name, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
            if len(idx) > 0:
                pos = (targets[idx] == 1).sum()
                neg = (targets[idx] == 0).sum()
                logger.info(f"  {name}: {len(idx)} samples - Class 0: {neg}, Class 1: {pos}")
    else:
        logger.info(f"Value distributions:")
        for name, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
            if len(idx) > 0:
                values = targets[idx]
                logger.info(f"  {name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
    
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def random_split_unified(smiles_list: List[str], targets: np.ndarray,
                        task_type: str = 'classification',
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        test_ratio: float = 0.1,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified random split, supports classification and regression"""
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    indices = np.arange(len(smiles_list))
    
    # First split: train vs (val+test)
    if task_type == 'classification':
    # Stratified split
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(val_ratio + test_ratio),
            stratify=targets,
            random_state=seed
        )
        
    # Second split: val vs test
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1 - relative_val_ratio),
            stratify=targets[temp_idx],
            random_state=seed
        )
    else:
    # Random split
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(val_ratio + test_ratio),
            random_state=seed
        )
        
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1 - relative_val_ratio),
            random_state=seed
        )
    
    logger.info(f"Actual split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return train_idx, val_idx, test_idx



# ================== Fusion head ==================

class MultiModalFusionHead(nn.Module):
    """Unified fusion head, supports classification and regression"""
    
    def __init__(self, input_dim: int = 512, 
                 task_type: str = 'classification',
                 fusion_type: str = 'concat_mlp', 
                 dropout_rate: float = 0.2,
                 output_activation: str = 'none'):
        super().__init__()
        self.task_type = task_type
        self.fusion_type = fusion_type
        self.output_activation = output_activation
        
    # Build network structure
        if fusion_type == 'mean':
            self.predictor = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1)
            )
        elif fusion_type == 'concat_mlp':
            self.predictor = nn.Sequential(
                nn.Linear(input_dim * 3, 768),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 1.5),
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 1)
            )
        elif fusion_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1, bias=False)
            )
            self.predictor = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 1)
            )
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        z1 = z1.clone()
        z2 = z2.clone()
        z3 = z3.clone()
        
        if self.fusion_type == 'mean':
            z_fused = (z1 + z2 + z3) / 3
            output = self.predictor(z_fused)
        elif self.fusion_type == 'concat_mlp':
            z_fused = torch.cat([z1, z2, z3], dim=1)
            output = self.predictor(z_fused)
        elif self.fusion_type == 'attention':
            features = torch.stack([z1, z2, z3], dim=1)
            attn_scores = self.attention(features)
            attn_weights = F.softmax(attn_scores, dim=1)
            z_fused = torch.sum(features * attn_weights, dim=1)
            output = self.predictor(z_fused)
        
    # Output activation for regression
        if self.task_type == 'regression' and self.output_activation != 'none':
            if self.output_activation == 'sigmoid':
                output = torch.sigmoid(output)
            elif self.output_activation == 'relu':
                output = F.relu(output)
        
        return output.squeeze(-1)



# ================== Complete model ==================

class CompleteModel(nn.Module):
    """Complete model: feature extractor + fusion head + pseudo-pair generator"""
    
    def __init__(self, feature_extractor, fusion_head, pseudo_generator):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fusion_head = fusion_head
        self.pseudo_generator = pseudo_generator
    
    def forward(self, smiles_list, return_features=False):
        """Forward pass"""
    # Extract three-way features
        z1, z2, z3 = self.feature_extractor(smiles_list)
        
    # Prediction
        predictions = self.fusion_head(z1, z2, z3)
        
        if return_features:
            return predictions, z1, z2, z3
        return predictions



# ================== Evaluation metrics ==================

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute classification metrics"""
    metrics = {}
    
    # ROC-AUC
    if y_prob is not None:
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) >= 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
            else:
                metrics['roc_auc'] = 0.5
        except:
            metrics['roc_auc'] = 0.5
    else:
        metrics['roc_auc'] = 0.5
    
    # Other metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['confusion_matrix'] = {
                'tn': int(tn), 'fp': int(fp), 
                'fn': int(fn), 'tp': int(tp)
            }
        else:
            metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    except:
        metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    
    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = float(mean_squared_error(y_true, y_pred))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    
    # R² score
    try:
        metrics['r2'] = float(r2_score(y_true, y_pred))
    except:
        metrics['r2'] = -999.0
    
    # Correlation coefficients
    try:
        pearson_corr, p_value = stats.pearsonr(y_true, y_pred)
        metrics['pearson_r'] = float(pearson_corr)
        metrics['pearson_p'] = float(p_value)
    except:
        metrics['pearson_r'] = 0.0
        metrics['pearson_p'] = 1.0
    
    try:
        spearman_corr, p_value = stats.spearmanr(y_true, y_pred)
        metrics['spearman_r'] = float(spearman_corr)
        metrics['spearman_p'] = float(p_value)
    except:
        metrics['spearman_r'] = 0.0
        metrics['spearman_p'] = 1.0
    
    # Error statistics
    errors = y_pred - y_true
    metrics['mean_error'] = float(np.mean(errors))
    metrics['std_error'] = float(np.std(errors))
    metrics['max_error'] = float(np.max(np.abs(errors)))
    
    # MAPE
    if np.all(y_true != 0):
        percentage_errors = np.abs(errors / y_true) * 100
        metrics['mape'] = float(np.mean(percentage_errors))
    else:
        metrics['mape'] = -1.0
    
    return metrics



# ================== Training and evaluation functions ==================

def train_epoch(model, dataloader, criterion, optimizer, device,
               task_type='classification',
               use_pseudo_pairs=True, pseudo_weight=0.1,
               alignment_weight=0.01, epoch=0, warmup_epochs=5):
    """Unified training function"""
    model.train()
    
    loss_components = defaultdict(list)
    all_preds = []
    all_targets = []
    all_probs = [] if task_type == 'classification' else None
    
    for batch_idx, (smiles_list, targets) in enumerate(dataloader):
        targets = targets.to(device)
        
        try:
            if use_pseudo_pairs:
                predictions, z1, z2, z3 = model(smiles_list, return_features=True)
                
                # Main task loss
                main_loss = criterion(predictions, targets)
                
                # Pseudo-pair loss
                pseudo_loss_dict = model.pseudo_generator.compute_contrastive_loss(
                    z1, z2, z3, return_details=True
                )
                
                # Dynamic weights
                if epoch < warmup_epochs:
                    current_pseudo_weight = pseudo_weight * (epoch / warmup_epochs)
                    current_align_weight = alignment_weight * (epoch / warmup_epochs)
                else:
                    current_pseudo_weight = pseudo_weight
                    current_align_weight = alignment_weight
                
                # Combined loss
                loss = (main_loss + 
                       current_pseudo_weight * pseudo_loss_dict['contrastive_loss'] +
                       current_align_weight * pseudo_loss_dict['align_loss'])
                
                # Record loss
                loss_components['main'].append(float(main_loss.item()))
                loss_components['contrastive'].append(float(pseudo_loss_dict['contrastive_loss'].item()))
                loss_components['align'].append(float(pseudo_loss_dict['align_loss'].item()))
                
            else:
                predictions = model(smiles_list)
                loss = criterion(predictions, targets)
                loss_components['main'].append(float(loss.item()))
            
            loss_components['total'].append(float(loss.item()))
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record predictions
            with torch.no_grad():
                if task_type == 'classification':
                    probs = torch.sigmoid(predictions)
                    preds = (probs > 0.5).float()
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                else:
                    all_preds.extend(predictions.cpu().numpy())
                
                all_targets.extend(targets.cpu().numpy())
                
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Compute metrics
    if task_type == 'classification':
        metrics = compute_classification_metrics(
            np.array(all_targets), 
            np.array(all_preds),
            np.array(all_probs)
        )
    else:
        metrics = compute_regression_metrics(
            np.array(all_targets),
            np.array(all_preds)
        )
    
    # Add loss info
    metrics['loss'] = float(np.mean(loss_components['total'])) if loss_components['total'] else 0.0
    metrics['loss_components'] = {
        k: float(np.mean(v)) if v else 0.0 
        for k, v in loss_components.items()
    }
    
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, task_type='classification'):
    """Unified evaluation function"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = [] if task_type == 'classification' else None
    
    for batch_idx, (smiles_list, targets) in enumerate(dataloader):
        targets = targets.to(device)
        
        try:
            predictions = model(smiles_list)
            loss = criterion(predictions, targets)
            
            if task_type == 'classification':
                probs = torch.sigmoid(predictions)
                preds = (probs > 0.5).float()
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            else:
                all_preds.extend(predictions.cpu().numpy())
            
            all_targets.extend(targets.cpu().numpy())
            total_loss += loss.item()
            
        except Exception as e:
            logger.error(f"Error in evaluation batch {batch_idx}: {e}")
            continue
    
    # Compute metrics
    if task_type == 'classification':
        metrics = compute_classification_metrics(
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs)
        )
    else:
        metrics = compute_regression_metrics(
            np.array(all_targets),
            np.array(all_preds)
        )
    
    metrics['loss'] = float(total_loss / max(1, len(dataloader)))
    
    return metrics



# ================== Early Stopping ==================

class EarlyStopping:
    """Unified Early stopping"""
    
    def __init__(self, patience: int = 10, mode: str = 'max', min_delta: float = 0.001):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience



# ================== k value recommendation function ==================

def get_recommended_k(dataset_size: int, batch_size: int) -> int:
    """Recommend k value based on dataset size and batch size"""
    
    if dataset_size < 1000:  # 小数据集
        if batch_size <= 32:
            return 4
        elif batch_size <= 64:
            return 8
        else:
            return 16
            
    elif dataset_size < 10000:  # 中等数据集
        if batch_size <= 32:
            return 8
        elif batch_size <= 64:
            return 16
        elif batch_size <= 128:
            return 32
        else:
            return 64
            
    else:  # Large dataset
        if batch_size <= 32:
            return 16
        elif batch_size <= 64:
            return 32
        elif batch_size <= 128:
            return 64
        elif batch_size <= 256:
            return 128
        else:
            return min(256, batch_size // 4)



# ================== Main function ==================

def main(args):
    """Unified main training function"""
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Task type: {args.task_type}")
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(args.data_path)
    smiles_list = df['smiles'].tolist()
    
    # Get target values
    if args.target_column in df.columns:
        targets = df[args.target_column].values.astype(np.float32)
    else:
    # Try common column names
        if 'labels' in df.columns:
            targets = df['labels'].values.astype(np.float32)
        elif 'label' in df.columns:
            targets = df['label'].values.astype(np.float32)
        elif 'y' in df.columns:
            targets = df['y'].values.astype(np.float32)
        else:
            logger.error(f"Target column '{args.target_column}' not found!")
            logger.info(f"Available columns: {list(df.columns)}")
            return
    
    logger.info(f"Total samples: {len(targets)}")
    
    # If k is not specified, recommend automatically
    if args.hard_negative_k == -1:
        args.hard_negative_k = get_recommended_k(len(targets), args.batch_size)
        logger.info(f"Auto-selected hard_negative_k = {args.hard_negative_k} based on dataset size and batch size")
    else:
        logger.info(f"Using specified hard_negative_k = {args.hard_negative_k}")
    
    # Task-specific statistics
    if args.task_type == 'classification':
        unique_classes = np.unique(targets)
        logger.info(f"Classes: {unique_classes}")
        for cls in unique_classes:
            count = (targets == cls).sum()
            logger.info(f"  Class {cls}: {count} samples ({count/len(targets)*100:.1f}%)")
    else:
        logger.info(f"Target statistics - Mean: {np.mean(targets):.3f}, Std: {np.std(targets):.3f}, "
                   f"Min: {np.min(targets):.3f}, Max: {np.max(targets):.3f}")
    
    # Data split
    logger.info(f"Performing {args.split_type} split with ratios - "
               f"Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")
    
    if args.split_type == 'scaffold':
        train_idx, val_idx, test_idx = scaffold_split_unified(
            smiles_list, targets,
            task_type=args.task_type,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    else:
        train_idx, val_idx, test_idx = random_split_unified(
            smiles_list, targets,
            task_type=args.task_type,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    
    # Create datasets
    train_dataset = UnifiedDataset(
        [smiles_list[i] for i in train_idx],
        targets[train_idx],
        task_type=args.task_type,
        normalize=(args.normalize_targets and args.task_type == 'regression')
    )
    val_dataset = UnifiedDataset(
        [smiles_list[i] for i in val_idx],
        targets[val_idx],
        task_type=args.task_type,
        normalize=(args.normalize_targets and args.task_type == 'regression')
    )
    test_dataset = UnifiedDataset(
        [smiles_list[i] for i in test_idx],
        targets[test_idx],
        task_type=args.task_type,
        normalize=(args.normalize_targets and args.task_type == 'regression')
    )
    
    # Log normalization parameters (if used)
    if args.task_type == 'regression' and args.normalize_targets:
        logger.info(f"Target normalization - Mean: {train_dataset.mean:.3f}, Std: {train_dataset.std:.3f}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    logger.info("Initializing model...")
    
    # Feature extractor
    feature_extractor = MultiModalFeatureExtractor(
        bert_path=args.bert_path,
        grover_path=args.grover_path,
        device=device
    )
    
    # Fusion head
    fusion_head = MultiModalFusionHead(
        input_dim=512,
        task_type=args.task_type,
        fusion_type=args.fusion_type,
        dropout_rate=args.dropout_rate,
        output_activation=args.output_activation if args.task_type == 'regression' else 'none'
    ).to(device)
    
    # Pseudo-pair generator - add k parameter
    pseudo_generator = AdvancedPseudoPairGenerator(
        embedding_dim=512,
        hidden_dim=256,
        projection_dim=128,
        temperature=args.temperature,
        learnable_temp=args.learnable_temp,
        use_projection_head=args.use_projection_head,
        use_feature_alignment=args.use_feature_alignment,
        similarity_metric=args.similarity_metric,
        hard_negative_mining=args.hard_negative_mining,
        hard_negative_k=args.hard_negative_k,  # 使用指定的k值
        hard_negative_ratio=args.hard_negative_ratio,  # 使用指定的比例
        momentum=args.momentum,
        use_memory_bank=args.use_memory_bank,
        queue_size=args.queue_size,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # Complete model
    model = CompleteModel(
        feature_extractor,
        fusion_head,
        pseudo_generator
    ).to(device)
    
    # Loss function
    if args.task_type == 'classification':
    # Compute positive sample weight
        train_targets = targets[train_idx]
        n_pos = (train_targets == 1).sum()
        n_neg = (train_targets == 0).sum()
        if n_pos > 0:
            pos_weight = torch.tensor([n_neg / n_pos]).to(device)
        else:
            pos_weight = torch.tensor([1.0]).to(device)
        logger.info(f"Positive class weight: {pos_weight.item():.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
    # Regression loss
        if args.loss_function == 'mse':
            criterion = nn.MSELoss()
        elif args.loss_function == 'mae':
            criterion = nn.L1Loss()
        elif args.loss_function == 'huber':
            criterion = nn.HuberLoss(delta=args.huber_delta)
        else:
            criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.scheduler == 'plateau':
        scheduler_mode = 'max' if args.task_type == 'classification' else 'min'
        scheduler = ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.5, patience=5)
    else:
        scheduler = None
    
    # Early stopping
    if args.task_type == 'classification':
        early_stop_mode = 'max'
        early_stop_metric = 'roc_auc'
    else:
        early_stop_mode = 'min' if args.early_stop_metric in ['mse', 'rmse', 'mae'] else 'max'
        early_stop_metric = args.early_stop_metric
    
    early_stopping = EarlyStopping(
        patience=args.patience, 
        mode=early_stop_mode,
        min_delta=0.0001
    )
    
    # Training history
    history = {
        'train': [],
        'val': [],
        'test': None,
        'config': vars(args),
        'data_split': {
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx)
        }
    }
    
    best_val_score = float('-inf') if early_stop_mode == 'max' else float('inf')
    best_epoch = 0
    
    # Training loop
    logger.info("Starting training...")
    logger.info(f"Training for {args.epochs} epochs")
    logger.info(f"Hard negative k = {args.hard_negative_k}, ratio = {args.hard_negative_ratio}")
    
    for epoch in range(args.epochs):
    # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            task_type=args.task_type,
            use_pseudo_pairs=args.use_pseudo_pairs,
            pseudo_weight=args.pseudo_weight,
            alignment_weight=args.alignment_weight,
            epoch=epoch,
            warmup_epochs=args.warmup_epochs
        )
        
    # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, task_type=args.task_type)
        
    # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics[early_stop_metric])
            else:
                scheduler.step()
        
    # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Loss: {train_metrics['loss']:.4f} "
            f"(Main: {train_metrics['loss_components']['main']:.4f}, "
            f"Contrast: {train_metrics['loss_components'].get('contrastive', 0):.4f})"
        )
        
        if args.task_type == 'classification':
            logger.info(
                f"  Train - AUC: {train_metrics['roc_auc']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
            )
            logger.info(
                f"  Val - AUC: {val_metrics['roc_auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
            )
        else:
            logger.info(
                f"  Train - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}"
            )
            logger.info(
                f"  Val - RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}"
            )
        
    # Save history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
    # Check for best model
        val_score = val_metrics[early_stop_metric]
        
        if early_stop_mode == 'max':
            is_better = val_score > best_val_score
        else:
            is_better = val_score < best_val_score
        
        if is_better:
            best_val_score = val_score
            best_epoch = epoch + 1
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': best_val_score,
                'args': vars(args)
            }, f'{args.output_dir}/best_model_{args.task_type}_k{args.hard_negative_k}.ckpt')
            
            logger.info(f"  -> New best model saved ({early_stop_metric}: {best_val_score:.4f})")
        
    # Early stopping
        if early_stopping(val_score, epoch):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model_path = f'best_model_{args.task_type}_k{args.hard_negative_k}.ckpt'
    if os.path.exists(model_path):
        logger.info(f"Loading best model from epoch {best_epoch}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    logger.info("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device, task_type=args.task_type)
    history['test'] = test_metrics
    
    # Print final results
    print("\n" + "="*50)
    print(f"FINAL TEST RESULTS ({args.task_type.upper()}, k={args.hard_negative_k})")
    print("="*50)
    print(f"Data Split: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    
    if args.task_type == 'classification':
        print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1-Score: {test_metrics['f1']:.4f}")
        print(f"Confusion Matrix:")
        cm = test_metrics['confusion_matrix']
        print(f"  TN: {cm['tn']}, FP: {cm['fp']}")
        print(f"  FN: {cm['fn']}, TP: {cm['tp']}")
    else:
        print(f"MSE: {test_metrics['mse']:.4f}")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"MAE: {test_metrics['mae']:.4f}")
        print(f"R²: {test_metrics['r2']:.4f}")
        print(f"Pearson R: {test_metrics['pearson_r']:.4f} (p={test_metrics['pearson_p']:.4e})")
        print(f"Spearman R: {test_metrics['spearman_r']:.4f} (p={test_metrics['spearman_p']:.4e})")
        if test_metrics['mape'] > 0:
            print(f"MAPE: {test_metrics['mape']:.2f}%")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save config
    config_path = output_dir / f'config_{args.task_type}_k{args.hard_negative_k}.json'
    with open(config_path, 'w') as f:
        json.dump(convert_to_json_serializable(vars(args)), f, indent=2)
    
    # Save test metrics
    metrics_path = output_dir / f'metrics_test_{args.task_type}_k{args.hard_negative_k}.json'
    with open(metrics_path, 'w') as f:
        json.dump(convert_to_json_serializable(test_metrics), f, indent=2)
    
    # Save full history (JSON format)
    history_path = output_dir / f'training_history_{args.task_type}_k{args.hard_negative_k}.json'
    with open(history_path, 'w') as f:
        json.dump(convert_to_json_serializable(history), f, indent=2)
    
    # Also save as pickle as backup
    pickle_path = output_dir / f'training_history_{args.task_type}_k{args.hard_negative_k}.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(history, f)
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Model saved as: {model_path}")
    
    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Molecular Property Prediction Training')

    # Task type - most important parameter
    parser.add_argument('--task-type', type=str, required=True,
                       choices=['classification', 'regression'],
                       help='Task type: classification or regression')

    # Data
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to CSV file with SMILES and target values')
    parser.add_argument('--target-column', type=str, default='labels',
                       help='Name of the target column in CSV')
    parser.add_argument('--split-type', type=str, default='scaffold',
                       choices=['scaffold', 'random'],
                       help='Type of data split')
    parser.add_argument('--normalize-targets', action='store_true', default=False,
                       help='Normalize target values (only for regression)')

    # Data split ratios
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio')

    # Model paths
    parser.add_argument('--bert-path', type=str, default='/root/autodl-tmp/AnchorMol/ChemBERTa-77M-MLM',
                       help='Path to ChemBERTa model')
    parser.add_argument('--grover-path', type=str, default='/root/autodl-tmp/AnchorMol/GROVER',
                       help='Path to GROVER model')

    # Model configuration
    parser.add_argument('--fusion-type', type=str, default='concat_mlp',
                       choices=['mean', 'concat_mlp', 'attention'],
                       help='Feature fusion strategy')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                       help='Dropout rate for fusion head')
    parser.add_argument('--output-activation', type=str, default='none',
                       choices=['none', 'sigmoid', 'relu'],
                       help='Output activation function (only for regression)')

    # Loss function (mainly for regression)
    parser.add_argument('--loss-function', type=str, default='mse',
                       choices=['mse', 'mae', 'huber'],
                       help='Loss function for regression')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                       help='Delta parameter for Huber loss')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['none', 'cosine', 'step', 'plateau'],
                       help='Learning rate scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Number of warmup epochs')

    # Pseudo-pair generation
    parser.add_argument('--use-pseudo-pairs', action='store_true', default=True,
                       help='Enable pseudo-pair generation')
    parser.add_argument('--pseudo-weight', type=float, default=0.1,
                       help='Weight for pseudo-pair contrastive loss')
    parser.add_argument('--alignment-weight', type=float, default=0.01,
                       help='Weight for feature alignment loss')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss')
    parser.add_argument('--learnable-temp', action='store_true', default=True,
                       help='Make temperature learnable')
    parser.add_argument('--use-projection-head', action='store_true', default=True,
                       help='Use projection head')
    parser.add_argument('--use-feature-alignment', action='store_true', default=True,
                       help='Use feature alignment')
    parser.add_argument('--similarity-metric', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'dot'],
                       help='Similarity metric')
    parser.add_argument('--hard-negative-mining', action='store_true', default=True,
                       help='Enable hard negative mining')

    # New: hard negative k parameter
    parser.add_argument('--hard-negative-k', type=int, default=-1,
                       choices=[-1, 1, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024],
                       help='Number of hard negatives to select (-1 for auto)')
    parser.add_argument('--hard-negative-ratio', type=float, default=0.25,
                       help='Maximum ratio of batch size for hard negative selection')

    parser.add_argument('--momentum', type=float, default=0.999,
                       help='Momentum for momentum encoder')
    parser.add_argument('--use-memory-bank', action='store_true', default=False,
                       help='Use memory bank')
    parser.add_argument('--queue-size', type=int, default=4096,
                       help='Size of memory bank')

    # System
    parser.add_argument('--output-dir', type=str, default='./train-claude-class-reg',
                       help='Output directory')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--early-stop-metric', type=str, default='rmse',
                       choices=['mse', 'rmse', 'mae', 'r2'],
                       help='Metric for early stopping (only for regression)')

    args = parser.parse_args()

    # Validate data split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error(f"Train, val and test ratios must sum to 1.0, got {total_ratio}")

    # Run training
    results = main(args)
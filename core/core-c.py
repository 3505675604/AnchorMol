"""
Core module for dynamic modal center library, streaming clustering, pseudo pair generation, and related utilities.
All comments and docstrings are in English. Unnecessary comments have been removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import pickle
import json
import hashlib
from collections import deque, defaultdict
from datetime import datetime
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
import warnings

# Try to import faiss, use fallback if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available, using PyTorch backend for similarity search")

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ================== Configuration Classes ==================

@dataclass
class CenterLibraryConfig:
    """Configuration for the modal center library."""
    embedding_dim: int = 768
    num_centers: int = 10000
    
    # Incremental update config
    incremental_batch_size: int = 1000
    update_momentum: float = 0.99
    min_samples_for_update: int = 100
    
    # Version management
    max_versions: int = 10
    version_retention_days: int = 30
    checkpoint_interval: int = 1000
    
    # Online learning
    online_learning_enabled: bool = True
    stream_buffer_size: int = 10000
    async_update: bool = True
    
    # Performance optimization
    use_faiss: bool = True and FAISS_AVAILABLE
    use_fp16: bool = True
    cache_size: int = 100000
    
    # Clustering config
    clustering_method: str = "streaming_kmeans"
    recompute_interval: int = 5000


@dataclass
class PairGeneratorConfig:
    """Configuration for pseudo pair generator."""
    top_k: int = 10
    temperature: float = 0.07
    
    # Negative sample strategy
    negative_sampling_ratio: int = 5
    hard_negative_mining: bool = True
    hard_negative_ratio: float = 0.3
    
    # Quality control
    min_similarity_threshold: float = 0.3
    max_similarity_threshold: float = 0.95
    confidence_threshold: float = 0.7
    
    # Online learning
    adaptive_temperature: bool = True
    temperature_decay: float = 0.995



# ================== Version Management System ==================

class VersionManager:
    """Version manager - supports version control and rollback for the center library."""
    
    def __init__(self, base_path: str = "./center_versions", max_versions: int = 10):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_versions = max_versions
        self.version_metadata = self._load_metadata()
        self.lock = threading.Lock()
        
    def _load_metadata(self) -> Dict:
        """Load version metadata."""
        metadata_file = self.base_path / "version_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {
            "versions": [],
            "current_version": None,
            "last_update": None
        }
    
    def save_version(self, data: Dict[str, Any], version_tag: Optional[str] = None) -> str:
        """Save a new version."""
        with self.lock:
            # Generate version ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_id = f"v_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
            
            if version_tag:
                version_id = f"{version_id}_{version_tag}"
            
            # Save data
            version_path = self.base_path / f"{version_id}.pkl"
            with open(version_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            version_info = {
                "id": version_id,
                "timestamp": timestamp,
                "tag": version_tag,
                "size": version_path.stat().st_size,
                "checksum": self._compute_checksum(data)
            }
            
            self.version_metadata["versions"].append(version_info)
            self.version_metadata["current_version"] = version_id
            self.version_metadata["last_update"] = timestamp
            
            # Clean up old versions
            self._cleanup_old_versions()
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Saved version: {version_id}")
            return version_id
    
    def load_version(self, version_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load a specific version."""
        with self.lock:
            if version_id is None:
                version_id = self.version_metadata.get("current_version")
            
            if not version_id:
                return None
            
            version_path = self.base_path / f"{version_id}.pkl"
            if not version_path.exists():
                logger.error(f"Version not found: {version_id}")
                return None
            
            with open(version_path, 'rb') as f:
                data = pickle.load(f)
            
            # Verify checksum
            if not self._verify_checksum(data, version_id):
                logger.warning(f"Checksum verification failed for version: {version_id}")
            
            return data
    
    def _cleanup_old_versions(self):
        """Clean up old versions exceeding the limit."""
        versions = self.version_metadata["versions"]
        if len(versions) > self.max_versions:
            # Keep only the latest max_versions versions
            to_remove = versions[:-self.max_versions]
            for version_info in to_remove:
                version_path = self.base_path / f"{version_info['id']}.pkl"
                if version_path.exists():
                    version_path.unlink()
            
            self.version_metadata["versions"] = versions[-self.max_versions:]
    
    def _compute_checksum(self, data: Dict) -> str:
        """Compute data checksum."""
        data_str = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(data_str).hexdigest()
    
    def _verify_checksum(self, data: Dict, version_id: str) -> bool:
        """Verify data integrity."""
        for version_info in self.version_metadata["versions"]:
            if version_info["id"] == version_id:
                expected_checksum = version_info.get("checksum")
                if expected_checksum:
                    actual_checksum = self._compute_checksum(data)
                    return actual_checksum == expected_checksum
        return True
    
    def _save_metadata(self):
        """Save metadata."""
        metadata_file = self.base_path / "version_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.version_metadata, f, indent=2)



# ================== Streaming Clustering Algorithm ==================

class StreamingKMeans:
    """Streaming K-means algorithm - supports online incremental updates."""
    
    def __init__(self, n_clusters: int, decay_factor: float = 0.99):
        self.n_clusters = n_clusters
        self.decay_factor = decay_factor
        self.centers = None
        self.counts = None
        self.initialized = False
        self.total_samples = 0
        
    def partial_fit(self, X: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
        """Incrementally update cluster centers."""
        if not self.initialized:
            self._initialize(X)
            return self
        
    # Assign samples to nearest centers
        assignments = self._assign_clusters(X)
        
    # Update centers
        self._update_centers(X, assignments, sample_weight)
        
        self.total_samples += len(X)
        return self
    
    def _initialize(self, X: torch.Tensor):
        """Initialize cluster centers."""
        n_samples = min(self.n_clusters, len(X))
        
    # Use K-means++ initialization
        self.centers = self._kmeans_plusplus(X, n_samples)
        self.counts = torch.zeros(self.n_clusters, device=X.device)
        self.initialized = True
    
    def _kmeans_plusplus(self, X: torch.Tensor, n_centers: int) -> torch.Tensor:
        """K-means++ initialization."""
        centers = []
        
    # Randomly select the first center
        idx = torch.randint(len(X), (1,)).item()
        centers.append(X[idx])
        
    # Select remaining centers
        for _ in range(1, n_centers):
            # Compute distance to nearest center for each point
            distances = torch.cdist(X, torch.stack(centers))
            min_distances, _ = distances.min(dim=1)
            
            # Select next center with probability proportional to squared distance
            probabilities = min_distances ** 2
            probabilities /= probabilities.sum()
            
            idx = torch.multinomial(probabilities, 1).item()
            centers.append(X[idx])
        
        centers_tensor = torch.stack(centers)
        
        # If more centers needed, fill with noise
        if n_centers < self.n_clusters:
            extra = self.n_clusters - n_centers
            noise = torch.randn(extra, X.size(1), device=X.device) * 0.01
            centers_tensor = torch.cat([centers_tensor, X.mean(0).unsqueeze(0) + noise])
        
        return centers_tensor
    
    def _assign_clusters(self, X: torch.Tensor) -> torch.Tensor:
        """Assign samples to nearest cluster center."""
        distances = torch.cdist(X, self.centers)
        return distances.argmin(dim=1)
    
    def _update_centers(self, X: torch.Tensor, assignments: torch.Tensor, 
                       sample_weight: Optional[torch.Tensor] = None):
        """Update centers using exponential moving average."""
        if sample_weight is None:
            sample_weight = torch.ones(len(X), device=X.device)
        
        for i in range(self.n_clusters):
            mask = assignments == i
            if mask.any():
                # Compute weighted average
                weights = sample_weight[mask]
                weighted_sum = (X[mask] * weights.unsqueeze(1)).sum(0)
                weight_sum = weights.sum()
                
                if weight_sum > 0:
                    new_center = weighted_sum / weight_sum
                    
                    # Exponential moving average update
                    if self.counts[i] > 0:
                        alpha = 1.0 / (1.0 + self.counts[i])
                        self.centers[i] = (1 - alpha) * self.centers[i] + alpha * new_center
                    else:
                        self.centers[i] = new_center
                    
                    # Update count (with decay)
                    self.counts[i] = self.counts[i] * self.decay_factor + weight_sum
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict cluster labels for samples."""
        if not self.initialized:
            raise ValueError("Model must be fitted before prediction")
        return self._assign_clusters(X)



# ================== Center Wrapper Module ==================

class CenterModule(nn.Module):
    """Wrap nn.Parameter as nn.Module for ModuleDict compatibility."""
    
    def __init__(self, num_centers: int, embedding_dim: int):
        super().__init__()
        self.centers = nn.Parameter(
            torch.randn(num_centers, embedding_dim)
        )
    # Initialization
        nn.init.xavier_uniform_(self.centers)
        self.centers.data = F.normalize(self.centers.data, p=2, dim=1)
    
    def forward(self):
        return self.centers



# ================== Dynamic Modal Center Library ==================

    """
    Production-level dynamic modal center library.
    Supports incremental updates, version management, and online learning.
    """
    
    def __init__(self, config: CenterLibraryConfig):
        super().__init__()
        self.config = config
        
    # Initialize components
        self._initialize_components()
        
    # Version management
        self.version_manager = VersionManager(
            base_path="./center_versions",
            max_versions=config.max_versions
        )
        
    # Statistics
        self.stats = {
            "total_samples": 0,
            "updates_count": 0,
            "last_update": None,
            "version_history": []
        }
        
    logger.info(f"Initialized DynamicModalCenterLibrary with config: {config}")
    
    def _initialize_components(self):
        """Initialize all components."""
    # Modal center storage - use ModuleDict to store modules
        self.modal_centers = nn.ModuleDict()
        
    # Clusterers
        self.clusterers = {}
        
    # Streaming buffers
        self.stream_buffers = defaultdict(lambda: deque(maxlen=self.config.stream_buffer_size))
        
    # FAISS indices (if enabled)
        self.indices = {}
        
    # Thread pool (for async operations)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    # Locking mechanism
        self.update_locks = defaultdict(threading.Lock)
        
    # Initialize modalities
        self.modalities = ['image', 'text', 'audio']  # 可配置
        for modality in self.modalities:
            self._initialize_modality(modality)
    
    def _initialize_modality(self, modality: str):
        """Initialize a single modality."""
    # Use CenterModule to wrap centers
        self.modal_centers[modality] = CenterModule(
            self.config.num_centers,
            self.config.embedding_dim
        )
        
    # Initialize clusterer
        if self.config.clustering_method == "streaming_kmeans":
            self.clusterers[modality] = StreamingKMeans(
                n_clusters=self.config.num_centers,
                decay_factor=self.config.update_momentum
            )
        
    # Initialize FAISS index
        if self.config.use_faiss:
            self._initialize_faiss_index(modality)
    
    def _initialize_faiss_index(self, modality: str):
        """Initialize FAISS index for fast retrieval."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping index initialization")
            return
            
        dim = self.config.embedding_dim
        
    # Use IVF index to support incremental updates
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(self.config.num_centers, 100))
        
    # Train index
        if modality in self.modal_centers:
            centers_np = self.modal_centers[modality].centers.detach().cpu().numpy()
            index.train(centers_np)
            index.add(centers_np)
        
        self.indices[modality] = index
    
    def get_centers(self, modality: str) -> torch.Tensor:
        """Get centers for the specified modality."""
        if modality in self.modal_centers:
            return self.modal_centers[modality].centers
        return None
    
    @torch.no_grad()
    def add_samples(self, samples: Dict[str, torch.Tensor], 
                   update_immediately: bool = False) -> Dict[str, Any]:
        """
        Add new samples to the center library.
        Supports batch and streaming processing.
        """
        results = {}
        
        for modality, embeddings in samples.items():
            if modality not in self.modalities:
                logger.warning(f"Unknown modality: {modality}")
                continue
            
            # Convert to appropriate precision
            if self.config.use_fp16:
                embeddings = embeddings.half()
            
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Add to streaming buffer
            self.stream_buffers[modality].extend(embeddings.unbind(0))
            
            # Check if update is needed
            buffer_size = len(self.stream_buffers[modality])
            should_update = (
                update_immediately or 
                buffer_size >= self.config.incremental_batch_size
            )
            
            if should_update:
                if self.config.async_update:
                    # Async update
                    future = self.executor.submit(
                        self._update_modality_centers,
                        modality
                    )
                    results[f"{modality}_update_future"] = future
                else:
                    # Sync update
                    update_result = self._update_modality_centers(modality)
                    results[f"{modality}_update"] = update_result
        
    # Update statistics
        self.stats["total_samples"] += sum(len(s) for s in samples.values())
        
    # Check if checkpoint needs to be saved
        if self.stats["total_samples"] % self.config.checkpoint_interval == 0:
            self._save_checkpoint()
        
        return results
    
    def _update_modality_centers(self, modality: str) -> Dict[str, Any]:
        """Update centers for a specific modality."""
        with self.update_locks[modality]:
            buffer = self.stream_buffers[modality]
            if len(buffer) < self.config.min_samples_for_update:
                return {"status": "insufficient_samples", "buffer_size": len(buffer)}
            
            # Convert to tensor
            embeddings = torch.stack(list(buffer))
            
            # Clear buffer
            self.stream_buffers[modality].clear()
            
            # Update clustering
            clusterer = self.clusterers[modality]
            clusterer.partial_fit(embeddings)
            
            # Get new centers
            new_centers = clusterer.centers
            
            # Update existing centers with momentum
            momentum = self.config.update_momentum
            old_centers = self.modal_centers[modality].centers.data
            updated_centers = momentum * old_centers + (1 - momentum) * new_centers
            
            # Normalize
            updated_centers = F.normalize(updated_centers, p=2, dim=1)
            
            # Update parameters
            self.modal_centers[modality].centers.data = updated_centers
            
            # Update FAISS index
            if self.config.use_faiss and modality in self.indices:
                self._update_faiss_index(modality, updated_centers)
            
            # Update statistics
            self.stats["updates_count"] += 1
            self.stats["last_update"] = datetime.now().isoformat()
            
            return {
                "status": "success",
                "samples_processed": len(embeddings),
                "centers_updated": len(updated_centers)
            }
    
    def _update_faiss_index(self, modality: str, centers: torch.Tensor):
        """Update FAISS index."""
        if not FAISS_AVAILABLE:
            return
            
        index = self.indices[modality]
        
    # Reset index and add new centers
        index.reset()
        centers_np = centers.detach().cpu().numpy()
        
    # Ensure index is trained
        if not index.is_trained:
            index.train(centers_np)
        
        index.add(centers_np)
    
    def query_centers(self, query_embeddings: torch.Tensor, 
                     modality: str, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the most similar centers.
        Returns distances and indices.
        """
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality: {modality}")
        
    # Normalize query
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        
        if self.config.use_faiss and modality in self.indices:
            # Use FAISS for fast retrieval
            query_np = query_embeddings.detach().cpu().numpy()
            distances, indices = self.indices[modality].search(query_np, k)
            
            return torch.tensor(distances), torch.tensor(indices)
        else:
            # Use PyTorch for retrieval
            centers = self.modal_centers[modality].centers
            similarities = torch.mm(query_embeddings, centers.t())
            
            # Get top-k
            topk_sims, topk_indices = torch.topk(similarities, k, dim=1)
            
            # Convert to distance
            distances = 2 - 2 * topk_sims  # 余弦距离
            
            return distances, topk_indices
    
    def _save_checkpoint(self):
        """Save checkpoint."""
        checkpoint_data = {
            "modal_centers": {k: v.centers.data.cpu() for k, v in self.modal_centers.items()},
            "stats": self.stats,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
    # Save version
        version_id = self.version_manager.save_version(
            checkpoint_data,
            version_tag="checkpoint"
        )
        
        self.stats["version_history"].append(version_id)
        logger.info(f"Saved checkpoint: {version_id}")
    
    def load_checkpoint(self, version_id: Optional[str] = None):
        """Load checkpoint."""
        checkpoint_data = self.version_manager.load_version(version_id)
        
        if checkpoint_data is None:
            logger.error("Failed to load checkpoint")
            return False
        
    # Restore centers
        for modality, centers in checkpoint_data["modal_centers"].items():
            if modality in self.modal_centers:
                self.modal_centers[modality].centers.data = centers.to(
                    self.modal_centers[modality].centers.device
                )
        
    # Restore statistics
        self.stats.update(checkpoint_data["stats"])
        
    # Rebuild index
        if self.config.use_faiss:
            for modality in self.modalities:
                self._update_faiss_index(modality, self.modal_centers[modality].centers)
        
        logger.info(f"Loaded checkpoint from version: {version_id}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        stats = self.stats.copy()
        
    # Add statistics for each modality
        for modality in self.modalities:
            centers = self.modal_centers[modality].centers
            stats[f"{modality}_stats"] = {
                "num_centers": len(centers),
                "center_norm_mean": centers.norm(dim=1).mean().item(),
                "center_norm_std": centers.norm(dim=1).std().item(),
                "buffer_size": len(self.stream_buffers[modality])
            }
        
        return stats



# ================== Pseudo Pair Generator ==================

    """
    Production-level pseudo pair generator.
    Supports online learning and adaptive strategies.
    """
    
    def __init__(self, config: PairGeneratorConfig, 
                 center_library: DynamicModalCenterLibrary):
        super().__init__()
        self.config = config
        self.center_library = center_library
        
    # Temperature parameter (learnable)
        self.temperature = nn.Parameter(torch.tensor(config.temperature))
        
    # Hard negative miner
        self.negative_miner = HardNegativeMiner(
            ratio=config.hard_negative_ratio
        )
        
    # Quality scorer
        self.quality_scorer = QualityScorer(
            min_threshold=config.min_similarity_threshold,
            max_threshold=config.max_similarity_threshold
        )
        
    # Statistics
        self.generation_stats = defaultdict(lambda: {"count": 0, "quality": []})
        
    def generate_pairs(self, source_embeddings: Dict[str, torch.Tensor],
                       target_modality: str,
                       return_stats: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate high-quality pseudo pairs.
        """
        results = {}
        stats = {}
        # Generate pseudo pairs for each source modality
        for source_modality, embeddings in source_embeddings.items():
            if source_modality == target_modality:
                continue
            
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Query similar centers
            distances, indices = self.center_library.query_centers(
                embeddings, 
                target_modality, 
                k=self.config.top_k
            )
            
            # Get target centers
            target_centers = self.center_library.get_centers(target_modality)
            retrieved_centers = target_centers[indices]
            
            # Generate pseudo pairs
            pseudo_pairs = self._create_weighted_pairs(
                embeddings, 
                retrieved_centers, 
                distances
            )
            
            # Quality control
            if self.config.hard_negative_mining:
                # Hard negative mining
                hard_negatives = self.negative_miner.mine(
                    embeddings,
                    target_centers,
                    num_negatives=self.config.negative_sampling_ratio * len(embeddings)
                )
                pseudo_pairs["hard_negatives"] = hard_negatives
            
            # Evaluate quality
            quality_scores = self.quality_scorer.score(
                pseudo_pairs["positive"],
                pseudo_pairs.get("hard_negatives")
            )
            
            # Filter low-quality pairs
            mask = quality_scores > self.config.confidence_threshold
            filtered_pairs = {
                k: v[mask] if len(v.shape) > 1 and len(mask) == len(v) else v
                for k, v in pseudo_pairs.items()
            }
            
            # Save results
            pair_key = f"{source_modality}_to_{target_modality}"
            results[pair_key] = filtered_pairs
            
            # Update statistics
            self.generation_stats[pair_key]["count"] += len(filtered_pairs["positive"])
            self.generation_stats[pair_key]["quality"].append(quality_scores.mean().item())
            
            if return_stats:
                stats[pair_key] = {
                    "num_pairs": len(filtered_pairs["positive"]),
                    "avg_quality": quality_scores.mean().item(),
                    "filtered_ratio": mask.float().mean().item()
                }
        
        # Adaptive temperature adjustment
        if self.config.adaptive_temperature and self.training:
            self._adjust_temperature(stats)
        if return_stats:
            results["stats"] = stats
        return results
    
    def _create_weighted_pairs(self, source_embeddings: torch.Tensor,
                              retrieved_centers: torch.Tensor,
                              distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Create weighted pseudo pairs.
        """
        batch_size = source_embeddings.size(0)
        k = retrieved_centers.size(1)
        # Compute weights (based on distance)
        weights = F.softmax(-distances / self.temperature, dim=1)
        # Weighted aggregation
        weighted_targets = torch.sum(
            retrieved_centers * weights.unsqueeze(-1),
            dim=1
        )
        # Normalize
        weighted_targets = F.normalize(weighted_targets, p=2, dim=1)
        # Create positive sample pairs
        positive_pairs = torch.stack([source_embeddings, weighted_targets], dim=1)
        # Create negative samples (in-batch negatives)
        negative_pairs = []
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    negative_pairs.append(
                        torch.stack([source_embeddings[i], weighted_targets[j]])
                    )
        if negative_pairs:
            negative_pairs = torch.stack(negative_pairs)
        else:
            negative_pairs = torch.empty(0, 2, source_embeddings.size(1))
        return {
            "positive": positive_pairs,
            "negative": negative_pairs,
            "weights": weights
        }
    
    def _adjust_temperature(self, stats: Dict):
        """Adaptively adjust temperature parameter."""
        if not stats:
            return
        # Adjust temperature based on quality score
        avg_quality = np.mean([s["avg_quality"] for s in stats.values()])
        if avg_quality < 0.5:
            # If quality too low, decrease temperature for more certain matches
            self.temperature.data *= 0.99
        elif avg_quality > 0.8:
            # If quality too high, increase temperature for more diversity
            self.temperature.data *= 1.01
        # Clamp temperature range
        self.temperature.data = torch.clamp(self.temperature.data, 0.01, 1.0)
    
    def compute_contrastive_loss(self, pairs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss.
        """
        total_loss = 0
        num_pairs = 0
        for pair_key, pair_data in pairs.items():
            if "stats" in pair_key:
                continue
            positive = pair_data.get("positive")
            negative = pair_data.get("negative")
            if positive is None or len(positive) == 0:
                continue
            # Compute positive sample similarity
            pos_sim = F.cosine_similarity(
                positive[:, 0], 
                positive[:, 1], 
                dim=1
            ) / self.temperature
            if negative is not None and len(negative) > 0:
                # Compute negative sample similarity
                neg_sim = F.cosine_similarity(
                    negative[:, 0],
                    negative[:, 1],
                    dim=1
                ) / self.temperature
                # InfoNCE loss
                loss = -torch.log(
                    torch.exp(pos_sim) / 
                    (torch.exp(pos_sim).sum() + torch.exp(neg_sim).sum())
                ).mean()
            else:
                # Loss with only positive samples
                loss = -pos_sim.mean()
            total_loss += loss
            num_pairs += 1
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, requires_grad=True)



# ================== Helper Components ==================

class HardNegativeMiner:
    """Hard negative miner."""
    
    def __init__(self, ratio: float = 0.3):
        self.ratio = ratio
    
    def mine(self, anchors: torch.Tensor, candidates: torch.Tensor,
             num_negatives: int) -> torch.Tensor:
        """
        Mine hard negative samples.
        """
        # Compute similarity
        similarities = torch.mm(anchors, candidates.t())
        # Select samples that are similar but not identical as hard negatives
        num_hard = int(num_negatives * self.ratio)
        num_random = num_negatives - num_hard
        hard_negatives = []
        # For each anchor, select hard negatives
        for i in range(len(anchors)):
            sim_scores = similarities[i]
            
            # Exclude self (if present in candidates)
            if i < len(sim_scores):
                sim_scores[i] = -float('inf')
            
            # Select highest similarity as hard negatives
            _, hard_indices = torch.topk(sim_scores, min(num_hard, len(candidates)-1))
            
            # Add random negatives
            remaining_indices = list(set(range(len(candidates))) - set(hard_indices.tolist()) - {i})
            if remaining_indices and num_random > 0:
                random_indices = torch.tensor(
                    np.random.choice(remaining_indices, 
                                   min(num_random, len(remaining_indices)),
                                   replace=False)
                )
                all_indices = torch.cat([hard_indices, random_indices])
            else:
                all_indices = hard_indices
            
            if len(all_indices) > 0:
                hard_negatives.append(candidates[all_indices])
        
        return torch.stack(hard_negatives) if hard_negatives else torch.empty(0, candidates.size(1))


class QualityScorer:
    """Quality scorer."""
    
    def __init__(self, min_threshold: float = 0.3, max_threshold: float = 0.95):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    def score(self, positive_pairs: torch.Tensor, 
              negative_pairs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute pseudo pair quality score.
        """
        # Positive sample similarity
        pos_sim = F.cosine_similarity(positive_pairs[:, 0], positive_pairs[:, 1], dim=1)
        # Base score
        scores = torch.zeros_like(pos_sim)
        # Positive sample score (within threshold)
        valid_mask = (pos_sim > self.min_threshold) & (pos_sim < self.max_threshold)
        scores[valid_mask] = (pos_sim[valid_mask] - self.min_threshold) / (self.max_threshold - self.min_threshold)
        # If negative samples exist, compute separation
        if negative_pairs is not None and len(negative_pairs) > 0:
            neg_sim = F.cosine_similarity(negative_pairs[:, 0], negative_pairs[:, 1], dim=1)
            # Greater separation between positive and negative, higher quality
            separation = pos_sim.mean() - neg_sim.mean()
            separation_bonus = torch.sigmoid(separation * 5)  # Map to [0,1]
            scores = scores * 0.7 + separation_bonus * 0.3
        return scores



# ================== Usage Example ==================

def main():
    """Main function example."""
    # Configuration
    center_config = CenterLibraryConfig(
        embedding_dim=128,  # Lower dimension for faster test
        num_centers=100,    # Fewer centers
        online_learning_enabled=True,
        use_faiss=False,    # Disable FAISS to avoid dependency
        incremental_batch_size=50,
        min_samples_for_update=10  # Lower update threshold
    )
    pair_config = PairGeneratorConfig(
        top_k=5,
        temperature=0.07,
        hard_negative_mining=True
    )
    # Initialize system
    print("Initializing system...")
    center_library = DynamicModalCenterLibrary(center_config)
    pair_generator = PseudoPairGenerator(pair_config, center_library)
    # Simulate online learning scenario
    print("\nStarting online learning simulation...")
    for epoch in range(5):  # Fewer epochs
        # Generate simulated data
        batch_size = 32  # Smaller batch size
        image_features = torch.randn(batch_size, 128)
        text_features = torch.randn(batch_size, 128)
        audio_features = torch.randn(batch_size, 128)
        samples = {
            "image": image_features,
            "text": text_features,
            "audio": audio_features
        }
        # Incrementally update center library
        update_results = center_library.add_samples(samples)
        # Generate pseudo pairs
        source_embeddings = {"image": image_features, "audio": audio_features}
        pairs = pair_generator.generate_pairs(
            source_embeddings,
            target_modality="text",
            return_stats=True
        )
        # Compute loss
        loss = pair_generator.compute_contrastive_loss(pairs)
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")
        # Print statistics
        if epoch % 2 == 0:
            stats = center_library.get_statistics()
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Updates count: {stats['updates_count']}")
    # Save final version
    print("\nSaving checkpoint...")
    center_library._save_checkpoint()
    print("Training completed and checkpoint saved")
    # Show final statistics
    final_stats = center_library.get_statistics()
    print("\nFinal Statistics:")
    for key, value in final_stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
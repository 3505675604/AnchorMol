import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import pickle
import json
import hashlib
from collections import deque, defaultdict
from datetime import datetime
import logging
from pathlib import Path
import warnings


# Try to import faiss for fast similarity search; fallback to PyTorch if unavailable
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available, using PyTorch backend for similarity search")

logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """L2 normalize a tensor along a given dimension."""
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps))



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
    recompute_interval: int = 5000  # Placeholder, not used by streaming_kmeans


@dataclass
class PairGeneratorConfig:
    """Configuration for pseudo-pair generator."""
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



class VersionManager:
    """Version manager for center library with versioning and rollback support."""

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
            # 生成版本ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_id = f"v_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

            if version_tag:
                version_id = f"{version_id}_{version_tag}"

            # 保存数据
            version_path = self.base_path / f"{version_id}.pkl"
            with open(version_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 更新元数据
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

            # 清理旧版本
            self._cleanup_old_versions()

            # 保存元数据
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

            # 验证校验和
            if not self._verify_checksum(data, version_id):
                logger.warning(f"Checksum verification failed for version: {version_id}")

            return data

    def _cleanup_old_versions(self):
        """Clean up old versions exceeding the limit."""
        versions = self.version_metadata["versions"]
        if len(versions) > self.max_versions:
            # 保留最新的max_versions个版本
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



class StreamingKMeans:
    """Streaming K-means algorithm with online incremental update support."""

    def __init__(self, n_clusters: int, decay_factor: float = 0.99):
        self.n_clusters = n_clusters
        self.decay_factor = decay_factor
        self.centers: Optional[torch.Tensor] = None
        self.counts: Optional[torch.Tensor] = None
        self.initialized = False
        self.total_samples = 0

    def partial_fit(self, X: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
        """Incrementally update cluster centers."""
        # 统一 float32，提高兼容性（尤其 CPU & cdist）
        X = X.float()
        if not self.initialized:
            self._initialize(X)
            return self

        assignments = self._assign_clusters(X)
        self._update_centers(X, assignments, sample_weight)
        self.total_samples += len(X)
        return self

    def _initialize(self, X: torch.Tensor):
        """Initialize cluster centers."""
        n_samples = min(self.n_clusters, len(X))
        # K-means++ 初始化
        self.centers = self._kmeans_plusplus(X, n_samples)
        self.counts = torch.zeros(self.n_clusters, device=self.centers.device, dtype=torch.float32)
        self.initialized = True

    def _kmeans_plusplus(self, X: torch.Tensor, n_centers: int) -> torch.Tensor:
        centers = []
        idx = torch.randint(len(X), (1,), device=X.device).item()
        centers.append(X[idx])

        for _ in range(1, n_centers):
            distances = torch.cdist(X, torch.stack(centers))  # [N, curr]
            min_distances, _ = distances.min(dim=1)
            prob = (min_distances ** 2)
            prob = prob / (prob.sum() + 1e-8)
            idx = torch.multinomial(prob, 1).item()
            centers.append(X[idx])

        centers_tensor = torch.stack(centers)
        if n_centers < self.n_clusters:
            extra = self.n_clusters - n_centers
            noise = torch.randn(extra, X.size(1), device=X.device) * 0.01
            centers_tensor = torch.cat([centers_tensor, X.mean(0, keepdim=True) + noise], dim=0)
        return centers_tensor

    def _assign_clusters(self, X: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(X, self.centers)  # [N, K]
        return distances.argmin(dim=1)

    def _update_centers(self, X: torch.Tensor, assignments: torch.Tensor,
                        sample_weight: Optional[torch.Tensor] = None):
        if sample_weight is None:
            sample_weight = torch.ones(len(X), device=X.device, dtype=torch.float32)

        for i in range(self.n_clusters):
            mask = (assignments == i)
            if mask.any():
                weights = sample_weight[mask]
                weighted_sum = (X[mask] * weights.unsqueeze(1)).sum(0)
                weight_sum = weights.sum()
                if weight_sum > 0:
                    new_center = weighted_sum / weight_sum
                    if self.counts[i] > 0:
                        alpha = 1.0 / (1.0 + self.counts[i])
                        self.centers[i] = (1 - alpha) * self.centers[i] + alpha * new_center
                    else:
                        self.centers[i] = new_center
                    # 计数带衰减
                    self.counts[i] = self.counts[i] * self.decay_factor + weight_sum

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            raise ValueError("Model must be fitted before prediction")
        X = X.float()
        return self._assign_clusters(X)


# ================== 中心包装器模块 ==================

class CenterModule(nn.Module):
    """Wrap nn.Parameter as nn.Module for ModuleDict compatibility."""

    def __init__(self, num_centers: int, embedding_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, embedding_dim))
        nn.init.xavier_uniform_(self.centers)
        with torch.no_grad():
            self.centers.data = F.normalize(self.centers.data, p=2, dim=1)

    def forward(self) -> torch.Tensor:
        return self.centers


# ================== 动态模态中心库 ==================

class DynamicModalCenterLibrary(nn.Module):
    """
    Production-level dynamic modal center library.
    Supports incremental update, version management, and online learning.
    """

    def __init__(self, config: CenterLibraryConfig):
        super().__init__()
        self.config = config

        # 初始化组件
        self._initialize_components()

        # 版本管理
        self.version_manager = VersionManager(
            base_path="./center_versions",
            max_versions=config.max_versions
        )

        # 统计信息
        self.stats = {
            "total_samples": 0,
            "updates_count": 0,
            "last_update": None,
            "version_history": []
        }

        logger.info(f"Initialized DynamicModalCenterLibrary with config: {config}")

    def _initialize_components(self):
        """Initialize all components."""
        self.modal_centers = nn.ModuleDict()
        self.clusterers: Dict[str, StreamingKMeans] = {}
        self.stream_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.stream_buffer_size))
        self.indices: Dict[str, Any] = {}  # FAISS
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.update_locks = defaultdict(threading.Lock)

        # 模态可配置：这里默认三模态
        self.modalities = ['image', 'text', 'audio']
        for modality in self.modalities:
            self._initialize_modality(modality)

    def _initialize_modality(self, modality: str):
        """Initialize a single modality."""
        self.modal_centers[modality] = CenterModule(
            self.config.num_centers,
            self.config.embedding_dim
        )
        if self.config.clustering_method == "streaming_kmeans":
            self.clusterers[modality] = StreamingKMeans(
                n_clusters=self.config.num_centers,
                decay_factor=self.config.update_momentum
            )
        if self.config.use_faiss:
            self._initialize_faiss_index(modality)

    def _initialize_faiss_index(self, modality: str):
        """Initialize FAISS index."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping index initialization")
            return
        dim = self.config.embedding_dim
        nlist = max(16, min(self.config.num_centers // 10, 4096))  # 经验值
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)

        centers_np = self.modal_centers[modality].centers.detach().float().cpu().numpy()
        index.train(centers_np)
        index.add(centers_np)
        self.indices[modality] = index

    def get_centers(self, modality: str) -> torch.Tensor:
        if modality in self.modal_centers:
            return self.modal_centers[modality].centers
        raise ValueError(f"Unknown modality: {modality}")

    @torch.no_grad()
    def add_samples(self, samples: Dict[str, torch.Tensor],
                    update_immediately: bool = False) -> Dict[str, Any]:
        """
        添加新样本到中心库（支持批量与流式）
        """
        results = {}
        for modality, embeddings in samples.items():
            if modality not in self.modalities:
                logger.warning(f"Unknown modality: {modality}")
                continue

            # dtype / device 策略
            if self.config.use_fp16 and embeddings.is_cuda:
                embeddings = embeddings.half()
            else:
                embeddings = embeddings.float()

            # 归一化
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # 流缓冲：统一放 CPU，便于聚类与 cdist
            embeddings_cpu = embeddings.detach().to('cpu', dtype=torch.float32)
            self.stream_buffers[modality].extend(embeddings_cpu.unbind(0))

            should_update = (update_immediately or
                             len(self.stream_buffers[modality]) >= self.config.incremental_batch_size)
            if should_update:
                if self.config.async_update:
                    future = self.executor.submit(self._update_modality_centers, modality)
                    results[f"{modality}_update_future"] = future
                else:
                    results[f"{modality}_update"] = self._update_modality_centers(modality)

        self.stats["total_samples"] += sum(len(s) for s in samples.values())

        if self.stats["total_samples"] > 0 and \
           self.stats["total_samples"] % max(1, self.config.checkpoint_interval) == 0:
            self._save_checkpoint()

        return results

    def _update_modality_centers(self, modality: str) -> Dict[str, Any]:
        """Update centers for a specific modality."""
        with self.update_locks[modality]:
            buffer = self.stream_buffers[modality]
            if len(buffer) < self.config.min_samples_for_update:
                return {"status": "insufficient_samples", "buffer_size": len(buffer)}

            embeddings = torch.stack(list(buffer))  # [N, D] (CPU, float32)
            self.stream_buffers[modality].clear()

            clusterer = self.clusterers[modality]
            clusterer.partial_fit(embeddings)

            new_centers = clusterer.centers  # CPU float32
            momentum = self.config.update_momentum

            # 取出参数与设备
            param = self.modal_centers[modality].centers
            old_centers = param.data.detach().to('cpu', dtype=torch.float32)

            updated = momentum * old_centers + (1.0 - momentum) * new_centers
            updated = F.normalize(updated, p=2, dim=1)

            # 回写到参数设备/精度
            param.data = updated.to(param.device, dtype=param.dtype)

            if self.config.use_faiss and modality in self.indices:
                self._update_faiss_index(modality, param.data)

            self.stats["updates_count"] += 1
            self.stats["last_update"] = datetime.now().isoformat()

            return {
                "status": "success",
                "samples_processed": len(embeddings),
                "centers_updated": len(updated)
            }

    def _update_faiss_index(self, modality: str, centers: torch.Tensor):
        """Update FAISS index."""
        if not FAISS_AVAILABLE:
            return
        index = self.indices[modality]
        index.reset()
        centers_np = centers.detach().float().cpu().numpy()
        if not index.is_trained:
            index.train(centers_np)
        else:
            # 可选：定期重训练（此处略）
            pass
        index.add(centers_np)

    def query_centers(self, query_embeddings: torch.Tensor,
                      modality: str, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        查询最相似的中心；返回 (distances, indices)
        - FAISS：返回 L2 距离
        - Torch：返回余弦距离 (2 - 2*cos)
        """
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality: {modality}")

        # 归一化
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        # 安全的 top-k
        k = max(1, min(k, self.config.num_centers))

        if self.config.use_faiss and modality in self.indices:
            query_np = query_embeddings.detach().float().cpu().numpy()
            distances, indices = self.indices[modality].search(query_np, k)
            return torch.from_numpy(distances), torch.from_numpy(indices)
        else:
            centers = self.modal_centers[modality].centers.detach()
            centers = centers.to(query_embeddings.device, dtype=query_embeddings.dtype)
            sims = query_embeddings @ centers.t()  # [B, K]
            topk_sims, topk_idx = torch.topk(sims, k, dim=1)
            distances = 2.0 - 2.0 * topk_sims  # 余弦距离
            return distances, topk_idx

    def _save_checkpoint(self):
        """Save checkpoint."""
        checkpoint_data = {
            "modal_centers": {k: v.centers.data.cpu() for k, v in self.modal_centers.items()},
            "stats": self.stats,
            "config": self.config.__dict__,
            "timestamp": datetime.now().isoformat()
        }
        version_id = self.version_manager.save_version(checkpoint_data, version_tag="checkpoint")
        self.stats["version_history"].append(version_id)
        logger.info(f"Saved checkpoint: {version_id}")

    def load_checkpoint(self, version_id: Optional[str] = None) -> bool:
        """Load checkpoint."""
        checkpoint_data = self.version_manager.load_version(version_id)
        if checkpoint_data is None:
            logger.error("Failed to load checkpoint")
            return False

        for modality, centers in checkpoint_data["modal_centers"].items():
            if modality in self.modal_centers:
                self.modal_centers[modality].centers.data = centers.to(
                    self.modal_centers[modality].centers.device,
                    dtype=self.modal_centers[modality].centers.dtype
                )

        self.stats.update(checkpoint_data["stats"])

        if self.config.use_faiss:
            for modality in self.modalities:
                self._update_faiss_index(modality, self.modal_centers[modality].centers)

        logger.info(f"Loaded checkpoint from version: {version_id}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        stats = dict(self.stats)
        for modality in self.modalities:
            centers = self.modal_centers[modality].centers.detach()
            stats[f"{modality}_stats"] = {
                "num_centers": int(centers.size(0)),
                "center_norm_mean": centers.norm(dim=1).mean().item(),
                "center_norm_std": centers.norm(dim=1).std().item(),
                "buffer_size": len(self.stream_buffers[modality])
            }
        return stats


# ================== 伪对生成器 ==================

class HardNegativeMiner:
    """Hard negative miner (returns [N, 2, D] negative pairs)."""

    def __init__(self, ratio: float = 0.3):
        self.ratio = ratio

    def mine(self, anchors: torch.Tensor, candidates: torch.Tensor,
             num_negatives: int) -> torch.Tensor:
        """
        Return shape [N_pairs, 2, D], each row is (anchor_i, negative_j).
        anchors: [B, D], candidates: [C, D]
        """
        anchors = F.normalize(anchors, p=2, dim=1)
        candidates = F.normalize(candidates, p=2, dim=1)
        sims = anchors @ candidates.t()  # [B, C]
        B, C = sims.shape
        num_hard = int(num_negatives * self.ratio)
        num_rand = max(0, num_negatives - num_hard)

        pairs = []
        for i in range(B):
            # 选择最相似的若干 hard negatives
            k_h = min(num_hard, C)
            _, hard_idx = torch.topk(sims[i], k=k_h, largest=True, sorted=False)

            # 随机补齐
            remain = torch.tensor(list(set(range(C)) - set(hard_idx.tolist())),
                                  device=candidates.device)
            if remain.numel() > 0 and num_rand > 0:
                rand_sel = remain[torch.randperm(remain.numel())[:num_rand]]
                all_idx = torch.cat([hard_idx, rand_sel], dim=0)
            else:
                all_idx = hard_idx

            if all_idx.numel() > 0:
                negs = candidates[all_idx]                    # [M, D]
                anc = anchors[i].expand(all_idx.numel(), -1)  # [M, D]
                pairs.append(torch.stack([anc, negs], dim=1)) # [M, 2, D]

        return torch.cat(pairs, dim=0) if pairs else torch.empty(0, 2, anchors.size(1),
                                                                 device=anchors.device, dtype=anchors.dtype)


class QualityScorer:
    """Quality scorer (supports negative pairs of shape [N,2,D])."""

    def __init__(self, min_threshold: float = 0.3, max_threshold: float = 0.95):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def score(self, positive_pairs: torch.Tensor,
              negative_pairs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return quality score for each positive sample [B].
        positive_pairs: [B, 2, D]
        negative_pairs: [N, 2, D] or None
        """
        pos_sim = F.cosine_similarity(positive_pairs[:, 0], positive_pairs[:, 1], dim=1)  # [B]
        scores = torch.zeros_like(pos_sim)

        valid_mask = (pos_sim > self.min_threshold) & (pos_sim < self.max_threshold)
        scores[valid_mask] = (pos_sim[valid_mask] - self.min_threshold) / (
            self.max_threshold - self.min_threshold + 1e-8
        )

        if negative_pairs is not None and negative_pairs.numel() > 0:
            neg_sim = F.cosine_similarity(negative_pairs[:, 0], negative_pairs[:, 1], dim=1)  # [N]
            # 分离度：正负均值差
            separation = pos_sim.mean() - neg_sim.mean()
            separation_bonus = torch.sigmoid(separation * 5.0)
            scores = scores * 0.7 + separation_bonus * 0.3

        return scores


class PseudoPairGenerator(nn.Module):
    """
    Production-level pseudo-pair generator.
    Supports online learning and adaptive strategies.
    """

    def __init__(self, config: PairGeneratorConfig,
                 center_library: DynamicModalCenterLibrary):
        super().__init__()
        self.config = config
        self.center_library = center_library

    # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(float(config.temperature)))

    # Negative miner & quality scorer
        self.negative_miner = HardNegativeMiner(ratio=config.hard_negative_ratio)
        self.quality_scorer = QualityScorer(
            min_threshold=config.min_similarity_threshold,
            max_threshold=config.max_similarity_threshold
        )

        self.generation_stats = defaultdict(lambda: {"count": 0, "quality": deque(maxlen=1000)})

    def _safe_topk(self, k: int) -> int:
        return max(1, min(k, self.center_library.config.num_centers))

    def generate_pairs(self, source_embeddings: Dict[str, torch.Tensor],
                       target_modality: str,
                       return_stats: bool = False) -> Dict[str, Any]:
        """
        生成高质量伪对。
        返回：
            {
              "<src>_to_<tgt>": {
                  "positive": [B', 2, D],
                  "negative": [N, 2, D],   # 可能为空
                  "weights":  [B, k]       # 检索时的 soft 权重
              },
              "stats": {...}  # 可选
            }
        """
        results: Dict[str, Any] = {}
        stats: Dict[str, Any] = {}

        for source_modality, embeddings in source_embeddings.items():
            if source_modality == target_modality:
                continue

            # 归一化（保留原 dtype/设备）
            emb = F.normalize(embeddings, p=2, dim=1)

            # 查询相似中心
            k = self._safe_topk(self.config.top_k)
            distances, indices = self.center_library.query_centers(emb, target_modality, k=k)

            # 取中心，并在 emb 的设备/精度下计算
            target_centers_all = self.center_library.get_centers(target_modality).detach()
            target_centers_all = target_centers_all.to(emb.device, dtype=emb.dtype)
            retrieved_centers = target_centers_all[indices]  # [B, k, D]

            # 基于距离生成权重（越近权重越大）
            weights = F.softmax(-distances.to(emb.device, dtype=emb.dtype) / (self.temperature + 1e-6), dim=1)  # [B, k]
            weighted_targets = torch.sum(retrieved_centers * weights.unsqueeze(-1), dim=1)  # [B, D]
            weighted_targets = F.normalize(weighted_targets, p=2, dim=1)

            # 正样本对 [B, 2, D]
            positive_pairs = torch.stack([emb, weighted_targets], dim=1)

            # 硬负样本（相对于目标空间）
            negative_pairs = torch.empty(0, 2, emb.size(1), device=emb.device, dtype=emb.dtype)
            if self.config.hard_negative_mining and self.training:
                # 用目标空间的所有中心做候选
                negatives = self.negative_miner.mine(
                    anchors=emb,
                    candidates=target_centers_all,
                    num_negatives=max(1, self.config.negative_sampling_ratio * emb.size(0))
                )
                if negatives.numel() > 0:
                    negative_pairs = negatives

            # 质量评估 + 过滤
            quality_scores = self.quality_scorer.score(positive_pairs, negative_pairs if negative_pairs.numel() > 0 else None)
            mask = (quality_scores > self.config.confidence_threshold)
            filtered_pos = positive_pairs[mask] if mask.any() else positive_pairs[:0]

            pair_key = f"{source_modality}_to_{target_modality}"
            results[pair_key] = {
                "positive": filtered_pos,
                "negative": negative_pairs,  # 供外部可用；损失不强依赖
                "weights": weights
            }

        # Statistics
            self.generation_stats[pair_key]["count"] += int(filtered_pos.size(0))
            self.generation_stats[pair_key]["quality"].append(float(quality_scores.mean().item()))
            if return_stats:
                stats[pair_key] = {
                    "num_pairs": int(filtered_pos.size(0)),
                    "avg_quality": float(quality_scores.mean().item()),
                    "filtered_ratio": float(mask.float().mean().item())
                }

    # Adaptive temperature (dynamically adjust based on average quality)
        if self.config.adaptive_temperature and self.training and len(stats) > 0:
            avg_quality = np.mean([s["avg_quality"] for s in stats.values()])
            with torch.no_grad():
                if avg_quality < 0.5:
                    self.temperature.mul_(0.99)
                elif avg_quality > 0.8:
                    self.temperature.mul_(1.01)
                self.temperature.clamp_(0.01, 1.0)

        if return_stats:
            results["stats"] = stats
        return results

    @staticmethod
    def _info_nce_matrix(anchors: torch.Tensor, targets: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Matrix-style InfoNCE (in-batch negatives).
        anchors: [B, D], targets: [B, D]
        """
        anchors = F.normalize(anchors, p=2, dim=1)
        targets = F.normalize(targets, p=2, dim=1)
        logits = anchors @ targets.t() / max(temperature, 1e-6)  # [B, B]
        labels = torch.arange(anchors.size(0), device=anchors.device)
        return F.cross_entropy(logits, labels)

    def compute_contrastive_loss(self, pairs: Dict[str, Any]) -> torch.Tensor:
        """
        Compute contrastive loss (matrix-style InfoNCE, numerically stable and efficient).
        """
        total_loss = 0.0
        num = 0
        for key, data in pairs.items():
            if key == "stats":
                continue
            pos = data.get("positive", None)
            if pos is None or pos.numel() == 0:
                continue
            anchors = pos[:, 0]
            targets = pos[:, 1]
            loss = self._info_nce_matrix(anchors, targets, float(self.config.temperature))
            total_loss += loss
            num += 1
        if num == 0:
            return torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else 'cpu')
        return total_loss / num


# ================== 使用示例 ==================

def main():
    """Example main function."""
    # 配置
    center_config = CenterLibraryConfig(
        embedding_dim=128,       # 降低维度以加快测试
        num_centers=100,         # 减少中心数量
        online_learning_enabled=True,
        use_faiss=False,         # 禁用FAISS以避免依赖问题（本地CPU跑）
        incremental_batch_size=50,
        min_samples_for_update=10
    )

    pair_config = PairGeneratorConfig(
        top_k=5,
        temperature=0.07,
        hard_negative_mining=True
    )

    # 初始化系统
    print("Initializing system...")
    center_library = DynamicModalCenterLibrary(center_config)
    pair_generator = PseudoPairGenerator(pair_config, center_library).train()  # 训练模式：启用硬负与自适应温度

    # 模拟在线学习
    print("\nStarting online learning simulation...")
    for epoch in range(5):
        batch_size = 32
        image_features = torch.randn(batch_size, center_config.embedding_dim)
        text_features = torch.randn(batch_size, center_config.embedding_dim)
        audio_features = torch.randn(batch_size, center_config.embedding_dim)

        samples = {
            "image": image_features,
            "text": text_features,
            "audio": audio_features
        }

        # 增量更新中心库（缓冲 & 触发异步/同步更新）
        _ = center_library.add_samples(samples)

        # 生成伪对（image/audio -> text）
        source_embeddings = {"image": image_features, "audio": audio_features}
        pairs = pair_generator.generate_pairs(
            source_embeddings,
            target_modality="text",
            return_stats=True
        )

        # 计算矩阵式 InfoNCE 损失
        loss = pair_generator.compute_contrastive_loss(pairs)

        # 模拟一次简单优化步骤（这里只演示）
        opt = torch.optim.AdamW(pair_generator.parameters(), lr=1e-3)
        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")
        if epoch % 2 == 0:
            stats = center_library.get_statistics()
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Updates count: {stats['updates_count']}")

    # 保存最终版本
    print("\nSaving checkpoint...")
    center_library._save_checkpoint()
    print("Training completed and checkpoint saved")

    # 显示最终统计
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

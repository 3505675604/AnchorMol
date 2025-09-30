import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')


# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)


class AdvancedPseudoPairGenerator(nn.Module):
    """
    Advanced pseudo-pair generator for contrastive learning with configurable options.
    Supports projection heads, feature alignment, memory bank, momentum encoder, and hard negative mining.
    """
    def __init__(
        self, 
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        projection_dim: int = 128,
        temperature: float = 0.07,
        learnable_temp: bool = True,
        use_projection_head: bool = True,
        use_feature_alignment: bool = True,
        similarity_metric: str = 'cosine',
        hard_negative_mining: bool = True,
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
            
            # No gradients needed for momentum encoders
            for param in self.momentum_proj_1.parameters():
                param.requires_grad = False
            for param in self.momentum_proj_2.parameters():
                param.requires_grad = False
            for param in self.momentum_proj_3.parameters():
                param.requires_grad = False
    
    def _build_projection_head(self, input_dim, hidden_dim, output_dim, dropout_rate):
        """
        Build a projection head (MLP) for feature transformation.
        No inplace operations are used for better stability.
        """
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
        # Note: Final BatchNorm removed or use non-affine version if needed

        return nn.Sequential(*layers)
    
    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update for momentum encoders.
        """
        if hasattr(self, 'momentum_proj_1'):
            for param_q, param_k in zip(self.proj_head_1.parameters(), self.momentum_proj_1.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            for param_q, param_k in zip(self.proj_head_2.parameters(), self.momentum_proj_2.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            for param_q, param_k in zip(self.proj_head_3.parameters(), self.momentum_proj_3.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_1, keys_2, keys_3):
        """
        Update memory bank with new keys.
        """
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
        """
        Compute similarity matrix between two sets of features.
        Supports cosine, euclidean, and dot product metrics.
        """
        # Use clone to avoid modifying the original tensor
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
        
    # Ensure temperature parameter is not modified inplace
        if isinstance(self.temperature, torch.Tensor):
            temp = self.temperature.clone()
        else:
            temp = self.temperature
        
        return sim / temp
    
    def hard_negative_sampling(self, sim_matrix, labels):
        """
        Hard negative mining (fixed version, supports memory bank columns).
        Selects the hardest negatives for each sample.
        """
        if not self.hard_negative_mining:
            return sim_matrix

        B, C = sim_matrix.shape
        device = sim_matrix.device

    # Positive sample column indices: default is diagonal of first B columns
        if labels is None:
            labels = torch.arange(B, device=device)
        labels = labels.long()

    # Construct B×C positive sample mask (no inplace)
        pos_mask = F.one_hot(labels, num_classes=C).to(torch.bool)

    # Only select hardest among negatives
        neg_sim = sim_matrix.masked_fill(pos_mask, -float('inf'))

    # Select top-k hardest negatives (k ∈ [1, min(10, C-1)])
        k = min(max(B // 4, 1), max(1, C - 1))
        k = min(k, 10)
        if k <= 0:
            return sim_matrix

        hard_negatives, _ = torch.topk(neg_sim, k, dim=1)

    # Reweight
        temp = self.temperature if not isinstance(self.temperature, torch.Tensor) else self.temperature
        weights = F.softmax(hard_negatives / temp, dim=1)

        weight_factor = 1 + weights.sum(dim=1, keepdim=True) * 0.1
        weighted_sim = sim_matrix * weight_factor
        return weighted_sim

    
    def nt_xent_loss(self, z1, z2, use_queue=False):
        """
        NT-Xent loss (fixed version).
        Computes contrastive loss between two sets of features.
        """
        batch_size = z1.size(0)
        device = z1.device
        
    # Compute similarity
        sim_matrix = self.compute_similarity(z1, z2)
        
    # If using memory bank, add extra negatives
        if use_queue and self.use_memory_bank:
            sim_1_queue = self.compute_similarity(z1, self.queue_2.T.clone())
            sim_2_queue = self.compute_similarity(z2, self.queue_1.T.clone())
            sim_matrix = torch.cat([sim_matrix, sim_1_queue], dim=1)
        
    # Apply hard negative mining
        labels = torch.arange(batch_size, device=device)
        sim_matrix = self.hard_negative_sampling(sim_matrix, labels)
        
    # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def alignment_loss(self, z1, z2, z3):
        """
        Feature alignment loss (fixed version).
        Aligns features from different modalities.
        """
        if not self.use_feature_alignment:
            return torch.tensor(0.0, device=z1.device)
        
    # Use detach to avoid gradient propagation issues
        z1_aligned_to_2 = self.align_1_to_2(z1)
        align_loss_12 = F.mse_loss(z1_aligned_to_2, z2.detach())
        
        z1_aligned_to_3 = self.align_1_to_3(z1)
        align_loss_13 = F.mse_loss(z1_aligned_to_3, z3.detach())
        
        z2_aligned_to_3 = self.align_2_to_3(z2)
        align_loss_23 = F.mse_loss(z2_aligned_to_3, z3.detach())
        
        return (align_loss_12 + align_loss_13 + align_loss_23) / 3
    
    def compute_contrastive_loss(self, z1, z2, z3, return_details=False):
        """
        Compute the full contrastive loss (fixed version).
        Returns total loss and optionally details for each component.
        """
        # Use clone to avoid modifying the original features
        z1 = z1.clone()
        z2 = z2.clone()  
        z3 = z3.clone()
        
    # Pass through projection heads
        p1 = self.proj_head_1(z1)
        p2 = self.proj_head_2(z2)
        p3 = self.proj_head_3(z3)
        
    # Normalize if using cosine similarity
        if self.similarity_metric == 'cosine':
            p1_norm = F.normalize(p1, p=2, dim=1)
            p2_norm = F.normalize(p2, p=2, dim=1)
            p3_norm = F.normalize(p3, p=2, dim=1)
        else:
            p1_norm = p1
            p2_norm = p2
            p3_norm = p3
        
    # Momentum update
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
        
    # Compute contrastive losses
        loss_12 = self.nt_xent_loss(p1_norm, p2_norm, use_queue=self.use_memory_bank)
        loss_13 = self.nt_xent_loss(p1_norm, p3_norm, use_queue=self.use_memory_bank)
        loss_23 = self.nt_xent_loss(p2_norm, p3_norm, use_queue=self.use_memory_bank)
        
        contrastive_loss = (loss_12 + loss_13 + loss_23) / 3
        
    # Add alignment loss
        align_loss = self.alignment_loss(p1_norm, p2_norm, p3_norm)
        
    # Update memory bank
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
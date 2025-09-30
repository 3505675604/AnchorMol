import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForMaskedLM
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
import numpy as np
import torch.nn as nn
from Projector import Ex_MCR_Head
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

class SMILESFeatureExtractor(nn.Module):
    """
    Feature extractor for SMILES strings using a pretrained transformer model (e.g., ChemBERTa).
    Projects the output to a fixed dimension (default 512).
    """
    def __init__(self, model_path, out_dim=512, device=None, train_encoder=True):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.encoder = AutoModelForMaskedLM.from_pretrained(model_path)  # with .roberta
        self.encoder.to(self.device)
        # If you only want to fine-tune a little, freeze encoder and only train projection layer
        for p in self.encoder.parameters():
            p.requires_grad = bool(train_encoder)
        # Use a linear layer to project hidden_size to 512, instead of padding/truncating
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, smiles_list, max_length=256):
        """
        Encode a list of SMILES strings and return projected [CLS] features.
        Args:
            smiles_list: List of SMILES strings
            max_length: Maximum token length (default 256)
        Returns:
            Tensor of shape [batch, out_dim]
        """
        tok = self.tokenizer(smiles_list, padding=True, truncation=True,
                             max_length=max_length, return_tensors='pt')
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.encoder.roberta(**tok)     # or: self.encoder(**tok, return_dict=True).logits also works
        cls = out.last_hidden_state[:, 0, :]  # <s>/CLS vector
        return self.proj(cls)                 # [B, 512]

class GROVERFeatureExtractor(nn.Module):
    """
    Feature extractor for SMILES using a pretrained GROVER model.
    Projects the output to a fixed dimension (default 512).
    """
    def __init__(self, model_path, out_dim=512, device=None, train_encoder=True):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.encoder = AutoModelForMaskedLM.from_pretrained(model_path)
        self.encoder.to(self.device)
        for p in self.encoder.parameters():
            p.requires_grad = bool(train_encoder)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, smiles_list, max_length=256):
        """
        Encode a list of SMILES strings and return mean pooled features.
        Args:
            smiles_list: List of SMILES strings
            max_length: Maximum token length (default 256)
        Returns:
            Tensor of shape [batch, out_dim]
        """
        tok = self.tokenizer(smiles_list, padding=True, truncation=True,
                             max_length=max_length, return_tensors='pt')
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.encoder(**tok, output_hidden_states=True, return_dict=True)
        last = out.hidden_states[-1]             # [B, L, H]
        mask = tok['attention_mask'].unsqueeze(-1).float()
        mean = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.proj(mean)                   # [B, 512]


def generate_3d_mol(smile):
    """
    Generate a 3D molecule from a SMILES string using RDKit.
    Returns None if generation fails.
    """
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if success != 0:
            return None
        AllChem.UFFOptimizeMolecule(mol)
        return mol
    except:
        return None
#3D feature extraction case, please replace it with the corresponding 3DDimeNet
def get_3d_descriptors(mol):
    """
    Calculate 10 types of 3D molecular descriptors for a molecule.
    Returns a numpy array of shape (10,).
    """
    try:
        desc = [
            Descriptors3D.Asphericity(mol),
            Descriptors3D.Eccentricity(mol),
            Descriptors3D.InertialShapeFactor(mol),
            Descriptors3D.NPR1(mol),
            Descriptors3D.NPR2(mol),
            Descriptors3D.PMI1(mol),
            Descriptors3D.PMI2(mol),
            Descriptors3D.PMI3(mol),
            Descriptors3D.RadiusOfGyration(mol),
            Descriptors3D.SpherocityIndex(mol),
        ]
        return np.array(desc, dtype=np.float32)
    except:
        return np.zeros(10, dtype=np.float32)

def extract_3d_features(smiles_list: List[str]) -> torch.Tensor:
    """
    Extract 3D descriptors for a list of SMILES and project to 512-dim features using an MLP.
    Returns a tensor of shape [N, 512].
    """
    features = []
    for smile in smiles_list:
        mol = generate_3d_mol(smile)
        if mol is None:
            features.append(np.zeros(10, dtype=np.float32))
        else:
            desc = get_3d_descriptors(mol)
            features.append(desc)
    features = np.stack(features, axis=0)
    features = torch.tensor(features)
    mlp = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
    )
    with torch.no_grad():
        output = mlp(features)
    return output  # [N, 512], torch.Tensor

class MultiModalFeatureExtractor(nn.Module):
    """
    Extracts three types of features for a list of SMILES:
    - Sequence features (ChemBERTa or similar)
    - Graph features (GROVER or similar)
    - 3D descriptors (projected by a trainable MLP)
    Returns three 512-dim feature tensors for each input.
    """
    def __init__(self, bert_path, grover_path, device=None, train_encoders=True):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_extractor  = SMILESFeatureExtractor(bert_path,  out_dim=512, device=self.device, train_encoder=train_encoders)
        self.grover_extractor= GROVERFeatureExtractor(grover_path,out_dim=512, device=self.device, train_encoder=train_encoders)
        self.projector = Ex_MCR_Head().to(self.device)
        self.mlp3d = nn.Sequential(
            nn.Linear(10,128), nn.ReLU(),
            nn.Linear(128,256), nn.ReLU(),
            nn.Linear(256,512)
        ).to(self.device)

    def forward(self, smiles_list: List[str]):
        """
        Args:
            smiles_list: List of SMILES strings
        Returns:
            emb_graph, emb_seq, emb_3d: Three [B,512] feature tensors
        """
        graph_feat    = self.grover_extractor(smiles_list)    # [B,512]
        sequence_feat = self.bert_extractor(smiles_list)      # [B,512]

        # 3D descriptors (not differentiable), but mlp3d is trainable
        feats = []
        for s in smiles_list:
            mol = generate_3d_mol(s)
            feats.append(get_3d_descriptors(mol) if mol is not None else np.zeros(10, np.float32))
        feat3d = torch.tensor(np.stack(feats), dtype=torch.float32, device=self.device)
        feat3d = self.mlp3d(feat3d)                           # [B,512]

        emb_seq, emb_graph, emb_3d = self.projector(sequence_feat, graph_feat, feat3d)
        return emb_graph, emb_seq, emb_3d

    


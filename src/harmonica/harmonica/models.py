"""
Semantic Learning Script for ENIGMA-PD Harmonizer
=================================================

This script trains models that learn CLINICAL SEMANTICS from expert assignments.

Input: finetuning_dataset.csv (contains item texts, E_ik distributions, questionnaire info)

Models:
1. PROTOTYPE MODEL:
   - Learns dimension prototypes from expert-assigned items
   - Prototype = weighted average of embeddings (weighted by E_ik)
   - Fast, no GPU required

2. CONTRASTIVE MODEL:  
   - Fine-tunes embedding model with contrastive learning
   - Same dimension items → pull together
   - Different dimension items → push apart
   - Deeper semantic learning, GPU recommended

Both models ALSO learn calibration (weights, biases, thresholds) on top of
the semantic representations.

Training signal: expert_proportions_e_ik (E_ik) - the soft label distribution from experts.
Note: The combined evidence score S_i (which combines expert + model confidence) is NOT 
      used for training - that's only for paper reporting.

Usage:
------
# Prototype model (fast, CPU)
python semantic_learning.py \
    --data finetuning_dataset.csv \
    --construct depression \
    --approach prototype \
    --output ./models

# Contrastive model (slower, GPU recommended)
python semantic_learning.py \
    --data finetuning_dataset.csv \
    --construct depression \
    --approach contrastive \
    --output ./models \
    --device cuda

# Cross-validation
python semantic_learning.py \
    --data finetuning_dataset.csv \
    --construct depression \
    --approach both \
    --cv \
    --output ./models
"""

import argparse
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pickle
import warnings

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from scipy.optimize import minimize

# Optional imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Contrastive model will not work.")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    warnings.warn("sentence-transformers not available.")


# =============================================================================
# Data Loading
# =============================================================================

def parse_e_ik(e_ik_str: str) -> Dict[int, float]:
    """Parse expert_proportions_e_ik column to dictionary."""
    if pd.isna(e_ik_str) or not e_ik_str:
        return {}
    try:
        # It's stored as a JSON-like dict string
        return {int(k): float(v) for k, v in ast.literal_eval(e_ik_str).items()}
    except:
        return {}


def load_construct_data(data_path: str, construct: str) -> pd.DataFrame:
    """Load data for a specific construct from finetuning_dataset.csv."""
    df = pd.read_csv(data_path)
    df = df[df['construct'] == construct].copy()
    
    # Parse E_ik
    df['e_ik_dict'] = df['expert_proportions_e_ik'].apply(parse_e_ik)
    
    return df


def prepare_training_data(
    df: pd.DataFrame
) -> Tuple[List[str], np.ndarray, np.ndarray, List[int], List[str], List[str]]:
    """
    Prepare training data from finetuning_dataset.csv.
    
    Returns:
        item_texts: List of question texts
        y_hard: Hard labels (final_dimension)
        y_soft: Soft labels matrix (n_items x n_dimensions) from E_ik
        dimensions: List of dimension IDs (including -1)
        dimension_labels: Human-readable dimension names
        questionnaires: List of questionnaire names (for LOGO CV)
    """
    # Get all dimensions from E_ik
    all_dims = set()
    for e_ik in df['e_ik_dict']:
        all_dims.update(e_ik.keys())
    
    # Sort dimensions: -1 first, then positive integers
    dimensions = sorted([d for d in all_dims if d == -1]) + sorted([d for d in all_dims if d != -1])
    if -1 not in dimensions:
        dimensions = [-1] + dimensions
    
    n_items = len(df)
    n_dims = len(dimensions)
    
    # Build arrays
    item_texts = df['question_text'].tolist()
    questionnaires = df['questionnaire'].tolist()
    y_hard = df['final_dimension'].values.astype(int)
    
    # Build soft label matrix from E_ik
    y_soft = np.zeros((n_items, n_dims))
    for i, e_ik in enumerate(df['e_ik_dict']):
        for dim, prob in e_ik.items():
            dim_idx = dimensions.index(dim)
            y_soft[i, dim_idx] = prob
        # Normalize (should already sum to 1, but ensure)
        if y_soft[i].sum() > 0:
            y_soft[i] /= y_soft[i].sum()
    
    # Get dimension labels (from first occurrence of each dimension)
    dim_to_label = {}
    for _, row in df.iterrows():
        dim = row['final_dimension']
        label = row['final_dimension_label']
        if dim not in dim_to_label:
            dim_to_label[dim] = label
    
    # Add -1 label if not present
    if -1 not in dim_to_label:
        dim_to_label[-1] = "Does not fit"
    
    dimension_labels = [dim_to_label.get(d, f"Dimension {d}") for d in dimensions]
    
    return item_texts, y_hard, y_soft, dimensions, dimension_labels, questionnaires


# =============================================================================
# Prototype Model
# =============================================================================

class PrototypeModel:
    """
    Prototype-based model that learns dimension representations from items.
    
    Learns:
    1. Prototype embeddings for each dimension (weighted by E_ik)
    2. Calibration weights/biases to adjust prototype similarities
    3. Thresholds for "does not fit" (-1) detection
    
    Does NOT use expert_confidence_p_i (S_i) - only E_ik soft labels.
    """
    
    def __init__(self, model_name: str = "hkunlp/instructor-large", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.encoder = None
        self.prototypes = None
        self.weights = None
        self.biases = None
        self.tau_max = 0.5
        self.tau_margin = 0.02
        self.p_minus1_base = 0.1
        self.temperature = 1.0
        self.dimensions = None
        self.dimension_labels = None
        self.n_real_dims = None  # Excludes -1
        self.minus1_idx = None
        self.fitted = False
    
    def fit(
        self,
        item_texts: List[str],
        y_soft: np.ndarray,
        dimensions: List[int],
        dimension_labels: List[str]
    ):
        """
        Learn prototypes and calibration from expert E_ik distributions.
        
        Parameters
        ----------
        item_texts : List[str]
            Question texts for computing embeddings
        y_soft : np.ndarray
            Soft labels (n_items x n_dimensions) from E_ik
        dimensions : List[int]
            Dimension IDs (including -1)
        dimension_labels : List[str]
            Human-readable dimension names
        """
        print(f"    Loading encoder: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name, device=self.device)
        
        self.dimensions = dimensions
        self.dimension_labels = dimension_labels
        self.minus1_idx = dimensions.index(-1) if -1 in dimensions else None
        
        # Real dimensions (excluding -1)
        self.real_dim_indices = [i for i, d in enumerate(dimensions) if d != -1]
        self.n_real_dims = len(self.real_dim_indices)
        self.real_dim_labels = [dimension_labels[i] for i in self.real_dim_indices]
        
        # Compute embeddings
        print(f"    Computing embeddings for {len(item_texts)} items...")
        embeddings = self.encoder.encode(item_texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Get E_ik for real dimensions only (excluding -1)
        e_dims = y_soft[:, self.real_dim_indices]
        
        # Get E_{i,-1} for threshold learning
        if self.minus1_idx is not None:
            e_minus1 = y_soft[:, self.minus1_idx]
        else:
            e_minus1 = np.zeros(len(item_texts))
        
        # Learn prototypes
        print(f"    Learning prototypes for {self.n_real_dims} dimensions...")
        embedding_dim = embeddings.shape[1]
        self.prototypes = np.zeros((self.n_real_dims, embedding_dim))
        
        for k in range(self.n_real_dims):
            weights_k = e_dims[:, k]
            if weights_k.sum() > 0:
                self.prototypes[k] = np.average(embeddings, axis=0, weights=weights_k + 1e-10)
                self.prototypes[k] /= (np.linalg.norm(self.prototypes[k]) + 1e-10)
                print(f"      {self.real_dim_labels[k][:40]}: {weights_k.sum():.1f} weighted items")
            else:
                print(f"      {self.real_dim_labels[k][:40]}: WARNING - no items!")
        
        # Compute prototype similarities for all items
        similarities = embeddings @ self.prototypes.T
        max_sim = np.max(similarities, axis=1)
        sorted_sims = np.sort(similarities, axis=1)[:, ::-1]
        margin = sorted_sims[:, 0] - sorted_sims[:, 1] if self.n_real_dims > 1 else sorted_sims[:, 0]
        
        # Learn thresholds for -1 detection (from E_{i,-1})
        print(f"    Learning -1 thresholds from E_{{i,-1}}...")
        self._learn_thresholds(max_sim, margin, e_minus1)
        
        # Learn calibration weights and biases (from E_{i,k})
        print(f"    Learning calibration weights from E_{{i,k}}...")
        self._learn_calibration(similarities, max_sim, margin, e_dims, e_minus1)
        
        self.fitted = True
        self._print_params()
        
        return self
    
    def _learn_thresholds(self, max_sim, margin, e_minus1):
        """Learn thresholds for -1 detection from E_{i,-1}."""
        if e_minus1.sum() == 0:
            print(f"      No -1 votes in data, using defaults")
            return
        
        best_score = -np.inf
        for tau_max in np.arange(0.30, 0.75, 0.025):
            for tau_margin in np.arange(0.005, 0.12, 0.01):
                for p_base in [0.02, 0.05, 0.10, 0.15]:
                    p_minus1 = self._compute_p_minus1(max_sim, margin, tau_max, tau_margin, p_base)
                    if np.std(p_minus1) > 0 and np.std(e_minus1) > 0:
                        corr = np.corrcoef(p_minus1, e_minus1)[0, 1]
                        if not np.isnan(corr) and corr > best_score:
                            best_score = corr
                            self.tau_max = tau_max
                            self.tau_margin = tau_margin
                            self.p_minus1_base = p_base
        
        print(f"      Best correlation with E_{{i,-1}}: {best_score:.3f}")
    
    def _learn_calibration(self, similarities, max_sim, margin, e_dims, e_minus1):
        """Learn calibration weights and biases to match E_{i,k}."""
        self.weights = np.ones(self.n_real_dims)
        self.biases = np.zeros(self.n_real_dims)
        
        def objective(params):
            w = params[:self.n_real_dims]
            b = params[self.n_real_dims:2*self.n_real_dims]
            temp = params[2*self.n_real_dims]
            
            # Calibrated similarities
            calibrated = similarities * w + b
            
            # Softmax
            scaled = calibrated / (temp + 1e-10)
            exp_scores = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            probs = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-10)
            
            # Adjust for P(-1)
            if self.minus1_idx is not None:
                p_m1 = self._compute_p_minus1(max_sim, margin, self.tau_max, self.tau_margin, self.p_minus1_base)
                probs = probs * (1 - p_m1[:, np.newaxis])
            
            # KL divergence from E_{i,k}
            e_safe = e_dims + 1e-10
            e_safe = e_safe / e_safe.sum(axis=1, keepdims=True)
            p_safe = probs + 1e-10
            p_safe = p_safe / p_safe.sum(axis=1, keepdims=True)
            
            kl = np.sum(e_safe * (np.log(e_safe) - np.log(p_safe)), axis=1).mean()
            
            # Regularization
            reg = 0.01 * (np.sum((w - 1)**2) + np.sum(b**2))
            
            return kl + reg
        
        x0 = np.concatenate([np.ones(self.n_real_dims), np.zeros(self.n_real_dims), [1.0]])
        bounds = [(0.1, 5.0)] * self.n_real_dims + [(-1.0, 1.0)] * self.n_real_dims + [(0.1, 5.0)]
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
        
        self.weights = result.x[:self.n_real_dims]
        self.biases = result.x[self.n_real_dims:2*self.n_real_dims]
        self.temperature = result.x[2*self.n_real_dims]
    
    def _compute_p_minus1(self, max_sim, margin, tau_max, tau_margin, p_base):
        """Compute P(-1) based on similarity features."""
        dist_max = (tau_max - max_sim) / (tau_max + 1e-10)
        dist_margin = (tau_margin - margin) / (tau_margin + 1e-10)
        dist = np.minimum(dist_max, dist_margin)
        return np.clip(p_base + (1 - p_base) / (1 + np.exp(-5 * dist)), 0.01, 0.99)
    
    def _print_params(self):
        """Print learned parameters."""
        print(f"\n    Learned parameters:")
        print(f"      Thresholds: τ_max={self.tau_max:.3f}, τ_margin={self.tau_margin:.3f}, p_base={self.p_minus1_base:.2f}")
        print(f"      Temperature: {self.temperature:.3f}")
        print(f"      Dimension weights and biases:")
        for i, label in enumerate(self.real_dim_labels):
            print(f"        {label[:40]:<40}: w={self.weights[i]:.3f}, b={self.biases[i]:+.3f}")
    
    def predict(self, item_texts: List[str]) -> np.ndarray:
        """Predict dimensions for new items."""
        proba = self.predict_proba(item_texts)
        pred_idx = np.argmax(proba, axis=1)
        return np.array([self.dimensions[i] for i in pred_idx])
    
    def predict_proba(self, item_texts: List[str]) -> np.ndarray:
        """Predict probability distribution over all dimensions."""
        # Compute embeddings
        embeddings = self.encoder.encode(item_texts, convert_to_numpy=True)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Similarity to prototypes
        similarities = embeddings @ self.prototypes.T
        max_sim = np.max(similarities, axis=1)
        sorted_sims = np.sort(similarities, axis=1)[:, ::-1]
        margin = sorted_sims[:, 0] - sorted_sims[:, 1] if self.n_real_dims > 1 else sorted_sims[:, 0]
        
        # Calibrate
        calibrated = similarities * self.weights + self.biases
        scaled = calibrated / (self.temperature + 1e-10)
        exp_scores = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        dim_probs = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-10)
        
        # P(-1)
        p_minus1 = self._compute_p_minus1(max_sim, margin, self.tau_max, self.tau_margin, self.p_minus1_base)
        
        # Build full probability matrix
        n_samples = len(item_texts)
        proba = np.zeros((n_samples, len(self.dimensions)))
        
        if self.minus1_idx is not None:
            proba[:, self.minus1_idx] = p_minus1
            for j, real_idx in enumerate(self.real_dim_indices):
                proba[:, real_idx] = dim_probs[:, j] * (1 - p_minus1)
        else:
            for j, real_idx in enumerate(self.real_dim_indices):
                proba[:, real_idx] = dim_probs[:, j]
        
        return proba
    
    def save(self, path: str):
        """Save model to file."""
        # Save encoder
        encoder_path = str(Path(path).parent / f"{Path(path).stem}_encoder")
        self.encoder.save(encoder_path)
        
        state = {
            'prototypes': self.prototypes,
            'weights': self.weights,
            'biases': self.biases,
            'tau_max': self.tau_max,
            'tau_margin': self.tau_margin,
            'p_minus1_base': self.p_minus1_base,
            'temperature': self.temperature,
            'dimensions': self.dimensions,
            'dimension_labels': self.dimension_labels,
            'real_dim_indices': self.real_dim_indices,
            'real_dim_labels': self.real_dim_labels,
            'n_real_dims': self.n_real_dims,
            'minus1_idx': self.minus1_idx,
            'model_name': self.model_name,
            'encoder_path': encoder_path
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"    Saved model to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'PrototypeModel':
        """Load model from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(model_name=state['model_name'], device=device)
        for key, value in state.items():
            if key not in ['encoder_path', 'model_name']:
                setattr(model, key, value)
        
        # Derive encoder path from pkl location (encoder dir is always a sibling)
        encoder_path = str(Path(path).parent / f"{Path(path).stem}_encoder")
        model.encoder = SentenceTransformer(encoder_path, device=device)
        model.fitted = True
        return model


# =============================================================================
# Contrastive Model
# =============================================================================

class ContrastiveModel:
    """
    Contrastive fine-tuning model for deep semantic learning.
    
    Fine-tunes the embedding model so that:
    - Items in same dimension → closer in embedding space
    - Items in different dimensions → farther apart
    
    Uses E_ik to weight the strength of pull/push:
    - High overlap in E_ik → strong pull (positive pair)
    - High confidence in different dimensions → strong push (negative pair)
    
    Then learns prototypes and calibration on fine-tuned embeddings.
    
    Does NOT use expert_confidence_p_i (S_i) - only E_ik soft labels.
    """
    
    def __init__(
        self,
        model_name: str = "hkunlp/instructor-large",
        device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
        learning_rate: float = 2e-5,
        epochs: int = 5,
        batch_size: int = 8,
        margin: float = 0.3
    ):
        self.model_name = model_name
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.margin = margin
        
        self.encoder = None
        self.prototypes = None
        self.weights = None
        self.biases = None
        self.tau_max = 0.5
        self.tau_margin = 0.02
        self.p_minus1_base = 0.1
        self.temperature = 1.0
        self.dimensions = None
        self.dimension_labels = None
        self.real_dim_indices = None
        self.real_dim_labels = None
        self.n_real_dims = None
        self.minus1_idx = None
        self.fitted = False
    
    def _create_pairs(
        self, 
        item_texts: List[str], 
        y_soft: np.ndarray,
        real_dim_indices: List[int]
    ) -> List[Tuple]:
        """
        Create contrastive pairs weighted by E_ik.
        
        Positive pairs: items assigned to same dimension
        Negative pairs: items assigned to different dimensions
        Weight by confidence in E_ik.
        """
        pairs = []
        n = len(item_texts)
        
        # Get E_ik for real dimensions only
        e_dims = y_soft[:, real_dim_indices]
        
        for i in range(n):
            for j in range(i + 1, n):
                e_i = e_dims[i]
                e_j = e_dims[j]
                
                # Normalize
                e_i = e_i / (e_i.sum() + 1e-10)
                e_j = e_j / (e_j.sum() + 1e-10)
                
                # Hard dimension assignment
                dim_i = np.argmax(e_i)
                dim_j = np.argmax(e_j)
                
                # Soft overlap: how much experts agree these are in same dimension
                overlap = np.sum(np.minimum(e_i, e_j))
                
                if dim_i == dim_j:
                    # Positive pair
                    label = 1
                    weight = overlap  # Higher weight if both strongly assigned to same dim
                else:
                    # Negative pair
                    label = 0
                    weight = e_i[dim_i] * e_j[dim_j]  # Weight by confidence of different assignments
                
                if weight > 0.05:  # Skip very low confidence pairs
                    pairs.append((item_texts[i], item_texts[j], label, weight))
        
        return pairs
    
    def fit(
        self,
        item_texts: List[str],
        y_soft: np.ndarray,
        dimensions: List[int],
        dimension_labels: List[str]
    ):
        """Fine-tune with contrastive learning, then learn prototypes and calibration."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for contrastive learning")
        
        self.dimensions = dimensions
        self.dimension_labels = dimension_labels
        self.minus1_idx = dimensions.index(-1) if -1 in dimensions else None
        self.real_dim_indices = [i for i, d in enumerate(dimensions) if d != -1]
        self.n_real_dims = len(self.real_dim_indices)
        self.real_dim_labels = [dimension_labels[i] for i in self.real_dim_indices]
        
        print(f"    Loading encoder: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name, device=self.device)
        
        # Create contrastive pairs
        print(f"    Creating contrastive pairs from E_{{i,k}}...")
        pairs = self._create_pairs(item_texts, y_soft, self.real_dim_indices)
        n_pos = sum(1 for p in pairs if p[2] == 1)
        n_neg = len(pairs) - n_pos
        print(f"      {len(pairs)} pairs: {n_pos} positive, {n_neg} negative")
        
        if len(pairs) < 20:
            print(f"      Too few pairs - falling back to prototype learning only")
            return self._fit_prototype_only(item_texts, y_soft)
        
        # Fine-tuning loop
        print(f"    Fine-tuning encoder ({self.epochs} epochs)...")
        
        # Get the underlying model for training with gradients
        # SentenceTransformer wraps a transformer model
        self.encoder.train()
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.epochs):
            np.random.shuffle(pairs)
            total_loss = 0
            n_batches = 0
            
            for batch_start in range(0, len(pairs), self.batch_size):
                batch = pairs[batch_start:batch_start + self.batch_size]
                texts_i = [p[0] for p in batch]
                texts_j = [p[1] for p in batch]
                labels = torch.tensor([p[2] for p in batch], dtype=torch.float32, device=self.device)
                weights = torch.tensor([p[3] for p in batch], dtype=torch.float32, device=self.device)
                
                # Tokenize
                features_i = self.encoder.tokenize(texts_i)
                features_j = self.encoder.tokenize(texts_j)
                
                # Move to device
                features_i = {k: v.to(self.device) for k, v in features_i.items()}
                features_j = {k: v.to(self.device) for k, v in features_j.items()}
                
                # Forward pass with gradients
                output_i = self.encoder.forward(features_i)
                output_j = self.encoder.forward(features_j)
                
                # Get sentence embeddings
                emb_i = output_i['sentence_embedding']
                emb_j = output_j['sentence_embedding']
                
                # Normalize
                emb_i = torch.nn.functional.normalize(emb_i, p=2, dim=1)
                emb_j = torch.nn.functional.normalize(emb_j, p=2, dim=1)
                
                # Cosine similarity
                sim = torch.sum(emb_i * emb_j, dim=1)
                
                # Contrastive loss
                pos_loss = labels * weights * (1 - sim) ** 2
                neg_loss = (1 - labels) * weights * torch.clamp(sim - self.margin, min=0) ** 2
                loss = (pos_loss + neg_loss).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            print(f"      Epoch {epoch+1}/{self.epochs}: loss={total_loss/n_batches:.4f}")
        
        self.encoder.eval()
        
        # Compute fine-tuned embeddings
        print(f"    Computing fine-tuned embeddings...")
        with torch.no_grad():
            embeddings = self.encoder.encode(item_texts, convert_to_numpy=True)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Learn prototypes and calibration
        self._learn_prototypes_and_calibration(embeddings, y_soft)
        
        self.fitted = True
        return self
    
    def _fit_prototype_only(self, item_texts, y_soft):
        """Fallback when not enough pairs for contrastive learning."""
        embeddings = self.encoder.encode(item_texts, convert_to_numpy=True)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        self._learn_prototypes_and_calibration(embeddings, y_soft)
        self.fitted = True
        return self
    
    def _learn_prototypes_and_calibration(self, embeddings, y_soft):
        """Learn prototypes and calibration from embeddings and E_ik."""
        # Get E_ik for real dimensions
        e_dims = y_soft[:, self.real_dim_indices]
        
        # E_{i,-1}
        if self.minus1_idx is not None:
            e_minus1 = y_soft[:, self.minus1_idx]
        else:
            e_minus1 = np.zeros(len(embeddings))
        
        # Learn prototypes
        print(f"    Learning prototypes...")
        embedding_dim = embeddings.shape[1]
        self.prototypes = np.zeros((self.n_real_dims, embedding_dim))
        
        for k in range(self.n_real_dims):
            w_k = e_dims[:, k]
            if w_k.sum() > 0:
                self.prototypes[k] = np.average(embeddings, axis=0, weights=w_k + 1e-10)
                self.prototypes[k] /= (np.linalg.norm(self.prototypes[k]) + 1e-10)
        
        # Compute similarities
        similarities = embeddings @ self.prototypes.T
        max_sim = np.max(similarities, axis=1)
        sorted_sims = np.sort(similarities, axis=1)[:, ::-1]
        margin = sorted_sims[:, 0] - sorted_sims[:, 1] if self.n_real_dims > 1 else sorted_sims[:, 0]
        
        # Learn thresholds
        print(f"    Learning thresholds from E_{{i,-1}}...")
        if e_minus1.sum() > 0:
            best_score = -np.inf
            for tau_max in np.arange(0.30, 0.75, 0.025):
                for tau_margin in np.arange(0.005, 0.12, 0.01):
                    for p_base in [0.02, 0.05, 0.10, 0.15]:
                        p_m1 = self._compute_p_minus1(max_sim, margin, tau_max, tau_margin, p_base)
                        if np.std(p_m1) > 0 and np.std(e_minus1) > 0:
                            corr = np.corrcoef(p_m1, e_minus1)[0, 1]
                            if not np.isnan(corr) and corr > best_score:
                                best_score = corr
                                self.tau_max, self.tau_margin, self.p_minus1_base = tau_max, tau_margin, p_base
        
        # Learn calibration
        print(f"    Learning calibration from E_{{i,k}}...")
        self.weights = np.ones(self.n_real_dims)
        self.biases = np.zeros(self.n_real_dims)
        
        def objective(params):
            w = params[:self.n_real_dims]
            b = params[self.n_real_dims:2*self.n_real_dims]
            temp = params[2*self.n_real_dims]
            
            cal = similarities * w + b
            exp_s = np.exp(cal / temp - np.max(cal / temp, axis=1, keepdims=True))
            probs = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-10)
            
            if self.minus1_idx is not None:
                p_m1 = self._compute_p_minus1(max_sim, margin, self.tau_max, self.tau_margin, self.p_minus1_base)
                probs = probs * (1 - p_m1[:, np.newaxis])
            
            e_safe = (e_dims + 1e-10) / (e_dims + 1e-10).sum(axis=1, keepdims=True)
            p_safe = (probs + 1e-10) / (probs + 1e-10).sum(axis=1, keepdims=True)
            kl = np.sum(e_safe * (np.log(e_safe) - np.log(p_safe)), axis=1).mean()
            
            return kl + 0.01 * (np.sum((w-1)**2) + np.sum(b**2))
        
        x0 = np.concatenate([np.ones(self.n_real_dims), np.zeros(self.n_real_dims), [1.0]])
        bounds = [(0.1, 5.0)] * self.n_real_dims + [(-1.0, 1.0)] * self.n_real_dims + [(0.1, 5.0)]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        self.weights = result.x[:self.n_real_dims]
        self.biases = result.x[self.n_real_dims:2*self.n_real_dims]
        self.temperature = result.x[2*self.n_real_dims]
        
        # Print results
        print(f"\n    Learned parameters:")
        print(f"      Thresholds: τ_max={self.tau_max:.3f}, τ_margin={self.tau_margin:.3f}, p_base={self.p_minus1_base:.2f}")
        print(f"      Temperature: {self.temperature:.3f}")
        for i, label in enumerate(self.real_dim_labels):
            print(f"      {label[:40]:<40}: w={self.weights[i]:.3f}, b={self.biases[i]:+.3f}")
    
    def _compute_p_minus1(self, max_sim, margin, tau_max, tau_margin, p_base):
        dist = np.minimum((tau_max - max_sim) / (tau_max + 1e-10), 
                          (tau_margin - margin) / (tau_margin + 1e-10))
        return np.clip(p_base + (1 - p_base) / (1 + np.exp(-5 * dist)), 0.01, 0.99)
    
    def predict(self, item_texts: List[str]) -> np.ndarray:
        proba = self.predict_proba(item_texts)
        return np.array([self.dimensions[i] for i in np.argmax(proba, axis=1)])
    
    def predict_proba(self, item_texts: List[str]) -> np.ndarray:
        emb = self.encoder.encode(item_texts, convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        
        sim = emb @ self.prototypes.T
        max_s = np.max(sim, axis=1)
        sorted_s = np.sort(sim, axis=1)[:, ::-1]
        margin = sorted_s[:, 0] - sorted_s[:, 1] if self.n_real_dims > 1 else sorted_s[:, 0]
        
        cal = sim * self.weights + self.biases
        exp_s = np.exp(cal / self.temperature - np.max(cal / self.temperature, axis=1, keepdims=True))
        dim_p = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-10)
        p_m1 = self._compute_p_minus1(max_s, margin, self.tau_max, self.tau_margin, self.p_minus1_base)
        
        proba = np.zeros((len(item_texts), len(self.dimensions)))
        if self.minus1_idx is not None:
            proba[:, self.minus1_idx] = p_m1
            for j, idx in enumerate(self.real_dim_indices):
                proba[:, idx] = dim_p[:, j] * (1 - p_m1)
        else:
            for j, idx in enumerate(self.real_dim_indices):
                proba[:, idx] = dim_p[:, j]
        return proba
    
    def save(self, path: str):
        encoder_path = str(Path(path).parent / f"{Path(path).stem}_encoder")
        self.encoder.save(encoder_path)
        state = {
            'prototypes': self.prototypes, 'weights': self.weights, 'biases': self.biases,
            'tau_max': self.tau_max, 'tau_margin': self.tau_margin, 'p_minus1_base': self.p_minus1_base,
            'temperature': self.temperature, 'dimensions': self.dimensions,
            'dimension_labels': self.dimension_labels, 'real_dim_indices': self.real_dim_indices,
            'real_dim_labels': self.real_dim_labels, 'n_real_dims': self.n_real_dims,
            'minus1_idx': self.minus1_idx, 'model_name': self.model_name, 'encoder_path': encoder_path
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"    Saved model to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'ContrastiveModel':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        model = cls(model_name=state['model_name'], device=device)
        for k, v in state.items():
            if k not in ['encoder_path', 'model_name']:
                setattr(model, k, v)
        # Derive encoder path from pkl location (encoder dir is always a sibling)
        encoder_path = str(Path(path).parent / f"{Path(path).stem}_encoder")
        model.encoder = SentenceTransformer(encoder_path, device=device)
        model.fitted = True
        return model


# =============================================================================
# Evaluation & Cross-Validation
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kappa = cohen_kappa_score(y_true, y_pred)
    except:
        kappa = 0.0
    
    # Accuracy excluding -1
    mask = y_true != -1
    acc_no_minus1 = accuracy_score(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 0.0
    
    return {
        'accuracy': acc,
        'accuracy_no_minus1': acc_no_minus1,
        'f1_macro': f1,
        'kappa': kappa if not np.isnan(kappa) else 0.0,
        'n_samples': len(y_true),
        'n_correct': int((y_true == y_pred).sum())
    }


def run_logo_cv(
    df: pd.DataFrame,
    approach: str,
    model_name: str,
    device: str,
    contrastive_epochs: int = 5,
    learning_rate: float = 2e-5,
    margin: float = 0.3
) -> List[Dict]:
    """Run leave-one-questionnaire-out cross-validation."""
    questionnaires = df['questionnaire'].unique()
    print(f"\n  Running LOGO CV with {len(questionnaires)} folds")
    
    all_results = []
    
    for test_q in questionnaires:
        print(f"\n  === Fold: {test_q} (held out) ===")
        
        train_df = df[df['questionnaire'] != test_q]
        test_df = df[df['questionnaire'] == test_q]
        
        if len(test_df) < 1 or len(train_df) < 5:
            print(f"    Skipped (insufficient data)")
            continue
        
        # Prepare training data
        texts_train, y_hard_train, y_soft_train, dimensions, dim_labels, _ = prepare_training_data(train_df)
        texts_test, y_hard_test, _, _, _, _ = prepare_training_data(test_df)
        
        print(f"    Train: {len(texts_train)} items, Test: {len(texts_test)} items")
        
        # Train model
        if approach == 'prototype':
            model = PrototypeModel(model_name=model_name, device=device)
        else:
            model = ContrastiveModel(model_name=model_name, device=device, epochs=contrastive_epochs,
                                     learning_rate=learning_rate, margin=margin)
        
        model.fit(texts_train, y_soft_train, dimensions, dim_labels)
        
        # Evaluate
        y_pred = model.predict(texts_test)
        metrics = evaluate(y_hard_test, y_pred)
        
        print(f"\n    Results: Acc={metrics['accuracy']:.3f}, Acc(no -1)={metrics['accuracy_no_minus1']:.3f}, F1={metrics['f1_macro']:.3f}")
        
        all_results.append({
            'fold': test_q,
            'n_train': len(train_df),
            'n_test': len(test_df),
            **metrics
        })
    
    # Summary
    if all_results:
        avg_acc = np.mean([r['accuracy'] for r in all_results])
        avg_acc_no_m1 = np.mean([r['accuracy_no_minus1'] for r in all_results])
        avg_f1 = np.mean([r['f1_macro'] for r in all_results])
        print(f"\n  ══════════════════════════════════════")
        print(f"  AVERAGE: Acc={avg_acc:.3f}, Acc(no -1)={avg_acc_no_m1:.3f}, F1={avg_f1:.3f}")
        print(f"  ══════════════════════════════════════")
    
    return all_results


# =============================================================================
# Main
# =============================================================================

ALL_CONSTRUCTS = ['depression', 'anxiety', 'psychosis', 'apathy', 'sleep', 'impulse_control']


def run_single_construct(
    df: pd.DataFrame,
    construct: str,
    approach: str,
    model_name: str,
    device: str,
    output_dir: Path,
    run_cv: bool,
    epochs: int,
    learning_rate: float = 2e-5,
    margin: float = 0.3
):
    """Run training for a single construct."""
    print(f"\n{'#' * 70}")
    print(f"# CONSTRUCT: {construct.upper()}")
    print(f"{'#' * 70}")
    
    # Get dimension info
    item_texts, y_hard, y_soft, dimensions, dim_labels, _ = prepare_training_data(df)
    print(f"\nLoaded {len(df)} items from {len(df['questionnaire'].unique())} questionnaires")
    print(f"Dimensions: {[f'{d}={l[:30]}' for d, l in zip(dimensions, dim_labels)]}")
    
    construct_output_dir = output_dir / construct
    construct_output_dir.mkdir(parents=True, exist_ok=True)
    
    approaches = ['prototype', 'contrastive'] if approach == 'both' else [approach]
    
    for app in approaches:
        print(f"\n{'=' * 70}")
        print(f"APPROACH: {app.upper()}")
        print("=" * 70)
        
        if run_cv:
            results = run_logo_cv(df, app, model_name, device, epochs, learning_rate, margin)
            
            # Save CV results
            results_df = pd.DataFrame(results)
            results_path = construct_output_dir / f'{app}_cv_results.csv'
            results_df.to_csv(results_path, index=False)
            print(f"\n  CV results saved to {results_path}")
        
        # Train final model on all data
        print(f"\n  Training final model on ALL data...")
        
        if app == 'prototype':
            model = PrototypeModel(model_name=model_name, device=device)
        else:
            model = ContrastiveModel(model_name=model_name, device=device, epochs=epochs,
                                     learning_rate=learning_rate, margin=margin)
        
        model.fit(item_texts, y_soft, dimensions, dim_labels)
        
        # Save model
        model_path = construct_output_dir / f'{app}_model.pkl'
        model.save(str(model_path))
        
        # Also save training performance
        y_pred = model.predict(item_texts)
        train_metrics = evaluate(y_hard, y_pred)
        print(f"\n  Training set performance: Acc={train_metrics['accuracy']:.3f}, F1={train_metrics['f1_macro']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic learning for ENIGMA-PD harmonizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single construct with CV
  python semantic_learning.py --data finetuning_dataset.csv --construct depression --approach prototype --cv
  
  # All constructs
  python semantic_learning.py --data finetuning_dataset.csv --construct all --approach prototype --cv
  
  # Contrastive model (GPU recommended)
  python semantic_learning.py --data finetuning_dataset.csv --construct anxiety --approach contrastive --device cuda
  
  # Both approaches on all constructs
  python semantic_learning.py --data finetuning_dataset.csv --construct all --approach both --cv
        """
    )
    parser.add_argument('--data', '-d', required=True, help='Path to finetuning_dataset.csv')
    parser.add_argument('--construct', '-c', required=True, 
                        choices=ALL_CONSTRUCTS + ['all'],
                        help='Construct to train on, or "all" for all constructs')
    parser.add_argument('--approach', '-a', choices=['prototype', 'contrastive', 'both'], default='prototype')
    parser.add_argument('--output', '-o', default='./models', help='Output directory for models')
    parser.add_argument('--model-name', default='hkunlp/instructor-large', help='Sentence transformer model')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--cv', action='store_true', help='Run cross-validation')
    parser.add_argument('--epochs', type=int, default=5, help='Contrastive training epochs')
    parser.add_argument('--learning-rate', '--lr', type=float, default=2e-5, help='Contrastive learning rate (default: 2e-5)')
    parser.add_argument('--margin', type=float, default=0.3, help='Contrastive margin (default: 0.3)')
    
    args = parser.parse_args()
    
    # Auto-detect GPU
    if args.device == 'cuda' and TORCH_AVAILABLE and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Determine which constructs to run
    if args.construct == 'all':
        constructs = ALL_CONSTRUCTS
    else:
        constructs = [args.construct]
    
    print("=" * 70)
    print(f"SEMANTIC LEARNING")
    print(f"Constructs: {', '.join(constructs)}")
    print(f"Approach: {args.approach}")
    print(f"Device: {args.device}")
    if args.approach in ('contrastive', 'both'):
        print(f"Epochs: {args.epochs}, LR: {args.learning_rate}, Margin: {args.margin}")
    print("=" * 70)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load full dataset once
    full_df = pd.read_csv(args.data)
    full_df['e_ik_dict'] = full_df['expert_proportions_e_ik'].apply(parse_e_ik)
    
    for construct in constructs:
        df = full_df[full_df['construct'] == construct].copy()
        
        if len(df) == 0:
            print(f"\nWARNING: No data found for construct '{construct}', skipping...")
            continue
        
        run_single_construct(
            df=df,
            construct=construct,
            approach=args.approach,
            model_name=args.model_name,
            device=args.device,
            output_dir=output_dir,
            run_cv=args.cv,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            margin=args.margin
        )


if __name__ == "__main__":
    main()
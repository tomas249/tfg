# common/models.py
"""
Neural network models for neuron classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from typing import Dict

class GNNModel(nn.Module):
    """Graph Neural Network for multi-task neuron classification."""
    
    def __init__(self, in_feats_dim: int, hidden_dim: int, embed_dim_node: int, embed_dim_edge: int,
                 num_in_np: int, num_out_np: int, num_edge_np: int, num_edge_nt: int,
                 num_super_classes: int, num_nt_classes: int, num_tag_classes: int, 
                 num_primary_classes: int, dropout: float = 0.1):
        super().__init__()
        
        # Embeddings
        self.in_np_emb = nn.Embedding(num_in_np, embed_dim_node)
        self.out_np_emb = nn.Embedding(num_out_np, embed_dim_node)
        self.edge_np_emb = nn.Embedding(num_edge_np, embed_dim_edge)
        self.edge_nt_emb = nn.Embedding(num_edge_nt, embed_dim_edge)
        
        # Graph convolution
        edge_feat_dim = embed_dim_edge + embed_dim_edge + 1
        node_feat_dim = in_feats_dim + 2 * embed_dim_node
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, node_feat_dim * hidden_dim)
        )
        self.conv = NNConv(node_feat_dim, hidden_dim, self.edge_mlp, aggr='mean')
        self.dropout = dropout
        
        # Output heads
        self.head_super = nn.Linear(hidden_dim, num_super_classes)
        self.head_nt = nn.Linear(hidden_dim, num_nt_classes)  
        self.head_tags = nn.Linear(hidden_dim, num_tag_classes)
        self.head_primary = nn.Linear(hidden_dim, num_primary_classes)
    
    def forward(self, batch):
        # Node features
        x_num = batch.x
        x_in = self.in_np_emb(batch.input_np)
        x_out = self.out_np_emb(batch.output_np)
        x = torch.cat([x_num, x_in, x_out], dim=1)
        
        # Edge features
        e_nt = self.edge_nt_emb(batch.edge_nt)
        e_np = self.edge_np_emb(batch.edge_np)
        e_sc = batch.edge_sc.view(-1, 1)
        edge_attr = torch.cat([e_nt, e_np, e_sc], dim=1)
        
        # Graph convolution
        x = F.relu(self.conv(x, batch.edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return (self.head_super(x), self.head_nt(x), self.head_tags(x), self.head_primary(x))


class NeuronInferenceModel(torch.nn.Module):
    """Inference wrapper for the trained GNN model."""
    
    def __init__(self, model: GNNModel, scaler_cont: StandardScaler, scaler_counts: StandardScaler,
                 category_mappings: Dict, reverse_mappings: Dict, tag_mlb: MultiLabelBinarizer, device: str = 'cpu'):
        super().__init__()
        self.model = model.to(device)
        self.scaler_cont = scaler_cont
        self.scaler_counts = scaler_counts
        self.tag_mlb = tag_mlb
        self.device = device
        
        # Fix mappings
        self.category_mappings = {cat: {int(k): v for k, v in mapping.items()} 
                                for cat, mapping in category_mappings.items()}
        self.reverse_mappings = {cat: {str(k): int(v) for k, v in mapping.items()}
                               for cat, mapping in reverse_mappings.items()}
        
        # Feature columns
        self.continuous_cols = ["centroid_x", "centroid_y", "centroid_z", "length_nm", "size_nm", "area_nm"]
        self.count_cols = ["input_synapses_count", "input_partners_count", "output_synapses_count", "output_partners_count"]
        self.onehot_cols = ['side_center', 'side_left', 'side_right', 'flow_afferent', 'flow_efferent', 'flow_intrinsic']
        self.cat_cols = ["nt_type", "super_class", "primary_type", "input_neuropil", "output_neuropil"]
    
    def predict(self, inputs):
        """Make predictions on input data."""
        if isinstance(inputs, dict):
            inputs = [inputs]
        df = pd.DataFrame(inputs).reset_index(drop=True)
        
        # Fill missing columns with defaults
        for col in self.continuous_cols:
            if col not in df:
                df[col] = float(self.scaler_cont.mean_[self.continuous_cols.index(col)])
        for col in self.count_cols + self.onehot_cols:
            if col not in df:
                df[col] = 0
        for col in self.cat_cols:
            if col not in df:
                df[col] = "Unknown"
        
        # Type conversions
        df[self.continuous_cols] = df[self.continuous_cols].astype(float)
        df[self.count_cols] = df[self.count_cols].astype(float)
        df[self.onehot_cols] = df[self.onehot_cols].astype(int)
        df[self.cat_cols] = df[self.cat_cols].astype(str)
        
        # Encode categories
        for cat in self.cat_cols:
            mapping = self.reverse_mappings.get(cat, {})
            df[f"{cat}_code"] = df[cat].map(lambda x: mapping.get(str(x), 0)).astype(int)
        
        # Scale features
        df[self.continuous_cols] = self.scaler_cont.transform(df[self.continuous_cols])
        df[self.count_cols] = self.scaler_counts.transform(np.log1p(df[self.count_cols]))
        
        # Create tensors
        feat_cols = self.continuous_cols + self.count_cols + self.onehot_cols
        X_num = torch.tensor(df[feat_cols].values, dtype=torch.float32, device=self.device)
        input_np = torch.tensor(df["input_neuropil_code"].values, dtype=torch.long, device=self.device)
        output_np = torch.tensor(df["output_neuropil_code"].values, dtype=torch.long, device=self.device)
        
        # Create dummy batch for isolated inference
        class Batch: pass
        batch = Batch()
        batch.x = X_num
        batch.input_np = input_np
        batch.output_np = output_np
        batch.edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        batch.edge_sc = torch.zeros((0, 1), dtype=torch.float32, device=self.device)
        batch.edge_np = torch.zeros((0,), dtype=torch.long, device=self.device)
        batch.edge_nt = torch.zeros((0,), dtype=torch.long, device=self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            out_super, out_nt, out_tags, out_primary = self.model(batch)
        
        # Convert predictions to labels
        pred_super = out_super.argmax(dim=1).cpu().numpy()
        pred_nt = out_nt.argmax(dim=1).cpu().numpy()
        pred_tags_binary = (torch.sigmoid(out_tags) > 0.5).cpu().numpy()
        pred_primary = out_primary.argmax(dim=1).cpu().numpy()
        
        # Format tag predictions
        tag_predictions = []
        for tag_vector in pred_tags_binary:
            active_tags = [self.tag_mlb.classes_[j] for j, is_active in enumerate(tag_vector) if is_active]
            tag_predictions.append(','.join(active_tags) if active_tags else 'no_tags')
        
        # Add predictions to dataframe
        df_out = df.copy()
        df_out["pred_super_class"] = [self.category_mappings["super_class"].get(int(c), f"Unknown_{c}") for c in pred_super]
        df_out["pred_nt_type"] = [self.category_mappings["nt_type"].get(int(c), f"Unknown_{c}") for c in pred_nt]
        df_out["pred_connectivity_tag"] = tag_predictions
        df_out["pred_primary_type"] = [self.category_mappings["primary_type"].get(int(c), f"Unknown_{c}") for c in pred_primary]
        
        return df_out
    
    def save(self, path: str):
        """Save the inference model."""
        torch.save(self, path)
    
    @staticmethod
    def load(path: str, device: str = 'cpu'):
        """Load a saved inference model."""
        model = torch.load(path, map_location=device, weights_only=False)
        model.device = device
        model.model = model.model.to(device)
        return model

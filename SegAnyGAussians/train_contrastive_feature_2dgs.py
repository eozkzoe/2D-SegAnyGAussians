import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
import os
from utils.general_utils import safe_state
from argparse import ArgumentParser
import json
from tqdm import tqdm

class SurfelFeatureNet(nn.Module):
    def __init__(self, input_dim=7, feature_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

class SurfelDataset(Dataset):
    def __init__(self, scene_path, iteration=None):
        # Load scales and parameters
        if iteration is None:
            checkpoints = [os.path.splitext(f)[0] for f in os.listdir(scene_path) if f.endswith('.ply')]
            iterations = [int(c.split('_')[-1]) for c in checkpoints]
            iteration = max(iterations) if iterations else 7000
            
        scale_file = os.path.join(scene_path, f'scales_{iteration}.json')
        with open(scale_file, 'r') as f:
            scales = json.load(f)
            
        # Convert to tensors
        self.positions = torch.tensor([s['position'] for s in scales], dtype=torch.float32)
        self.scales = torch.tensor([s['scale'] for s in scales], dtype=torch.float32).unsqueeze(1)
        self.normals = torch.tensor([s['normal'] for s in scales], dtype=torch.float32)
        self.opacities = torch.tensor([s['opacity'] for s in scales], dtype=torch.float32).unsqueeze(1)
        
        # Combine features
        self.features = torch.cat([
            self.positions,
            self.scales,
            self.normals,
            self.opacities
        ], dim=1)
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.features[idx]

def train_features(scene_path, output_path, iteration=None, feature_dim=32, batch_size=1024, epochs=100):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and dataloader
    dataset = SurfelDataset(scene_path, iteration)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = SurfelFeatureNet(input_dim=7, feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            
            # Get features
            features = model(batch)
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(features, features.t())
            
            # Compute positive pairs (nearby points)
            pos_mask = torch.zeros_like(sim_matrix)
            for i in range(len(batch)):
                # Find nearby points based on position and normal similarity
                pos = batch[i, :3]
                normal = batch[i, 3:6]
                
                dist = torch.norm(batch[:, :3] - pos.unsqueeze(0), dim=1)
                normal_sim = torch.abs(torch.sum(batch[:, 3:6] * normal.unsqueeze(0), dim=1))
                
                # Points are positive pairs if they are close and have similar normals
                pos_pairs = (dist < 0.1) & (normal_sim > 0.9)
                pos_mask[i] = pos_pairs.float()
            
            # Remove self-similarity
            pos_mask.fill_diagonal_(0)
            
            # Compute InfoNCE loss
            temperature = 0.07
            sim_matrix = sim_matrix / temperature
            
            # For numerical stability
            sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0]
            
            exp_sim = torch.exp(sim_matrix)
            log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
            
            mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
            loss = -mean_log_prob_pos.mean()
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        pbar.set_description(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Save features
    model.eval()
    with torch.no_grad():
        all_features = []
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            features = model(batch.to(device))
            all_features.append(features.cpu())
        all_features = torch.cat(all_features, dim=0)
    
    # Save features and model
    torch.save({
        'features': all_features,
        'model_state_dict': model.state_dict()
    }, output_path)
    
    print(f"Saved features and model to {output_path}")
    return all_features

if __name__ == "__main__":
    parser = ArgumentParser(description="Train contrastive features for 2D Gaussian surfels")
    parser.add_argument("--scene_path", required=True, help="Path to the scene directory")
    parser.add_argument("--output_path", required=True, help="Path to save features and model")
    parser.add_argument("--iteration", type=int, help="Iteration to load (default: latest)")
    parser.add_argument("--feature_dim", type=int, default=32, help="Dimension of feature vectors")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()
    
    train_features(
        args.scene_path,
        args.output_path,
        args.iteration,
        args.feature_dim,
        args.batch_size,
        args.epochs
    ) 
import torch
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

class PoseEstimator:
    def __init__(self, gaussians, mask):
        self.gaussians = gaussians
        self.mask = mask
        
    def estimate_pose(self):
        """Estimate pose (position and normal) of segmented Gaussians"""
        # Get positions of selected Gaussians
        xyz = self.gaussians._xyz[self.mask]
        
        # Convert to numpy for calculations
        points = xyz.detach().cpu().numpy()
        
        # Calculate centroid (position)
        centroid = np.mean(points, axis=0)
        
        # Fit PCA to estimate principal directions
        pca = PCA(n_components=3)
        pca.fit(points - centroid)
        
        # Normal is the least significant principal component
        normal = pca.components_[2]
        
        # Ensure normal points outward from the centroid
        points_centered = points - centroid
        if np.mean(np.sum(points_centered * normal, axis=1)) < 0:
            normal = -normal
            
        # Calculate confidence based on eigenvalues
        planarity = 1 - pca.explained_variance_ratio_[2]
        
        return {
            'centroid': centroid,
            'normal': normal,
            'planarity': planarity,
            'points': points
        }
        
    def get_oriented_bbox(self):
        """Get oriented bounding box of segmented Gaussians"""
        pose = self.estimate_pose()
        points = pose['points']
        
        # Project points onto principal components
        pca = PCA(n_components=3)
        projected = pca.fit_transform(points - pose['centroid'])
        
        # Get min/max along principal axes
        min_proj = np.min(projected, axis=0)
        max_proj = np.max(projected, axis=0)
        
        # Calculate dimensions
        dimensions = max_proj - min_proj
        
        return {
            'center': pose['centroid'],
            'axes': pca.components_,
            'dimensions': dimensions
        }
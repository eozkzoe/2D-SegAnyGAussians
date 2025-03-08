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
        xyz = self.gaussians._xyz[self.mask]
        points = xyz.detach().cpu().numpy()
        centroid = np.mean(points, axis=0)
        pca = PCA(n_components=3)
        pca.fit(points - centroid)
        normal = pca.components_[2]
        points_centered = points - centroid
        if np.mean(np.sum(points_centered * normal, axis=1)) < 0:
            normal = -normal
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
        
        pca = PCA(n_components=3)
        projected = pca.fit_transform(points - pose['centroid'])
        
        min_proj = np.min(projected, axis=0)
        max_proj = np.max(projected, axis=0)
        
        dimensions = max_proj - min_proj
        
        return {
            'center': pose['centroid'],
            'axes': pca.components_,
            'dimensions': dimensions
        }
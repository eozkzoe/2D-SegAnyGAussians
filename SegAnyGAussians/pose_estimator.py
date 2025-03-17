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
        # Get positions
        xyz = self.gaussians._xyz[self.mask]
        points = xyz.detach().cpu().numpy()
        centroid = np.mean(points, axis=0)
        
        # Get normals directly from rotation matrices
        rotations = self.gaussians.get_rotation[self.mask].detach().cpu().numpy()
        # For 2D Gaussians, the rotation matrix directly gives us the normal
        normals = rotations
        mean_normal = np.mean(normals, axis=0)
        mean_normal = mean_normal / np.linalg.norm(mean_normal)  # Normalize
        
        # Get opacity weights for weighted calculations
        opacities = self.gaussians.get_opacity[self.mask].detach().cpu().numpy()
        weights = opacities / opacities.sum()
        
        # Calculate weighted centroid and normal
        weighted_centroid = np.sum(points * weights[:, None], axis=0)
        weighted_normal = np.sum(normals * weights[:, None], axis=0)
        weighted_normal = weighted_normal / np.linalg.norm(weighted_normal)
        
        # Calculate planarity using PCA for completeness
        pca = PCA(n_components=3)
        pca.fit(points - centroid)
        planarity = 1 - pca.explained_variance_ratio_[2]
        
        return {
            'centroid': weighted_centroid,
            'normal': weighted_normal,
            'unweighted_normal': mean_normal,
            'planarity': planarity,
            'points': points,
            'normals': normals
        }
        
    def get_oriented_bbox(self):
        """Get oriented bounding box of segmented Gaussians"""
        pose = self.estimate_pose()
        points = pose['points']
        
        # Use normal as primary axis
        normal = pose['normal']
        # Find perpendicular vectors to form coordinate system
        v1 = np.array([1, 0, 0]) if abs(normal[1]) > abs(normal[0]) else np.array([0, 1, 0])
        axis1 = np.cross(normal, v1)
        axis1 = axis1 / np.linalg.norm(axis1)
        axis2 = np.cross(normal, axis1)
        
        # Create transformation matrix
        axes = np.vstack([axis1, axis2, normal])
        
        # Project points onto new coordinate system
        projected = (points - pose['centroid']) @ axes.T
        
        min_proj = np.min(projected, axis=0)
        max_proj = np.max(projected, axis=0)
        
        dimensions = max_proj - min_proj
        print(f"center: {pose['centroid']}, 'axes': {axes}, 'dims': {dimensions}")
        
        return {
            'center': pose['centroid'],
            'axes': axes,
            'dimensions': dimensions
        }
        
    def get_mean_normal(self):
        """Get the mean normal direction of segmented Gaussians, weighted by opacity"""
        # Get normals directly from rotation matrices
        normals = self.gaussians.get_rotation[self.mask].detach().cpu().numpy()
        opacities = self.gaussians.get_opacity[self.mask].detach().cpu().numpy()
        
        # Weight normals by opacity
        weights = opacities / opacities.sum()
        weighted_normal = np.sum(normals * weights[:, None], axis=0)
        weighted_normal = weighted_normal / np.linalg.norm(weighted_normal)
        
        return weighted_normal
import os
import torch
import numpy as np
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from scipy.spatial import KDTree
from sklearn.decomposition import PCA


class GaussianSegmenter:
    def __init__(
        self,
        scene_path,
        model_path,
        mask_path,
        iteration=15000,
        output_dir="./segmented_gaussians",
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.mask_path = mask_path

        # Load the scene and model
        class DummyArgs:
            def __init__(self):
                self.source_path = scene_path
                self.model_path = model_path
                self.images = "images"
                self.eval = False
                self.sh_degree = 3
                self.white_background = False
                self.feature_dim = 256
                self.load_iteration = -1
                self.allow_principle_point_shift = False
                self.need_features = False
                self.need_masks = False
                self.resolution = 1
                self.data_device = "cuda"

        args = DummyArgs()
        self.gaussian_model = GaussianModel(3)
        self.gaussian_model.load_ply(
            os.path.join(
                model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"
            )
        )

        # Load and process the mask
        self.mask = torch.load(mask_path)
        if torch.count_nonzero(self.mask) == 0:
            print("Mask is empty, inverting mask")
            self.mask = ~self.mask

        # Ensure mask matches Gaussian count
        n_gaussians = len(self.gaussian_model.get_xyz)
        if len(self.mask.flatten()) != n_gaussians:
            print(f"Resizing mask from {len(self.mask.flatten())} to {n_gaussians}")
            flat_mask = self.mask.flatten()
            if len(flat_mask) > n_gaussians:
                self.mask = flat_mask[:n_gaussians]
            else:
                self.mask = torch.cat(
                    [
                        flat_mask,
                        torch.zeros(n_gaussians - len(flat_mask), dtype=torch.bool),
                    ]
                )
            self.mask = self.mask.to(device="cuda")

    def compute_normal_from_neighbors(self, selected_idx):
        """Compute normal from selected Gaussian and its 4 nearest neighbors"""
        xyz = self.gaussian_model.get_xyz.detach().cpu().numpy()
        
        # Build KD-tree for nearest neighbor search
        tree = KDTree(xyz)
        
        # Find 5 nearest neighbors (including the point itself)
        distances, indices = tree.query(xyz[selected_idx:selected_idx+1], k=5)
        
        # Get the points forming the plane
        plane_points = xyz[indices[0]]
        
        # Use PCA to find the normal (third principal component)
        pca = PCA(n_components=3)
        pca.fit(plane_points)
        normal = pca.components_[2]
        
        # Ensure normal points outward (assuming center is at origin)
        center_to_point = xyz[selected_idx] - np.mean(xyz, axis=0)
        if np.dot(normal, center_to_point) < 0:
            normal = -normal
            
        return normal

    def segment_and_save(self):
        """Segment the Gaussians using the mask and save as a new PLY"""
        # Apply segmentation
        self.gaussian_model.segment(self.mask)

        # Get mask filename without extension and path
        mask_filename = os.path.splitext(os.path.basename(self.mask_path))[0]
        output_filename = f"{mask_filename}_gaussians.ply"
        output_path = os.path.join(self.output_dir, output_filename)

        # Save the segmented model
        self.gaussian_model.save_ply(output_path)
        print(f"Segmented Gaussians saved to: {output_path}")

        # Clear segmentation to restore original model
        self.gaussian_model.clear_segment()

        return output_path

    def process_selected_gaussian(self, selected_idx):
        """Process a selected Gaussian and print its normal"""
        normal = self.compute_normal_from_neighbors(selected_idx)
        print(f"\nSelected Gaussian {selected_idx}:")
        print(f"Position: {self.gaussian_model.get_xyz[selected_idx].detach().cpu().numpy()}")
        print(f"Computed normal: {normal}")
        print(f"Normal magnitude: {np.linalg.norm(normal)}")
        return normal


def main():
    parser = ArgumentParser(
        description="Segment Gaussians using a mask and save as PLY"
    )
    parser.add_argument(
        "--scene", type=str, required=True, help="Path to the COLMAP scene directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the pre-trained 3DGS model directory",
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to the segmentation mask (.pt file)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=9000,
        help="Iteration number to load (default: 9000)",
    )
    parser.add_argument(
        "--output", type=str, default="./segmented_gaussians", help="Output directory"
    )
    parser.add_argument("--selected_idx", type=int, help="Index of Gaussian to analyze normal")
    
    args = parser.parse_args()
    
    segmenter = GaussianSegmenter(
        scene_path=args.scene,
        model_path=args.model,
        mask_path=args.mask,
        iteration=args.iteration,
        output_dir=args.output,
    )
    
    output_path = segmenter.segment_and_save()
    print(f"Segmentation complete. Results saved to: {output_path}")
    
    if args.selected_idx is not None:
        segmenter.process_selected_gaussian(args.selected_idx)


if __name__ == "__main__":
    main()

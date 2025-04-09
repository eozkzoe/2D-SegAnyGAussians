import numpy as np
import cv2
import torch
from stereo_vision import StereoVision
from pathlib import Path
from tqdm import tqdm


class StereoVisionTester:
    def __init__(self, test_data_dir="test_data"):
        self.test_data_dir = Path(test_data_dir)
        self.stereo = StereoVision()

    def load_image_pair(self, left_path, right_path):
        """Load and preprocess stereo image pair."""
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))

        if left_img is None or right_img is None:
            raise ValueError(f"Could not load images: {left_path}, {right_path}")

        # Convert to RGB
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        return left_img, right_img

    def load_ground_truth(self, depth_path, normal_path=None):
        """Load ground truth depth and normal maps if available."""
        depth_gt = None
        normal_gt = None

        if depth_path.exists():
            depth_gt = np.load(str(depth_path))

        if normal_path is not None and normal_path.exists():
            normal_gt = np.load(str(normal_path))

        return depth_gt, normal_gt

    def compute_depth_metrics(self, pred_depth, gt_depth, mask=None):
        """Compute depth estimation metrics."""
        if mask is None:
            mask = gt_depth > 0

        # Apply mask
        pred = pred_depth[mask]
        gt = gt_depth[mask]

        # Scale invariant metrics
        scale = np.median(gt) / np.median(pred)
        pred_scaled = pred * scale

        # Compute metrics
        abs_rel = np.mean(np.abs(pred_scaled - gt) / gt)
        sq_rel = np.mean(((pred_scaled - gt) ** 2) / gt)
        rmse = np.sqrt(np.mean((pred_scaled - gt) ** 2))
        rmse_log = np.sqrt(np.mean((np.log(pred_scaled) - np.log(gt)) ** 2))

        # Thresholded accuracy metrics
        thresh = np.maximum((gt / pred_scaled), (pred_scaled / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25**2).mean()
        a3 = (thresh < 1.25**3).mean()

        return {
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "rmse": rmse,
            "rmse_log": rmse_log,
            "a1": a1,
            "a2": a2,
            "a3": a3,
        }

    def compute_normal_metrics(self, pred_normal, gt_normal, mask=None):
        """Compute normal estimation metrics."""
        if mask is None:
            mask = np.all(gt_normal != 0, axis=-1)

        # Apply mask
        pred = pred_normal[mask]
        gt = gt_normal[mask]

        # Normalize vectors
        pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-6)
        gt_norm = gt / (np.linalg.norm(gt, axis=-1, keepdims=True) + 1e-6)

        # Compute angular error
        dot_product = np.sum(pred_norm * gt_norm, axis=-1)
        dot_product = np.clip(dot_product, -1, 1)
        angular_error = np.arccos(dot_product) * 180 / np.pi

        # Compute metrics
        mean_angular_error = np.mean(angular_error)
        median_angular_error = np.median(angular_error)
        rmse = np.sqrt(np.mean(angular_error**2))

        # Thresholded accuracy
        a1 = (angular_error < 5).mean()
        a2 = (angular_error < 10).mean()
        a3 = (angular_error < 20).mean()

        return {
            "mean": mean_angular_error,
            "median": median_angular_error,
            "rmse": rmse,
            "a1": a1,
            "a2": a2,
            "a3": a3,
        }

    def test_single_pair(
        self, left_path, right_path, depth_gt_path=None, normal_gt_path=None
    ):
        """Test stereo vision on a single image pair."""
        # Load images
        left_img, right_img = self.load_image_pair(left_path, right_path)

        # Load ground truth if available
        depth_gt, normal_gt = self.load_ground_truth(depth_gt_path, normal_gt_path)

        # Estimate depth and normals
        depth_pred, normal_pred = self.stereo.estimate_depth_and_normals(
            left_img, right_img
        )

        if depth_pred is None:
            return None, None

        # Compute metrics if ground truth is available
        depth_metrics = None
        normal_metrics = None

        if depth_gt is not None:
            depth_metrics = self.compute_depth_metrics(depth_pred, depth_gt)

        if normal_gt is not None:
            normal_metrics = self.compute_normal_metrics(normal_pred, normal_gt)

        return depth_metrics, normal_metrics

    def run_test_suite(self):
        """Run tests on all image pairs in test directory."""
        # Find all image pairs
        left_images = sorted(self.test_data_dir.glob("*_left.png"))

        results = []
        for left_path in tqdm(left_images, desc="Testing stereo pairs"):
            # Construct paths
            right_path = left_path.parent / left_path.name.replace(
                "_left.png", "_right.png"
            )
            depth_gt_path = left_path.parent / left_path.name.replace(
                "_left.png", "_depth.npy"
            )
            normal_gt_path = left_path.parent / left_path.name.replace(
                "_left.png", "_normal.npy"
            )

            # Run test
            depth_metrics, normal_metrics = self.test_single_pair(
                left_path, right_path, depth_gt_path, normal_gt_path
            )

            if depth_metrics is not None or normal_metrics is not None:
                results.append(
                    {
                        "pair_name": left_path.stem,
                        "depth_metrics": depth_metrics,
                        "normal_metrics": normal_metrics,
                    }
                )

        return results


def main():
    # Create test directory if it doesn't exist
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)

    # Initialize tester
    tester = StereoVisionTester(test_data_dir)

    # Run test suite
    results = tester.run_test_suite()

    # Print results
    if results:
        print("\nTest Results:")
        for result in results:
            print(f"\nImage Pair: {result['pair_name']}")

            if result["depth_metrics"]:
                print("\nDepth Metrics:")
                for metric, value in result["depth_metrics"].items():
                    print(f"{metric}: {value:.4f}")

            if result["normal_metrics"]:
                print("\nNormal Metrics:")
                for metric, value in result["normal_metrics"].items():
                    print(f"{metric}: {value:.4f}")
    else:
        print("\nNo test results available. Please ensure test data is present.")


if __name__ == "__main__":
    main()

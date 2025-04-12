import numpy as np
import cv2
from PIL import Image
import torch
import json


def load_normal_map(path):
    """Load a normal map image and convert it to the correct format."""
    # Load image and convert to float32 numpy array in range [0, 1]
    img = np.array(Image.open(path)).astype(np.float32) / 255.0

    # Convert from [0,1] to [-1,1] range which is typical for normal maps
    img = img * 2.0 - 1.0
    return img


def compute_mse(normal_map1, normal_map2):
    """Compute Mean Squared Error between two normal maps."""
    if isinstance(normal_map1, torch.Tensor):
        normal_map1 = normal_map1.cpu().numpy()
    if isinstance(normal_map2, torch.Tensor):
        normal_map2 = normal_map2.cpu().numpy()

    return np.mean((normal_map1 - normal_map2) ** 2)


def compute_cosine_similarity(normal_map1, normal_map2):
    """Compute average cosine similarity between normal vectors."""
    if isinstance(normal_map1, torch.Tensor):
        normal_map1 = normal_map1.cpu().numpy()
    if isinstance(normal_map2, torch.Tensor):
        normal_map2 = normal_map2.cpu().numpy()

    # Normalize vectors
    norm1 = np.linalg.norm(normal_map1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(normal_map2, axis=-1, keepdims=True)

    normal_map1_normalized = normal_map1 / (norm1 + 1e-7)
    normal_map2_normalized = normal_map2 / (norm2 + 1e-7)

    # Compute dot product
    similarity = np.sum(normal_map1_normalized * normal_map2_normalized, axis=-1)

    # Average over all pixels
    return np.mean(similarity)


def compute_angular_error(normal_map1, normal_map2):
    """Compute average angular error in degrees between normal vectors."""
    if isinstance(normal_map1, torch.Tensor):
        normal_map1 = normal_map1.cpu().numpy()
    if isinstance(normal_map2, torch.Tensor):
        normal_map2 = normal_map2.cpu().numpy()

    # Normalize vectors
    norm1 = np.linalg.norm(normal_map1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(normal_map2, axis=-1, keepdims=True)

    normal_map1_normalized = normal_map1 / (norm1 + 1e-7)
    normal_map2_normalized = normal_map2 / (norm2 + 1e-7)

    # Compute dot product and clip to valid range [-1, 1]
    dot_product = np.clip(
        np.sum(normal_map1_normalized * normal_map2_normalized, axis=-1), -1.0, 1.0
    )

    # Convert to angles in degrees
    angles = np.arccos(dot_product) * 180.0 / np.pi

    return np.mean(angles)


def compute_psnr(normal_map1, normal_map2, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio between two normal maps."""
    mse = compute_mse(normal_map1, normal_map2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_val / np.sqrt(mse))


def compare_normal_maps(normal_map1_path, normal_map2_path):
    """Compare two normal maps using multiple metrics.

    Args:
        normal_map1_path: Path to the first normal map image
        normal_map2_path: Path to the second normal map image

    Returns:
        dict: Dictionary containing various comparison metrics
    """
    # Load normal maps
    normal_map1 = load_normal_map(normal_map1_path)
    normal_map2 = load_normal_map(normal_map2_path)

    # Ensure same shape
    if normal_map1.shape != normal_map2.shape:
        raise ValueError(
            f"Normal maps have different shapes: {normal_map1.shape} vs {normal_map2.shape}"
        )

    # Compute metrics
    metrics = {
        "mse": compute_mse(normal_map1, normal_map2),
        "cosine_similarity": compute_cosine_similarity(normal_map1, normal_map2),
        "angular_error": compute_angular_error(normal_map1, normal_map2),
        "psnr": compute_psnr(normal_map1, normal_map2),
    }

    return metrics


def load_labelme_mask(json_path, normal_map_shape=None):
    """Load a LabelMe annotation JSON file and create a binary mask.
    
    Args:
        json_path: Path to the LabelMe annotation JSON file
        normal_map_shape: Shape of the normal map to match (height, width)
    
    Returns:
        Binary mask array
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Get image dimensions from the JSON
    json_height = data["imageHeight"]
    json_width = data["imageWidth"]
    
    # Use normal map dimensions if provided, otherwise use JSON dimensions
    target_height = normal_map_shape[0] if normal_map_shape else json_height
    target_width = normal_map_shape[1] if normal_map_shape else json_width

    # Create empty mask with target dimensions
    mask = np.zeros((target_height, target_width), dtype=np.uint8)

    # Fill mask with annotation
    for shape in data["shapes"]:
        # Convert pixel coordinates to relative coordinates (0-1)
        points = np.array(shape["points"], dtype=np.float32)
        points[:, 0] = points[:, 0] / json_width  # normalize x coordinates
        points[:, 1] = points[:, 1] / json_height  # normalize y coordinates

        # Convert relative coordinates to target pixel coordinates
        points[:, 0] = points[:, 0] * target_width
        points[:, 1] = points[:, 1] * target_height
        points = points.astype(np.int32)

        cv2.fillPoly(mask, [points], 1)

    return mask


def compute_normal_consistency(normal_map, mask):
    """Compute metrics for normal vector consistency within a masked region.

    Args:
        normal_map: Normal map array of shape (H, W, 3)
        mask: Binary mask array of shape (H, W)

    Returns:
        dict: Dictionary containing consistency metrics
    """
    # Extract masked region
    masked_normals = normal_map[mask > 0]

    if len(masked_normals) == 0:
        return {
            "variance": float("inf"),
            "mean_angular_deviation": float("inf"),
            "planarity_score": 0.0,
        }

    # Compute mean normal vector
    mean_normal = np.mean(masked_normals, axis=0)
    mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-7)

    # Compute variance of normal vectors
    variance = np.mean(np.var(masked_normals, axis=0))

    # Compute angular deviation from mean normal
    dot_products = np.clip(np.sum(masked_normals * mean_normal, axis=1), -1.0, 1.0)
    angles = np.arccos(dot_products) * 180.0 / np.pi
    mean_angular_deviation = np.mean(angles)

    # Compute planarity score (1 - normalized variance)
    planarity_score = 1.0 - min(variance, 1.0)

    return {
        "variance": variance,
        "mean_angular_deviation": mean_angular_deviation,
        "planarity_score": planarity_score,
    }


def analyze_flat_surface(normal_map_path, mask_json_path):
    """Analyze the flatness quality of a surface in a normal map using a LabelMe mask.

    Args:
        normal_map_path: Path to the normal map image
        mask_json_path: Path to the LabelMe annotation JSON file

    Returns:
        dict: Dictionary containing flatness analysis metrics
    """
    # Load normal map and mask
    normal_map = load_normal_map(normal_map_path)
    mask = load_labelme_mask(mask_json_path, normal_map.shape[:2])

    # No need to check compatibility since we're creating the mask with the right dimensions
    
    # Compute consistency metrics
    metrics = compute_normal_consistency(normal_map, mask)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two normal maps and analyze surface flatness"
    )
    parser.add_argument(
        "--normal1", type=str, required=True, help="Path to first normal map"
    )
    parser.add_argument(
        "--normal2", type=str, required=True, help="Path to second normal map"
    )
    parser.add_argument(
        "--mask", type=str, help="Path to LabelMe mask JSON file for flatness analysis"
    )
    args = parser.parse_args()

    # Compare the two normal maps
    print("\nComparing normal maps...")
    metrics = compare_normal_maps(args.normal1, args.normal2)
    print("Comparison Metrics:")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"Cosine Similarity: {metrics['cosine_similarity']:.6f}")
    print(f"Angular Error (degrees): {metrics['angular_error']:.2f}")
    print(f"PSNR: {metrics['psnr']:.2f}")

    # If mask is provided, analyze surface flatness
    if args.mask:
        print("\nAnalyzing surface flatness...")
        flatness_metrics = analyze_flat_surface(args.normal1, args.mask)
        print("Flatness Metrics:")
        print(f"Normal Vector Variance: {flatness_metrics['variance']:.6f}")
        print(
            f"Mean Angular Deviation: {flatness_metrics['mean_angular_deviation']:.2f} degrees"
        )
        print(f"Planarity Score: {flatness_metrics['planarity_score']:.6f}")

import argparse
from hole_detector import HoleDetector

def main():
    parser = argparse.ArgumentParser(description="Detect circular holes in segmented 3DGS objects")
    parser.add_argument("--scene", type=str, required=True, help="Path to the 3DGS scene point cloud")
    parser.add_argument("--mask", type=str, required=True, help="Path to the segmentation mask")
    parser.add_argument("--output", type=str, default="./hole_detection_results", help="Output directory")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum number of viewpoints to try")
    parser.add_argument("--max-optimizations", type=int, default=5, help="Maximum optimization steps per viewpoint")
    
    args = parser.parse_args()
    
    # Initialize and run detector
    detector = HoleDetector(args.scene, args.mask, args.output)
    results = detector.detect_hole(args.max_iterations, args.max_optimizations)
    
    if results["found"]:
        print("\nHole detected!")
        print(f"Circularity: {results['circularity']:.3f}")
        print(f"Dimensions: {results['width']:.1f} x {results['height']:.1f} pixels")
    else:
        print("\nNo holes detected in the object.")

if __name__ == "__main__":
    main()
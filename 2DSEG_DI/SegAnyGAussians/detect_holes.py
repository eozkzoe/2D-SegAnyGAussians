import argparse
from hole_detector import HoleDetector

def main():
    parser = argparse.ArgumentParser(description="Detect circular holes in segmented 3DGS objects")
    parser.add_argument("--scene", type=str, required=True, help="Path to the COLMAP scene directory")
    parser.add_argument("--model", type=str, required=True, help="Path to the pre-trained 3DGS model directory")
    parser.add_argument("--mask", type=str, required=True, help="Path to the segmentation mask")
    parser.add_argument("--output", type=str, default="./hole_detection_results", help="Output directory")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum number of viewpoints to try")
    parser.add_argument("--max-optimizations", type=int, default=5, help="Maximum optimization steps per viewpoint")
    parser.add_argument("--debug", action="store_true", help="Save debug renders of all viewpoints")
    args = parser.parse_args()
    
    # Initialize and run detector
    detector = HoleDetector(args.scene, args.model, args.mask, args.output, debug=args.debug)
    results = detector.detect_hole(args.max_iterations, args.max_optimizations)
    
    if results["found"]:
        print("\nHole detected!")
        print(f"Circularity: {results['circularity']:.3f}")
        print(f"Dimensions: {results['width']:.1f} x {results['height']:.1f} pixels")
    else:
        print("\nNo holes detected in the object.")

if __name__ == "__main__":
    main()
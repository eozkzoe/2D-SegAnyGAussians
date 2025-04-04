import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from utils.graphics_utils import focal2fov, fov2focal


class StereoVision:
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height

    def compute_fundamental_matrix(self, matches, kpts1, kpts2):
        """Compute fundamental matrix from matched keypoints."""
        if len(matches) < 8:
            return None

        pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches])

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        return F, mask, pts1, pts2

    def compute_essential_matrix(self, F, K):
        """Compute essential matrix from fundamental matrix and intrinsics."""
        E = K.T @ F @ K
        return E

    def decompose_essential_matrix(self, E, K, pts1, pts2):
        """Decompose essential matrix to get relative camera pose."""
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        return R, t, mask

    def compute_disparity(self, img1, img2):
        """Compute disparity map using Semi-Global Block Matching."""
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2

        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )

        disparity = stereo.compute(img1_gray, img2_gray).astype(np.float32) / 16.0
        return disparity

    def depth_from_disparity(self, disparity, baseline, focal_length):
        """Convert disparity to depth using triangulation."""
        depth = (baseline * focal_length) / (disparity + 1e-6)
        return depth

    def compute_normals(self, depth):
        """Compute surface normals from depth map using cross product method."""
        zy, zx = np.gradient(depth)

        # Create grid of coordinates
        x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))

        # Stack X, Y, Z coordinates
        normals = np.dstack((-zx, -zy, np.ones_like(depth)))

        # Normalize
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / (norm + 1e-6)

        return normals

    def estimate_depth_and_normals(self, img1, img2, baseline=0.1, fov=60):
        """Main function to estimate depth and surface normals from stereo images."""
        # Compute focal length from FoV
        focal_length = fov2focal(np.radians(fov), self.height)

        # Create camera intrinsic matrix
        K = np.array(
            [
                [focal_length, 0, self.width / 2],
                [0, focal_length, self.height / 2],
                [0, 0, 1],
            ]
        )

        # Feature detection and matching
        sift = cv2.SIFT_create()
        kpts1, desc1 = sift.detectAndCompute(img1, None)
        kpts2, desc2 = sift.detectAndCompute(img2, None)

        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Compute fundamental matrix
        F, mask, pts1, pts2 = self.compute_fundamental_matrix(
            good_matches, kpts1, kpts2
        )
        if F is None:
            return None, None

        # Compute essential matrix
        E = self.compute_essential_matrix(F, K)

        # Recover pose
        R, t, pose_mask = self.decompose_essential_matrix(E, K, pts1, pts2)

        # Compute disparity
        disparity = self.compute_disparity(img1, img2)

        # Convert disparity to depth
        depth = self.depth_from_disparity(disparity, baseline, focal_length)

        # Compute surface normals
        normals = self.compute_normals(depth)

        return depth, normals


def test_stereo_vision():
    """Test function for stereo vision module."""
    # Load test images
    img1 = cv2.imread("test_data/left.png")
    img2 = cv2.imread("test_data/right.png")

    if img1 is None or img2 is None:
        print("Error: Could not load test images")
        return

    # Create stereo vision object
    stereo = StereoVision()

    # Estimate depth and normals
    depth, normals = stereo.estimate_depth_and_normals(img1, img2)

    if depth is None:
        print("Error: Could not estimate depth and normals")
        return

    # Visualize results
    cv2.imshow("Depth Map", depth / depth.max())

    # Visualize normals as RGB
    normals_rgb = (normals + 1) / 2
    cv2.imshow("Surface Normals", normals_rgb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_stereo_vision()

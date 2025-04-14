import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_holes(json_path, vector_length=0.1):
    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Calculate mean center of all points
    centers = np.array([np.array(hole["center"]) for hole in data["holes"]])
    mean_center = np.mean(centers, axis=0)

    # Rotation matrix for 90 degrees CCW
    rot_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Plot each hole center and normal
    for hole in data["holes"]:
        # Get center and normal
        center = np.array(hole["center"])
        normal = np.array(hole["normal"])

        # Translate to origin, rotate, and translate back
        center_centered = center - mean_center
        center = (rot_matrix @ center_centered) + mean_center
        normal = rot_matrix @ normal

        # Rotate normal vector
        normal = rot_matrix @ normal

        # Normalize normal vector and scale to desired length
        normal = normal / np.linalg.norm(normal) * vector_length

        # Calculate end point of normal vector
        end_point = center + normal

        # Plot center point
        ax.scatter(center[0], center[1], center[2], color="blue", s=100)

        # Plot normal vector
        ax.quiver(
            center[0],
            center[1],
            center[2],
            normal[0],
            normal[1],
            normal[2],
            color="red",
            arrow_length_ratio=0.2,
        )

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set title
    ax.set_title("Hole Centers and Normal Vectors")

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Add legend
    ax.scatter([], [], [], color="blue", s=100, label="Hole Centers")
    ax.quiver([], [], [], [], [], [], color="red", label="Normal Vectors")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize hole centers and normals")
    parser.add_argument(
        "json_file", type=str, help="Path to the JSON file containing hole information"
    )
    parser.add_argument(
        "--vector_length",
        type=float,
        default=0.1,
        help="Length of normal vectors in visualization",
    )

    args = parser.parse_args()

    visualize_holes(args.json_file, args.vector_length)

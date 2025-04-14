import pymeshlab
import numpy as np

# Load the .pptpl file
ms = pymeshlab.MeshSet()
ms.load_new_mesh("your_pointcloud.pptpl")

# Access the current mesh
mesh = ms.current_mesh()

# Get vertex normals (shape: N x 3)
normals = mesh.vertex_normal_array()

# Compute mean normal vector
mean_normal = np.mean(normals, axis=0)

# Normalize the mean normal
mean_normal /= np.linalg.norm(mean_normal)

print("Mean normal:", mean_normal)

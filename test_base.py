import numpy as np
from stl import mesh
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# ----- PARAMETERS -----
input_filename = 'data/orders/1738280583--dA7XXUF3qlBSoJiZAAAB/mesh.stl'    # your original STL file
output_filename = 'data/orders/1738280583--dA7XXUF3qlBSoJiZAAAB/mesh_base4.stl'
base_z = 0.0                    # z coordinate for the flat base


import numpy as np
from stl import mesh
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Load your STL file (change 'terrain.stl' as needed)
stl_mesh = mesh.Mesh.from_file(input_filename)  # :contentReference[oaicite:1]{index=1}

# Extract all vertices from the triangles (v0, v1, v2)
xyz = np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))

# Determine the minimum z-value (i.e. the "lowest point" of the mesh)
z_min = xyz[:, 2].min()
print("Minimum z-value (extrusion height):", z_min)

# Project vertices to 2D by taking only the x and y coordinates
xy = xyz[:, :2]

# Compute the convex hull of the 2D points to get the border (outline)
hull = ConvexHull(xy)
border_points_2d = xy[hull.vertices]

# Extrude these border points: assign each a z value equal to z_min
extruded_border_points = np.hstack((border_points_2d,
                                      z_min * np.ones((border_points_2d.shape[0], 1))))

import meshlib.mrmeshpy
import meshlib.mrmeshnumpy


vector_3d = meshlib.mrmeshnumpy.fromNumpyArray(extruded_border_points)
mesh = meshlib.mrmeshpy.terrainTriangulation(vector_3d)

meshlib.mrmeshpy.saveMesh(mesh, output_filename)


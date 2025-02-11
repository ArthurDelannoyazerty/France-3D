import numpy as np
import meshlib.mrmeshpy
import meshlib.mrmeshnumpy
import trimesh

from stl import mesh
from scipy.spatial import ConvexHull



terrain_stl_filepath = 'data/orders/1738280583--dA7XXUF3qlBSoJiZAAAB/mesh.stl'    # your original STL file
base_stl_filepath = 'data/orders/1738280583--dA7XXUF3qlBSoJiZAAAB/mesh_base4.stl'
base_z = 0.0                    # z coordinate for the flat base

# Load your STL file (change 'terrain.stl' as needed)
stl_mesh_terrain = mesh.Mesh.from_file(terrain_stl_filepath)  # :contentReference[oaicite:1]{index=1}

# Extract all vertices from the triangles (v0, v1, v2)
xyz_terrain = np.vstack((stl_mesh_terrain.v0, stl_mesh_terrain.v1, stl_mesh_terrain.v2))

# Determine the minimum z-value (i.e. the "lowest point" of the mesh)
z_min_terrain = xyz_terrain[:, 2].min()
print("Minimum z-value (extrusion height):", z_min_terrain)

# Project vertices to 2D by taking only the x and y coordinates
xy_terrain = xyz_terrain[:, :2]

# Compute the convex hull of the 2D points to get the border (outline)
hull_terrain = ConvexHull(xy_terrain)
xyz_hull_terrain = xyz_terrain[hull_terrain.vertices]
xy_hull_terrain  = xy_terrain[hull_terrain.vertices]

# Extrude these border points: assign each a z value equal to z_min
xyz_hull_base = np.hstack((xy_hull_terrain, z_min_terrain * np.ones((xy_hull_terrain.shape[0], 1))))

# Triangulate the point cloud int oa stl file
# vector_3d = meshlib.mrmeshnumpy.fromNumpyArray(extruded_border_points)
# flat_base_mesh = meshlib.mrmeshpy.terrainTriangulation(vector_3d)
# meshlib.mrmeshpy.saveMesh(flat_base_mesh, output_filename)



stl_mesh_base = mesh.Mesh.from_file(base_stl_filepath)


if len(xyz_hull_terrain) != len(xyz_hull_base):
    def sort_polygon(points):
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        order = np.argsort(angles)
        return points[order]
    xyz_hull_terrain = sort_polygon(xyz_hull_terrain)
    xyz_hull_base = sort_polygon(xyz_hull_base)


n = len(xyz_hull_terrain)  # Number of boundary points
print(n)

# --- Build the side faces connecting the terrain boundary to the base boundary ---
# We create a new set of vertices by stacking the terrain boundary (top) and base boundary (bottom)
side_vertices = np.vstack((xyz_hull_terrain, xyz_hull_base))
side_faces = []
for i in range(n):
    i_next = (i + 1) % n
    # Define indices for the top (terrain) and bottom (base) vertices.
    top_i = i             # vertex i on terrain
    top_next = i_next     # next vertex on terrain
    bottom_i = i + n      # corresponding vertex on base (offset by n)
    bottom_next = i_next + n  # next vertex on base

    # Create two triangles for the quad between consecutive boundary points:
    # Triangle 1: top_i, top_next, bottom_i
    # Triangle 2: top_next, bottom_next, bottom_i
    side_faces.append([top_i, top_next, bottom_i])
    side_faces.append([top_next, bottom_next, bottom_i])


side_faces = np.array(side_faces)
side_mesh = trimesh.Trimesh(vertices=side_vertices, faces=side_faces)
side_mesh.export('side_mesh.stl')

# --- Merge the three parts into one final mesh ---
# (terrain, base, and the newly created side walls)
final_mesh = trimesh.util.concatenate([stl_mesh_terrain, stl_mesh_base, side_mesh])

# Export the merged mesh as a new STL file
final_mesh.export('merged.stl')
print("Merged mesh saved as 'merged.stl'")
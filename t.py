import numpy as np
from stl import mesh

def load_and_extrude_stl(input_file, output_file):
    # 1. Load the original STL mesh
    original_mesh = mesh.Mesh.from_file(input_file)
    
    # 2. Determine the minimum z value among all vertices in the original mesh
    #    original_mesh.vectors has shape (n_triangles, 3, 3)
    min_z = original_mesh.vectors[:, :, 2].min()
    
    # 3. Extract all vertices from the triangle soup and obtain unique vertices.
    #    Reshape the (n_triangles, 3, 3) array into (-1, 3) and then use np.unique.
    all_vertices = original_mesh.vectors.reshape(-1, 3)
    unique_vertices, inverse = np.unique(all_vertices, axis=0, return_inverse=True)
    # Faces (triangles) will be defined by indices into unique_vertices.
    faces = inverse.reshape(-1, 3)
    n_unique = unique_vertices.shape[0]
    
    # 4. Create the flattened (bottom) vertices as a copy of unique vertices with z set to min_z.
    flattened_vertices = unique_vertices.copy()
    flattened_vertices[:, 2] = min_z
    
    # 5. Build the new vertex list for the volume:
    #    The first n_unique vertices are the original (top) vertices,
    #    and the next n_unique are the flattened (bottom) vertices.
    new_vertices = np.vstack((unique_vertices, flattened_vertices))
    
    # 6. The top face of the volume is just the original faces.
    top_faces = faces
    #    The bottom face is the copy of the top face but with vertex indices shifted by n_unique.
    #    Reverse the order so the normal points in the opposite direction.
    bottom_faces = faces[:, ::-1] + n_unique

    # 7. Find the boundary edges of the top surface.
    #    For each triangle face, define its three edges.
    edges1 = faces[:, [0, 1]]
    edges2 = faces[:, [1, 2]]
    edges3 = faces[:, [2, 0]]
    edges = np.vstack((edges1, edges2, edges3))
    # Sort each edge so that the order is canonical (lowest index first)
    # Ensure edges is contiguous
    edges = np.sort(edges, axis=1)
    edges = np.ascontiguousarray(edges)  # Make it contiguous in memory

    # Now create a structured view of the edges
    dtype = np.dtype([('v0', edges.dtype), ('v1', edges.dtype)])
    structured_edges = edges.view(dtype)
    unique_struct_edges, counts = np.unique(structured_edges, return_counts=True)
    unique_edges = unique_struct_edges.view(edges.dtype).reshape(-1, 2)
    # Boundary edges appear only once.
    boundary_edges = unique_edges[counts == 1]
    
    # 8. For each boundary edge (defined by vertex indices v0, v1), create two side triangles:
    #    Triangle 1: [v0, v1, v1 + n_unique]
    #    Triangle 2: [v0, v1 + n_unique, v0 + n_unique]
    side_faces = []
    for edge in boundary_edges:
        v0, v1 = edge
        side_faces.append([v0, v1, v1 + n_unique])
        side_faces.append([v0, v1 + n_unique, v0 + n_unique])
    side_faces = np.array(side_faces)
    
    # 9. Combine all faces: the top faces, the bottom faces, and the side faces.
    all_faces = np.vstack((top_faces, bottom_faces, side_faces))
    
    # 10. Create a new triangle array from the new vertices and combined face indices.
    new_triangles = new_vertices[all_faces]
    
    # 11. Build the new mesh using the numpy-stl Mesh constructor.
    new_mesh = mesh.Mesh(np.zeros(new_triangles.shape[0], dtype=mesh.Mesh.dtype))
    new_mesh.vectors = new_triangles
    
    # 12. Save the resulting volumized mesh to an STL file.
    new_mesh.save(output_file)
    print(f"Extruded mesh saved to {output_file}")

# Example usage:
if __name__ == '__main__':
    input_stl = 'data/orders/1739534558--1ZmQ9kF02LO6c9TLAAAB/mesh.stl'         # Path to your input STL file
    output_stl = 'extruded.stl'     # Path for the output (volumized) STL file
    load_and_extrude_stl(input_stl, output_stl)

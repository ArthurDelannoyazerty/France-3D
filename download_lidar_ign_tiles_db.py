import sys
import wget
import logging
import py7zr
import os
import zipfile
import laspy
import lazrs
import numpy as np
import pyvista as pv
import laspy
import numpy as np
import open3d as o3d


from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def download_lidar_ign_tiles_db(output_dir:str):
    url1 = "https://diffusion-lidarhd-classe.ign.fr/download/lidar/shp/classe"
    url2 = "https://zenodo.org/records/13793544/files/grille.zip"

    try:
        logger.info(f'Downloading IGN LIDAR Database from url : {url1}')
        wget.download(url=url1, out=output_dir)
    except:
        logger.info(f'Error while downloading from : {url1}  | Trying with : {url2}')
        try:
            wget.download(url=url2, out=output_dir)
        except:
            logger.error('Download failed for both url, shuting down...')
            sys.exit(1)
    logger.info(f'Files downloaded in : {output_dir}')


def extract_zip(filepath:str, folderpath:str):
    """Extract the zip file(filepath) to the desired folder(folderpath)."""
    with zipfile.ZipFile(filepath, 'r') as archive:
        archive.extractall(folderpath)


def laz_to_numpy(filepath_in) -> np.ndarray:
    logger.info('Transforming .laz into numpy array.')
    las = laspy.read(filepath_in)
    return las.xyz
    # index 0  : X
    # index 1  : Y
    # index 2  : Z
    # index 3  : ?
    # index 4  : 
    # index 5  : ?
    # index 6  : ?
    # index 7  : ?
    # index 8  : ?
    # index 9  : ?
    # index 10 : ?


def numpy_to_ply(xyz:np.ndarray, filepath_out:str):
    logger.info('Transforming numpy array into a .ply file')
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Save the point cloud to a .ply file
    o3d.io.write_point_cloud(filepath_out, pcd)


def decimate_array(array:np.ndarray, percentage_to_remove):
    """
    Remove a given percentage of rows from a NumPy array randomly.

    Parameters:
    - array (np.ndarray): Input array of shape (n, 3).
    - percentage (float): Percentage of rows to remove (between 0 and 100).
    - seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    - np.ndarray: Array with the specified percentage of rows removed.
    """
    logger.info(f'Decimating array of shape {array.shape} by {percentage_to_remove}%')
    if not 0 <= percentage_to_remove <= 100:
        raise ValueError("Percentage must be between 0 and 100.")

    total_rows = array.shape[0]
    num_rows_to_remove = int(total_rows * (percentage_to_remove / 100.0))

    # Generate random indices to remove
    indices_to_remove = np.random.choice(total_rows, num_rows_to_remove, replace=False)

    # Create a mask that is True for rows to keep
    mask = np.ones(total_rows, dtype=bool)
    mask[indices_to_remove] = False

    # Return the array with the specified rows removed
    return array[mask]


def show_point_cloud(points):
    """
    Display a 3D point cloud using PyVista.
    
    Parameters:
    - points (numpy.ndarray): A Nx3 array of 3D points (x, y, z).
    """
    logger.info('Display point cloud')
    point_cloud = pv.PolyData(points)
    plotter = pv.Plotter()
    plotter.add_points(point_cloud, cmap="viridis", point_size=1)
    plotter.set_background("white")
    plotter.show()


def ply_pointcloud_to_ply_mesh(ply_pointcloud_filepath:str, ply_mesh_filepath:str):
    logger.info('Begin operation point cloud to mesh')
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(ply_pointcloud_filepath)

    # Estimate normals
    pcd.estimate_normals()
    logger.info('Estimated normals done')

    # Compute average distance between points
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    logger.info('Distances nearest neighbors done')

        # Initialize the mesh
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius*2])
    )

    # Save the mesh
    o3d.io.write_triangle_mesh(ply_mesh_filepath, mesh)


def display_ply_mesh(mesh_filepath:str):
    logger.info('Display mesh.')
    mesh = o3d.io.read_triangle_mesh(mesh_filepath)

    # Check if the mesh has vertex normals; if not, compute them
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])



if __name__=="__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    
    zip_output_dir = "sandbox/data_grille/"
    default_zip_name = 'grille.zip'

    # Download zip if it doesn't exists
    zip_filepath = zip_output_dir + default_zip_name
    if not os.path.isfile(zip_filepath):
        download_lidar_ign_tiles_db(zip_output_dir)

    # Extract files from zip
    # extract_zip(zip_filepath, zip_output_dir)

    # download tile
    example_url = 'https://storage.sbg.cloud.ovh.net/v1/AUTH_63234f509d6048bca3c9fd7928720ca1/ppk-lidar/BE/LHD_FXX_0188_6861_PTS_C_LAMB93_IGN69.copc.laz'
    # wget.download(example_url, out='sandbox/data_grille')

    laz_filepath = 'sandbox/data_grille/LHD_FXX_0188_6861_PTS_C_LAMB93_IGN69.copc.laz'

    # .laz to numpy
    # xyz = laz_to_numpy(laz_filepath)
    # xyz = decimate_array(xyz, 75)

    # show_point_cloud(xyz)

    # # numpy to .ply
    # ply_filepath = laz_filepath.split('.')[0] + '.ply'
    # numpy_to_ply(xyz, ply_filepath)

    
    # # .ply point cloud to .ply mesh
    mesh_filepath =  'sandbox/data_grille/mesh.ply'
    # ply_pointcloud_to_ply_mesh(ply_filepath, mesh_filepath)

    display_ply_mesh(mesh_filepath)
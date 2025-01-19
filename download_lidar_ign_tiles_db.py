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
    with laspy.open(filepath_in, 'r') as las:
        return np.array(las.read().points.array.tolist())       # Covnert numpy void to np array
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


def show_point_cloud(points, types):
    """
    Display a 3D point cloud using PyVista.
    
    Parameters:
    - points (numpy.ndarray): A Nx3 array of 3D points (x, y, z).
    """
    point_cloud = pv.PolyData(points)
    point_cloud["point_color"] = types
    plotter = pv.Plotter()
    plotter.add_points(point_cloud, cmap="viridis", point_size=1)
    plotter.set_background("white")
    plotter.show()


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


    # 'laz' to 'las'
    array = laz_to_numpy('sandbox/data_grille/LHD_FXX_0188_6861_PTS_C_LAMB93_IGN69.copc.laz')
    print(f'All data shape : {array.shape}')
    print(f'example first data : {array[0]}')

    points = array[:, :3]
    types =  array[:, 4]

    # Call the function to display the point cloud
    show_point_cloud(points, types)
import sys
import wget
import logging
import py7zr
import os
import zipfile

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
    extract_zip(zip_filepath, zip_output_dir)

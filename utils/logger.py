import logging

from datetime import datetime
from pathlib import Path


def setup_logging(log_name:str,
                  log_dir:Path  = Path('data/logs/'),
                  level_library = logging.INFO,
                  level_project = logging.DEBUG):
    """Setup the logging global parameters.
    :param str log_name: The name of the gile that contains the logs for this session
    :param str log_dir: The name of the directory in which the log file will be saved, defaults to 'logs/'
    :param _type_ level_library: The level of debug for the file other than the current project. Technically of type `int` but advise `logging.{DEBUG/INFO/WARNING}`, defaults to logging.INFO
    :param _type_ level_project: The level of debug for the current project. Technically of type `int` but advise `logging.{DEBUG/INFO/WARNING}`, defaults to logging.DEBUG
    """
    log_filename = log_name + '---' + str(int(datetime.now().timestamp()*1000)) + '.log'
    log_filepath = log_dir / log_filename
    logging.basicConfig(
        level=level_library,
        format='[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ],
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("src").setLevel(level_project)    # Get DEBUG level for all of the project
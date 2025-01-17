import logging
from datetime import datetime

def setup_logging(level_library = logging.INFO,
                  level_project = logging.DEBUG):
    """Setup the logging global parameters.
    :param str log_name: The name of the gile that contains the logs for this session
    :param str log_dir: The name of the directory in which the log file will be saved, defaults to 'logs/'
    :param _type_ level_library: The level of debug for the file other than the current project. Technically of type `int` but advise `logging.{DEBUG/INFO/WARNING}`, defaults to logging.INFO
    :param _type_ level_project: The level of debug for the current project. Technically of type `int` but advise `logging.{DEBUG/INFO/WARNING}`, defaults to logging.DEBUG
    """
    logging.basicConfig(
        level=level_library,
        format='[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ],
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("src").setLevel(level_project)    # Get DEBUG level for all of the project
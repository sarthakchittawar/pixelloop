import logging
import os
import sys

# disable matplotlib and PIL logging
logger_blocklist = [
    "matplotlib",
    "PIL",
    "h5py",
    "wandb",
    "git",
]

for module in logger_blocklist:
    logging.getLogger(module).setLevel(logging.WARNING)

def setup_logging(filename=None, level=logging.WARNING, console=False, console_level=None):
    # Central logging configuration

    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level if console_level is not None else level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if filename is not None:
        add_file_handler(logging.getLogger(), filename)

def create_file_handler(filename):
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
    return file_handler

def add_file_handler(logger, filename):
    file_handler = create_file_handler(filename)
    logger.addHandler(file_handler)

def update_file_handler(logger, filename):
    # Remove existing FileHandler
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()  # Close the old file handler

    new_file_handler = create_file_handler(filename)
    logger.addHandler(new_file_handler)

def update_file_handler_root(filename):
    root_logger = logging.getLogger()
    
    # Remove existing FileHandler only, keep console handlers
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    # Add the new file handler
    file_handler = create_file_handler(filename)
    root_logger.addHandler(file_handler)
    
    # Ensure root logger level allows INFO messages
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
        
    # Also add console handler if none exists
    has_console_handler = any(isinstance(h, logging.StreamHandler) and 
                             hasattr(h, 'stream') and h.stream == sys.stdout
                             for h in root_logger.handlers)
    if not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
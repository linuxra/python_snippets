import logging
import os
from datetime import datetime
import functools
class CustomLogger:
    def __init__(self, name, log_level=logging.INFO, log_file=None, log_format=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        if not log_format:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(log_format)

        if log_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file_with_timestamp = f"{log_file}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file_with_timestamp)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def set_log_level(self, log_level):
        self.logger.setLevel(log_level)

    def remove_handlers(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    def add_handler(self, handler):
        self.logger.addHandler(handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)





# Create a logger instance with a log file
logger = CustomLogger('my_module', log_file='my_log_file')

# Change the log level to DEBUG
logger.set_log_level(logging.INFO)

# Log messages
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.debug('This is a debug message')

# Remove existing handlers and add a new file handler with a custom format
logger.remove_handlers()
custom_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
file_handler = logging.FileHandler('custom_format_log.log')
file_handler.setFormatter(logging.Formatter(custom_format))
logger.add_handler(file_handler)

# Log messages with the new format
logger.info('This is an info message with a custom format')
logger.warning('This is a warning message with a custom format')
logger.error('This is an error message with a custom format')
logger.debug('This is a debug message with a custom format')

def logging_decorator(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"Calling function '{func.__name__}' with args: {args} and kwargs: {kwargs}")
                result = func(*args, **kwargs)
                logger.info(f"Function '{func.__name__}' returned: {result}")
                return result
            except Exception as e:
                logger.error(f"Function '{func.__name__}' raised an exception: {e}", exc_info=True)
                raise
        return wrapper
    return decorator
@logging_decorator(logger)
def add(a,b):
    return a + b
result = add(10,9)
logger.info(f"{result}")



import logging
import os
from datetime import datetime
import pandas as pd
import functools
class CustomLogger:
    """
    A custom logger class that simplifies the process of configuring and using a logger.
    
    This class wraps the Python logging module and provides an interface to easily
    create, configure, and use a logger with both console and file output.
    """

    def __init__(self, name, log_level=logging.INFO, log_file=None, log_format=None):
        """
        Initialize a CustomLogger instance.

        :param name: The name of the logger.
        :param log_level: The logging level. Defaults to logging.INFO.
        :param log_file: The path to the log file. If provided, log messages will be written to the file.
        :param log_format: The format for log messages. Defaults to '%(asctime)s - %(name)s - %(levelname)s - %(message)s'.
        """
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
        """
        Set the logging level for this logger.

        :param log_level: The logging level.
        """
        self.logger.setLevel(log_level)

    def remove_handlers(self):
        """
        Remove all handlers from this logger.
        """
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    def add_handler(self, handler):
        """
        Add a handler to this logger.

        :param handler: The handler to add.
        """
        self.logger.addHandler(handler)

    def info(self, message):
        """
        Log an info-level message.

        :param message: The message to log.
        """
        self.logger.info(message)

    def warning(self, message):
        """
        Log a warning-level message.

        :param message: The message to log.
        """
        self.logger.warning(message)

    def error(self, message):
        """
        Log an error-level message.

        :param message: The message to log.
        """
        self.logger.error(message)

    def debug(self, message):
        """
        Log a debug-level message.

        :param message: The message to log.
        """
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
def logging_decorator1(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def arg_repr(arg):
                if isinstance(arg, pd.DataFrame):
                    return f"DataFrame(shape={arg.shape})"
                else:
                    return repr(arg)

            args_repr = ', '.join(arg_repr(arg) for arg in args)
            kwargs_repr = ', '.join(f"{k}={arg_repr(v)}" for k, v in kwargs.items())

            try:
                logger.info(f"Calling function '{func.__name__}' with args: ({args_repr}) and kwargs: {{{kwargs_repr}}}")
                result = func(*args, **kwargs)
                logger.info(f"Function '{func.__name__}' returned: {arg_repr(result)}")
                return result
            except Exception as e:
                logger.error(f"Function '{func.__name__}' raised an exception: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

def logging_decorator2(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def arg_repr(arg):
                return f"<{arg.__class__.__name__} object at {hex(id(arg))}>"

            args_repr = ', '.join(arg_repr(arg) for arg in args)
            kwargs_repr = ', '.join(f"{k}={arg_repr(v)}" for k, v in kwargs.items())

            try:
                logger.info(f"Calling function '{func.__name__}' with args: ({args_repr}) and kwargs: {{{kwargs_repr}}}")
                result = func(*args, **kwargs)
                logger.info(f"Function '{func.__name__}' returned: {arg_repr(result)}")
                return result
            except Exception as e:
                logger.error(f"Function '{func.__name__}' raised an exception: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

def logging_decorator3(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def arg_repr(arg):
                if isinstance(arg, pd.DataFrame):
                    return f"DataFrame: \n{arg.head()}"
                elif isinstance(arg, str):
                    return arg
                else:
                    return f"<{arg.__class__.__name__} object at {hex(id(arg))}>"

            args_repr = ', '.join(arg_repr(arg) for arg in args)
            kwargs_repr = ', '.join(f"{k}={arg_repr(v)}" for k, v in kwargs.items())

            try:
                logger.info(f"Calling function '{func.__name__}' with args: ({args_repr}) and kwargs: {{{kwargs_repr}}}")
                result = func(*args, **kwargs)
                logger.info(f"Function '{func.__name__}' returned: {arg_repr(result)}")
                return result
            except Exception as e:
                logger.error(f"Function '{func.__name__}' raised an exception: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

def logging_decorator(logger):
    """
    A decorator to log information about the execution of decorated functions.
    
    This decorator logs the function call with its arguments and the returned result.
    It handles various argument types like DataFrame, string, int, float, list, tuple, and dict
    by providing informative and truncated string representations.

    :param logger: A logger object responsible for logging messages.
    :return: A decorator function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def arg_repr(arg):
                if isinstance(arg, pd.DataFrame):
                    return f"DataFrame: \n{arg.head()}"
                elif isinstance(arg, (str, int, float)):
                    return str(arg)
                elif isinstance(arg, (list, tuple)):
                    return f"{arg.__class__.__name__}({str(arg)[:50] + '...' if len(str(arg)) > 50 else str(arg)})"
                elif isinstance(arg, dict):
                    return f"dict({str(arg)[:50] + '...' if len(str(arg)) > 50 else str(arg)})"
                else:
                    return f"<{arg.__class__.__name__} object at {hex(id(arg))}>"

            args_repr = ', '.join(arg_repr(arg) for arg in args)
            kwargs_repr = ', '.join(f"{k}={arg_repr(v)}" for k, v in kwargs.items())

            try:
                logger.info(f"Calling function '{func.__name__}' with args: ({args_repr}) and kwargs: {{{kwargs_repr}}}")
                result = func(*args, **kwargs)
                logger.info(f"Function '{func.__name__}' returned: {arg_repr(result)}")
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



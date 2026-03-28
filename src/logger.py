import logging
import os

def setup_logger():
    log_level = os.getenv("LOG_LEVEL", "INFO")

    logger = logging.getLogger("food_app")
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)


    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
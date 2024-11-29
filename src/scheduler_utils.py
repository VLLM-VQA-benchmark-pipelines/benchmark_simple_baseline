import json
import logging

from jsonschema import ValidationError, validate

from benchmark_utils import (
    evaluate_model_metrics_by_answers,
    get_model_answers_on_VQA_dataset,
)

# Setup logging
logging.basicConfig(
    filename="config_error.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log_and_print(message):
    """Logs a message and prints it to the console.
    Args:
        message (str): The message to log and print.
    Returns:
        None
    """
    print(message)
    logging.info(message)


# Define jsonschema for config file
config_schema = {
    "type": "object",
    "properties": {
        "images": {"type": "object", "additionalProperties": {"type": "string"}},
        "datasets": {"type": "object", "additionalProperties": {"type": "string"}},
        "models": {"type": "object", "additionalProperties": {"type": "string"}},
        "metrics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["metrics", "datasets", "models"],
}


class JSONParser:
    """Parser for loading and validating JSON configuration files.
    This class provides methods to load configuration data from a JSON file,
    validate the data against a predefined schema, and optionally display the
    JSON data in a formatted way.
    Attributes:
        None
    """

    def __init__(self, file_path, display_json=False):
        self.cfg = self.load_config(file_path, display_json)

        print(f"Валидация конфигурационного файла: {file_path}")
        if self.validate_config(self.cfg):
            print("Конфигурационный файл корректен")
        else:
            print("Конфигурационный файл не корректен!")

    def load_config(self, file_path, display_json=False):
        """Loads configuration from a JSON file.
        Args:
            file_path (str): The path to the JSON configuration file.
            display_json (bool, optional): If True, prints the JSON data in a formatted way. Defaults to False.
        Returns:
            dict: The loaded configuration data.
        Raises:
            Exception: If there is an error loading the configuration.
        """
        try:
            with open(file_path, "r") as file:
                config_data = json.load(file)
            if self.validate_config(config_data):
                if display_json == True:
                    print(json.dumps(config_data, indent=4))
                log_and_print(f"Configuration loaded from file: {file_path}")
                return config_data
        except Exception as err:
            log_and_print(f"Configuration loading error: {err}")
            raise

    def validate_config(self, config_data):
        """Validates the configuration data against a predefined schema.
        Args:
            config_data (dict): The configuration data to validate.
        Returns:
            bool: True if the configuration data is valid.
        Raises:
            ValidationError: If the configuration data does not conform to the schema.
        """
        try:
            validate(instance=config_data, schema=config_schema)
            return True
        except ValidationError as err:
            log_and_print(f"Config validation error: {err.message}")
            raise


class BenchmarkScheduler:
    """Scheduler for running benchmarks on models across datasets.
    This class is responsible for managing the configuration of images, models, and datasets,
    updating parameters in a YAML file, and executing benchmarks using DVC.
    Attributes:
        images (list): List of images to be used in the benchmarks.
        models (dict): Dictionary of models to be benchmarked.
        datasets (dict): Dictionary of datasets to be used in the benchmarks.
    """

    def __init__(self, config_path, display_json=False):
        cfg_parser = JSONParser(config_path, display_json)
        config = cfg_parser.cfg

        self.models = config["models"]
        self.datasets = config["datasets"]

    def run_scheduler(
        self, datasets_dir_path, answers_dir_path, cache_directory, metrics_dir_path
    ):
        """Runs benchmarks for all models on all datasets.
        This function iterates through each dataset and model, updates the parameters in the YAML file,
        and executes the DVC reproduction command.
        Returns:
            None
        """
        datasets = tuple(self.datasets.keys())
        models = tuple(self.models.keys())

        for dataset_name in datasets:
            for model_name in models:
                # Блок кода для одного бенчмарка 1 модели на 1 датасете

                # Получаем ответы "1 модели  на 1 датасете"
                get_model_answers_on_VQA_dataset(
                    model_name,
                    dataset_name,
                    datasets_dir_path,
                    answers_dir_path,
                    cache_directory,
                    iter_log=True,
                )

                # Оцениваем метрики "1 модели  на 1 датасете"
                evaluate_model_metrics_by_answers(
                    model_name,
                    dataset_name,
                    datasets_dir_path,
                    answers_dir_path,
                    metrics_dir_path,
                )

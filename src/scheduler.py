import json
from jsonschema import validate, ValidationError
import logging
import os
import yaml

# Setup logging
logging.basicConfig(
    filename="config_error.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
         "images": {
            "type": "object",
            "additionalProperties": {"type": "string"}
            },
        "datasets": {
            "type": "object",
            "additionalProperties": {"type": "string"}
            },
        "models": {
            "type": "object",
            "additionalProperties": {"type": "string"}
            },
        "metrics": {
            "type": "array",
            "items": {"type": "string"}
            },
        },
    "required": ["images", "datasets", "models"]
}


class JSONParser():
    """Parser for loading and validating JSON configuration files.
    This class provides methods to load configuration data from a JSON file,
    validate the data against a predefined schema, and optionally display the
    JSON data in a formatted way.
    Attributes:
        None
    """

    def __init__(self):
        pass    
  
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
            with open(file_path, 'r') as file:
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
        


class BenchmarkScheduler():
    """Scheduler for running benchmarks on models across datasets.
    This class is responsible for managing the configuration of images, models, and datasets,
    updating parameters in a YAML file, and executing benchmarks using DVC.
    Attributes:
        images (list): List of images to be used in the benchmarks.
        models (dict): Dictionary of models to be benchmarked.
        datasets (dict): Dictionary of datasets to be used in the benchmarks.
        dvc_params_yaml_path (str): Path to the YAML file for DVC parameters.
    """   

    def __init__(self, config_file, dvc_params_yaml_path):
        self.images = config_file['images']
        self.models = config_file['models']
        self.datasets = config_file['datasets']
        self.dvc_params_yaml_path = dvc_params_yaml_path


    def update_params_yaml(self, dataset, model):
        """Updates the parameters in a YAML file.
        Args:
            dataset (str): The name of the dataset.
            model (str): The name of the model.
        Returns:
            None
        """

        # Данные для записи в файл YAML
        params_yaml = {
                        'images': self.images,
                        'dataset': dataset,
                        'model': model
                    }                

        with open(self.dvc_params_yaml_path, 'w') as file:
            yaml.dump(params_yaml, file, default_flow_style=False)
            logging.info(f"Params write to: {self.dvc_params_yaml_path}")
    

    def run_scheduler(self):
        """Runs benchmarks for all models on all datasets.
        This function iterates through each dataset and model, updates the parameters in the YAML file,
        and executes the DVC reproduction command.
        Returns:
            None
        """      
        for dataset_name, dataset_path in zip(self.datasets.keys(), self.datasets.values()):
            for model_name, model_path in zip(self.models.keys(), self.models.values()):
                log_and_print(f"\nDataset: {dataset_name}\nModel: {model_name}")
                self.update_params_yaml(dataset_name, model_name)
                os.system("dvc repro")
        
    

if __name__ == "__main__":

    pasre_json = JSONParser()
    config_file = pasre_json.load_config("config.json", display_json=False)
    benchmarks = BenchmarkScheduler(config_file, "params.yaml")
    benchmarks.run_scheduler()


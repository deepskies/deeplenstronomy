from typing import Union
import yaml
import os
from deeplenstronomy.settings import config_template

class YAMLObject():
    """
    ConfigConstructor is an abstract class that defines the interface for all configuration files.
    """
    def __init__(self, config_file_path: str):
        """
        """
        self.__check_file_path_exists__(config_file_path)

        self.file_path = config_file_path    

    def create_yaml_template(self, variables: dict, filename='default_config.yaml'):
        
        if not os.path.exists((self.file_path + filename)):
            with open(self.file_path, 'w') as yaml_file:
                yaml_file.write(yaml.dump(variables, default_flow_style=False))
        else:
            raise ValueError(f"The file {self.file_path} already exists. If you'd like to use another config file, you need to change the file name.")

    def parse_yaml(self, file_path):
        """
        Parse the configuration YAML.
        """
        with open(file_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)


    def update_yaml(self, variables: dict):
        """
        Update a YAML file with the variables provided.

        Args:
            variables (Dict): Dictionary of variables to change and their updated values.
            file_path (str): Path to the YAML file.

        Returns:
            None
        
        Raises:
            yaml.YAMLError: If the YAML file is not valid.
            ValueError: If the variable is not in the YAML file.
        """
        yaml_options = self.__get_yaml_options__(self.file_path)

        for variable in variables:
            if variable.lower().strip() not in yaml_options:
                raise ValueError(f"Variable {variable} is not an option in the YAML file. The options are as follows: {yaml_options}")
            
        original_config = self.parse_yaml(self.file_path)

        for section in original_config.keys():
            for key in original_config[section].keys():
                if key in variables:
                    original_config[section][key] = variables[key]

        with open(self.file_path, 'w') as yaml_file:
            yaml_file.write( yaml.dump(original_config, default_flow_style=False))

    def __get_yaml_options__(self, file_path: Union[None,str]):

        yaml_dict = self.parse_yaml(file_path)

        yaml_key_list = []

        for k in yaml_dict.keys():
            yaml_key_list.append(k.lower().strip())
            for k_2 in yaml_dict[k].keys():
                yaml_key_list.append(k_2.lower().strip())

        return yaml_key_list

    def __check_file_path_exists__(self, file_path):

        if not os.path.exists(file_path):
            raise FileNotFoundError("A configuration file at the path: {file_path} doesn't exist. \
                                    Please choose a new path or create a file using appropriate function.")

                




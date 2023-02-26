from abc import ABC, abstractmethod
from deeplenstronomy.configuration.config_constructor import ConfigConstructor
from typing import Union

class PhysicsConfigConstructur(ConfigConstructor):

    def __init__(self, input_name: Union[None, str]):
        super().__init__()
        self.input = input_name
    
    # TODO: Fill out this method.
    # if the survey is None, produce message about possible surveys and defaulting to a random survey.
    def create_config(self, param_dict: Union[None, dict]):
        pass



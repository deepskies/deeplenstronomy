from abc import ABC, abstractmethod


class ConfigConstructor(ABC):

    def __init__(self):
        pass    
    
    @abstractmethod
    def create_config(self, param_dict: dict, *args):
        pass

    # TODO: fill out this method
    @staticmethod
    def modify_config(self, param_dict: dict):
        pass

    





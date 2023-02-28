from abc import ABC, abstractmethod
from deeplenstronomy.configuration.config_constructor import ConfigConstructor
from typing import Union

class SurveyConfigConstructor(ConfigConstructor):

    def __init__(self, survey_name: Union[None, str]):
        super().__init__()
        self.survey = survey_name
    
    # TODO: Fill out this method.
    # if the survey is None, produce message about possible surveys and defaulting to a random survey.
    def create_config(self, param_dict: Union[None, dict]):
        pass



    

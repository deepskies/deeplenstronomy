import os
from typing import Optional 
import yaml 
from deeplenstronomy.configuration.yaml_object import YAMLObject
from deeplenstronomy.settings import general_config_template

class ConfigObject(YAMLObject):
    """Object used for interfacing with the configuration files."""

    def __init__(self, config_file_path: Optional[str], survey: str="DES"):
        
        super().__init__(config_file_path=config_file_path)

        self._full_config_dict = self.parse_yaml(config_file_path)
    

    @property
    def dataset_params(self):

        return self._full_config_dict['DATASET']
    

    @property
    def cosmology_params(self):

        return self._full_config_dict["COSMOLOGY"]
    

    @property 
    def image_params(self):

        return self._full_config_dict["IMAGE"]
    

    @property
    def survey_params(self):

        return self._full_config_dict["SURVEY"]
    

    @property
    def background_params(self):

        return self._full_config_dict["BACKGROUNDS"]
    

    # TODO: Check that the entered value for the species major and sub-objects is valid.
    def species_params(self, major_object: Optional[str], sub_object: Optional[str]):

        if (major_object and sub_object) is not None:
            call =  self._full_config_dict["SPECIES"][major_object.upper()][sub_object.upper()]
        elif sub_object is None and major_object is not None:
            call = self._full_config_dict["SPECIES"][major_object.upper()]
        else:
            call = self._full_config_dict["SPECIES"]
        
        return call
    

    # TODO: Check that the entered value for the geometry object is valid.
    def geometry_params(self, geometry_object: Optional[str]):
        
        if geometry_object:
            call = self._full_config_dict["GEOMETRY"][geometry_object.upper]
        else:
            call = self._full_config_dict["GEOMETRY"]
        
        return call
        
    # TODO: Check that the entered value for the distribution object is valid.
    def distribution_params(self, dist_object: str):
        
        if dist_object:
            call = self._full_config_dict["DISTRIBUTIONS"][dist_object.upper]
        else:
            call = self._full_config_dict["DISTRIBUTIONS"]
        
        return call

    # TODO: Account for change in the survey with the template type variable. 
    # TODO: Return the configuration template for a specific survey.

    def create_config_template(self, template_type: str='default'):
        
        self.create_yaml_template(variables=general_config_template)
    





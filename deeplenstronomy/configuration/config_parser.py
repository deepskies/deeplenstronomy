import os
from typing import List, Optional 
import yaml 
from deeplenstronomy.configuration.yaml_object import YAMLObject
from deeplenstronomy.settings import general_config_template
from benedict.dicts import KeypathDict
import deeplenstronomy.configuration.check as big_check

class ConfigParser(YAMLObject):
    """Object used for interfacing with the configuration files."""

    def __init__(self, config_file_path: Optional[str], survey: str="DES"):
        
        super().__init__(config_file_path=config_file_path)

        self._full_config_dict = self.parse_yaml(config_file_path)
        
        # Update user inputs in the configuration template.
        if len(self.input_locations) > 0:
            self.update_yaml(variables=self.input_locations)

        self.__check_valid_config_file__()

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
    
    @property
    def input_paths(self):
        return self.__get_keyword_paths__("INPUT")

    @property
    def dist_file_paths(self):   

        return self.__get_dist_file_paths__()

    @property 
    def user_image_file_data(self):

        return self.__get_user_image_data__()

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
    
    def __get_keyword_paths__(self, keyword: str) -> List[dict]:
        """
        Find locations in main dictionary where a keyword is used

        :param kw: str, a keyword to search the dict keys for
        :return: paths: list, the keypaths to all occurances of kw
        """
        d = KeypathDict(self._full_config_dict, keypath_separator='.')
        locs = [x.find(keyword) for x in d.keypaths()]
        paths = [y for y in [x[0:k-1] if k != -1 else '' for x, k in zip(d.keypaths(), locs)] if y != '']
        return paths

    def __get_dist_file_paths__(self):

        file_paths = []     
        if "DISTRIBUTIONS" in self._full_config_dict.keys():
            for k in self._full_config_dict['DISTRIBUTIONS'].keys():
                file_paths.append('DISTRIBUTIONS.' + k)
            
        return file_paths

    def __get_user_image_data__(self):

        file_paths = []
        image_configurations = []
        if "BACKGROUNDS" in self._full_config_dict.keys():
            file_paths.append(self._full_config_dict['BACKGROUNDS']['PATH'])
            image_configs = self.full_dict['BACKGROUNDS']['CONFIGURATIONS'][:]

        return (file_paths, image_configs)

    def __check_valid_config_type__(self):
        pass

    def __check_valid_config_file__(self):
        """
        Check configuration file for possible user errors.
        """
        big_check._run_checks(self._full_config_dict, None)
        


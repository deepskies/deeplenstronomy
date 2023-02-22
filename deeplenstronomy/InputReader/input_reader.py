from abc import ABC, abstractmethod


class InputReader(ABC):
    def __init__(self, input_source: str=None):
        self.dataset_parameters = self.read_input(input_source)

    def read_input(self, input_source):
        # Opens and validates and sets all the params given by the input
        if input_source is None:
            input_source = {"DEFAULT PARAMS"} # Todo. Fill these in.
        dataset_params = self._open_input_source(input_source)
        self._input_source_is_valid()

        return dataset_params

    def _open_input_source(self, input_source):
        # How the open input is read in.
        raise NotImplementedError

    def _input_source_is_valid(self):
        # Verify the given input source matches the requirements of the program
        raise NotImplementedError

    # Methods to set each param individually
    # Making using the input_source optional
    def survey(self, **kwargs):
        pass

    def objects(self):
        pass

    def cosmology(self):
        pass

    def image(self):
        pass

    def noise(self):
        pass

    def geometry(self):
        pass

    def species(self):
        pass

    def _species_light_profiles(self):
        pass

    def _species_mass_profiles(self):
        pass

    def _species_shear_profiles(self):
        pass

    def _species_special(self):
        pass

    def save(self, output_path):
        pass

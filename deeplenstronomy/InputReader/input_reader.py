from abc import ABC, abstractmethod


class InputReader(ABC):
    def __init__(self, configuration: str = None):
        self.configuration = configuration

        # Break that dictionary into segments that can be used by the input reader
        # Without parsing through the whole dic every time a parameters is set

    # Methods to set each param individually
    # Making using the input_source optional
    def bands(self, bands: list = None):
        # Reset bands that will be used to format the dictionary
        pass

    def survey(self, survey=None, **kwargs):
        pass

    def objects(self, object_id=None, **kwargs):
        pass

    def cosmology(self, *kwargs):
        # I don't know what sort of things are allowed for cosmology
        # TODO Consult Brian or docs on cosmo parameters
        pass

    def image(self, **kwargs):
        pass

    def noise(self, number_noise_sources=None, noise_source=None, **kwargs):
        pass

    def geometry(self,
                 plane=None,
                 red_shift=None,
                 **kwargs):
        # Sets the parameters for a specific plane
        pass

    def species(self, plane=None, species_name=None, profile=None):
        # Adds a species to a particular plane
        # Uses the _species_* helper functions to set them based on the passed profile
        pass

    def _species_light_profiles(self):
        pass

    def _species_mass_profiles(self):
        pass

    def _species_shear_profiles(self):
        pass

    def _species_special(self):
        pass

    def _format_dict(self):
        # Add all these parameters to the dictionary, used when calling the reader
        # Will work with all these default ones,
        # and let it be extended into different types of survey
        raise NotImplementedError

    def save(self, output_path):
        # Save results as {preformed config that can be re-run by feeding it to the io}
        pass

    def __call__(self, *args, **kwargs):
        # Put this whole thing together into the dictionary that can be used by lenstronomy
        return self._format_dict()
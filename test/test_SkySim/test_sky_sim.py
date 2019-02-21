from deeplenstronomy.SkySim.sky_sim import SkySim
from deeplenstronomy.PopSim.population_model import PopulationModel
import deeplenstronomy.SkySurveyModel.sky_survey as sky_survey


class TestSkySim(object):

    def setup(self):
        from astropy.cosmology import default_cosmology
        cosmo = default_cosmology.get()
        lensPop = PopulationModel()
        skySurvey = sky_survey.survey_constructor(survey_name='LSST', band_name='g')
        self.skySim = SkySim(lensPop, skySurvey, cosmo)

    def test_draw_lens_systems(self):
        num_images = 100
        image_list = self.skySim.draw_lens_systems(num=num_images, cutout_size=52, seed=41)
        assert len(image_list) == num_images
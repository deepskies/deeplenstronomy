import numpy as np
from deeplenstronomy.SkySurveyModel.sky_survey import calculate_background_noise


class TestSkySurvey(object):

    def setup(self):
        pass

    def test_calculate_background_noise(self):
        np.random.seed(0)
        test_image = np.random.normal(0,10,100)
        bg_noise = calculate_background_noise(test_image)
        assert abs(bg_noise - 10.0941191653) < 0.000000001


if __name__ == '__main__':
    pytest.main()


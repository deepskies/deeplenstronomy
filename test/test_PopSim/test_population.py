from deeplenstronomy.PopSim.population import Population
import pytest


class TestPopulation(object):

    def setup(self):
        pass

    def test_draw_model(self):

        pop = Population()
        kwargs_params, kwargs_model = pop.draw_model(with_lens_light=True, with_quasar=True)
        assert kwargs_model['lens_model_list'][0] == 'SIE'

    def test_draw_physical_model(self):

        pop = Population()
        kwargs_params, kwargs_model = pop.draw_model(with_lens_light=True, with_quasar=True, mode='complex')
        assert kwargs_model['lens_model_list'][0] == 'SIE'

if __name__ == '__main__':
    pytest.main()

from deeplenstronomy.InputReader.input_reader import InputReader

class SimulationInput(InputReader): 
    def __init__(self, configuration: dict = None):
        super().__init__(configuration)

    def _format_dict(self):
        pass 
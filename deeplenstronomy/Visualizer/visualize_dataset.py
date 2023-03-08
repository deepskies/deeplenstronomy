from deeplenstronomy.Visualizer.visualizer import Visualizer

class VisualizeDataset(Visualizer): 
    def __init__(self, n_subplots: int = None, figure_size: tuple = None):
        super().__init__(n_subplots, figure_size)

    def plot(self): 
        pass
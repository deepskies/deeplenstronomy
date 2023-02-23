from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Visualizer(ABC):
    def __init__(self, n_subplots: int = None, figure_size: tuple = None):
        plt.cla()
        plt.clf()
        # TODO logic on the n_subplots
        self.figure = plt.subplots(1, 1, figsize=figure_size)

    def plot(self):
        raise NotImplementedError

    def axes(self,
             x_label: str = '', x_ticks: list = None,
             y_label: str = '',  y_ticks: list = None,
             z_label: str = '', z_ticks: list = None
             ):
        pass

    def title(self, title="default"):
        pass

    def save(self, save_path):
        pass

    def __call__(self, *args, **kwargs):
        pass

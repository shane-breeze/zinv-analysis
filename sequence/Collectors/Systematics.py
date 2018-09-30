import os
import operator

from utils.Histogramming import Histograms
from Histogrammer import Config, HistReader, HistCollector

SystematicsReader = HistReader

class SystematicsCollector(HistCollector):
    def draw(self, histograms):
        return []

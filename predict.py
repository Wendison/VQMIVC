import json
import os
import tempfile
import zipfile
from pathlib import Path

import cog
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import tensorflow as tf
import yaml
from tensorflow_tts.inference import AutoConfig, AutoProcessor, TFAutoModel


class Predictor(cog.Predictor):
    def setup(self):
        """Load the fastspeech and melgan models"""


    @cog.input("input", type=str, help="String to be converted to speech audio")

    def predict(self, input, speaker_id, speed_ratio, f0_ratio, energy_ratio):
        """Compute TTS on input string"""
        # inference
        print ("colgione del cazzo sei patetico ")

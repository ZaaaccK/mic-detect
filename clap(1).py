import numpy as np


def isClap(audio, threshold):
    rms = np.sqrt(np.mean(audio**2))
    return 1 if rms > threshold else 0

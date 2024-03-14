import convert.MidiConvertToNumPy as conv
import numpy as np
datasets = "data/JazzMidi/"

converter = conv.MidiConvertToNumPy(datasets + "55Dive.mid")
np.set_printoptions(precision=2, suppress=True)
data = converter.convert()
print(data[0])

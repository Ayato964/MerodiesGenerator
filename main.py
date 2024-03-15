import convert.MidiConvertToNumPy as midiNum
import convert.ChangingKey as changeKey
import numpy as np
datasets = "data/JazzMidi/"

midi_data = changeKey.ChangingKey(datasets + "55Dive.mid").convert()

converter = midiNum.MidiConvertToNumPy(datasets + "55Dive.mid", data=midi_data)
numpy_data = converter.convert()
np.set_printoptions(precision=2, suppress=True)
print(numpy_data[0])

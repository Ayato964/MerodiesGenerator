import convert.MidiConvertToNumPy as midiNum
import convert.ChangingKey as changeKey
import numpy as np
datasets = "data/JazzMidi/"


converter = midiNum.MidiConvertToNumPy(datasets + "55Dive.mid")
numpy_data = converter.convert()
change = changeKey.ChangingKey(datasets + "55Dive.mid", numpy_data)
change.set_convert_key("C")
numpy_data = change.convert()

np.set_printoptions(precision=2, suppress=True)
print(numpy_data[0])


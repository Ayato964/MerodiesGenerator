import convert.MidiConvertToNumPy as midiNum
import convert.ChangingKey as changeKey
import numpy as np
import os

datasets = "data/JazzMidi/"
out = "output/np/JazzMidi/"
files = os.listdir(datasets)
for file in files:
    converter = midiNum.MidiConvertToNumPy(datasets + "/" + file)
    if not converter.is_error:
        change = changeKey.ChangingKey(datasets + "/" + file, converter.convert())
        change.set_convert_key("C")
        numpy_data = change.convert()
        np.savez(out + file, *numpy_data)

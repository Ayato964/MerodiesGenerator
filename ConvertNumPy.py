import convert.ConvertMidi as cm
import os

"""
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



"""

datasets = "data/JazzMidi/"
files = os.listdir(datasets)
con = cm.ConvertNumPy(datasets + files[0], cm.ConvertProperties().change_key("C"))
con.convert()
np_notes = con.get_np_notes()

print(np_notes.size)

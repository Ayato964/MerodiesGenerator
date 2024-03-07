import convert.MidiConvertToNumPy as conv
datasets = "data/JazzMidi/"

converter = conv.MidiConvertToNumPy(datasets + "55Dive.mid")
converter.convert(range(0, 8))

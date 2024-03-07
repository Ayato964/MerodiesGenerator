import pretty_midi
import pretty_midi as midi
import numpy

class MidiConvertToNumPy:
    directory = ""
    midi_data = midi.PrettyMIDI()

    def __init__(self, dir):
        self.directory = dir
        self.midi_data = midi.PrettyMIDI(dir)

    def convert(self, prog):
        prog_midi = self.getProgramMidi(prog)
        write = midi.PrettyMIDI()
        write.instruments.append(prog_midi)
        write.write("temp.mid")
        print(prog_midi.notes[60].velocity)

    def getProgramMidi(self, prog):
        # print(self.directory + "  " + str(prog[0]))
        for inst in self.midi_data.instruments:
            if inst.program in prog:
                new_inst = pretty_midi.Instrument(program=inst.program)
                new_inst.notes = inst.notes
                return new_inst
        return None

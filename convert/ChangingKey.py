import array

import convert.AbstractConverter
from convert import AbstractConverter
import numpy
import pretty_midi as midi


class ChangingKey(AbstractConverter.AbstractConverter):
    key = -1

    def convert(self):
        return self.midi_data
        pass

    def __init__(self, directory, data=None):
        super().__init__(directory, data)
        #self.set_key()

    def set_key(self):
        inst = self.midi_data.instruments
        pitches = numpy.zeros(12)

        for ins in inst:
            self.get_frequency_midi(ins, pitches)
        return pitches

    @staticmethod
    def get_frequency_midi(inst, pitches):
        for note in inst.notes:
            r_pitch = AbstractConverter.get_root_pitch(note.pitch)
            pitches[r_pitch] += 1

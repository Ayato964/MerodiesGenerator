from convert import AbstractConverter
import numpy
import pretty_midi as midi
import music21 as m21


class ChangingKey(AbstractConverter.AbstractConverter):
    key = None
    key_type = None
    c_key = None
    np_data = []

    def convert(self):
        number_key = ChangingKey.get_key_number(self.key.tonic.name)
        number_c_key = ChangingKey.get_key_number(self.c_key)
        num = abs(number_key-number_c_key)
        for i in range(len(self.np_data)):
            for c in range(len(self.np_data[i])):
                self.np_data[i][c][0] -= num
                self.np_data[i][c][3] += 12
                self.np_data[i][c][3] = AbstractConverter.get_root_pitch(self.np_data[i][c][3] - num)

        return self.np_data

    def set_convert_key(self, c):
        self.c_key = c

    def __init__(self, directory, np_data, data=None):
        super().__init__(directory, data)
        if not np_data is None:
            self.np_data = np_data
        mid_score = m21.converter.parse(directory)
        self.key = mid_score.analyze("key")
        print(f"{directory}'s key is {self.key.tonic.name} {self.key.mode}")
        if self.key.mode == "minor":
            self.key_type = 1
        else:
            self.key_type = 0

    @staticmethod
    def get_key_number(key):
        if key == "C":
            return 0
        if key == "C#" or key == "D-":
            return 1
        if key == "D":
            return 2
        if key == "D#" or key == "E-":
            return 3
        if key == "E":
            return 4
        if key == "F":
            return 5
        if key == "F#" or key == "G-":
            return 6
        if key == "G":
            return 7
        if key == "G#" or key == "A-":
            return 8
        if key == "A":
            return 9
        if key == "A#" or key == "B-":
            return 10
        if key == "B":
            return 11

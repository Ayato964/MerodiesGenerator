from abc import ABC, abstractmethod
import pretty_midi as midi


def get_root_pitch(pitch):
    return pitch % 12


class AbstractConverter(ABC):
    directory = ""
    midi_data = midi.PrettyMIDI()

    def __init__(self, directory, data):
        self.directory = directory
        if data is None:
            self.midi_data = midi.PrettyMIDI(directory)
        else:
            self.midi_data = data

    @abstractmethod
    def convert(self):
        pass

import string
from abc import abstractmethod
import pretty_midi as midi
from util.ArrayList import ArrayList


def _get_midi_datasets(directory):
    try:
        return midi.PrettyMIDI(directory)
    except OSError:
        return None


class _AbstractConvert:
    @abstractmethod
    def convert(self, midi_data):
        pass


class ConvList:
    _conv_list = []

    def __init__(self):
        self._conv_list = ArrayList[_AbstractConvert]()

    def change_key(self):
        self._conv_list.add(_ConvertChangeKey())
        return self

    def convert(self, midi_data):
        for i in range(self._conv_list.size()):
            self._conv_list.get(i).convert(midi_data)


class ConvertNumPy:
    directory = ""
    midi_data = midi.PrettyMIDI()

    def __init__(self, directory: str, conv_list: ConvList):
        self.directory = directory
        self.conv = conv_list
        self.midi_data = _get_midi_datasets(directory)

        pass

    def convert(self):
        self.conv.convert(self.midi_data)


class _ConvertChangeKey(_AbstractConvert):
    def convert(self, midi_data):
        pass

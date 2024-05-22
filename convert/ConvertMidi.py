import string
from abc import abstractmethod
import os
import numpy
import pretty_midi as midi
from pretty_midi import PrettyMIDI
import music21 as m21
import numpy as np
from util.ArrayList import ArrayList

PITCH = 0
VELOCITY = 1
DURATION_INT = 2
DURATION_FEW = 3
BEGIN_TIME_INT = 4
BEGIN_TIME_FEW = 5


def _get_midi_datasets(directory):
    try:
        return midi.PrettyMIDI(directory)
    except OSError:
        return None


class _AbstractConvert:
    @abstractmethod
    def convert(self, directory: str, midi_data: PrettyMIDI):
        pass


class ConvertProperties:
    _conv_list = []

    def __init__(self):
        self._conv_list = ArrayList[_AbstractConvert]()

    def change_key(self, key: str):
        self._conv_list.add(_ConvertChangeKey(key))
        return self

    def convert(self, directory, midi_data):
        for i in range(self._conv_list.size()):
            self._conv_list.get(i).convert(directory, midi_data)


class ConvertNumPy:
    directory = ""
    midi_data = midi.PrettyMIDI()
    np_note = []
    isError = False

    def __init__(self, directory: str, conv_list: ConvertProperties):
        self.directory = directory
        self.conv = conv_list
        try:
            self.midi_data = _get_midi_datasets(directory)
        except OSError:
            self.isError = True

        pass

    # [音高, 強さ, 長さ(整数), 長さ(小数点以下),  前のノーツからの開始時間（整数）,　前のノーツからの開始時間(少数以下), 泊のタイミング(4泊), 泊のタイミング(8泊), ルート音]
    def convert(self):
        if not self.isError:
            self.conv.convert(self.directory, self.midi_data)
        else:
            print("エラーが発生し、読み込めません。")

        np_notes = np.array([[-1, -1, -1, -1, -1, -1, -1]])
        for inst in self.midi_data.instruments:
            before: midi.Note = midi.Note(-1, -1, -1, -1)
            if not inst.is_drum and inst.program in range(1, 10):
                for note in inst.notes:
                    pitch = int(note.pitch)  # 音高

                    velocity = int(note.velocity)  # 強さ

                    duration_int, duration_few = self.split_float_to_ints(note.get_duration())  # 長さ

                    begin_time_int, begin_time_few = \
                        self.split_float_to_ints(self.get_begin_time(before, note))  # 前のノーツからの開始時間

                    root_note = self.get_root_note(note.start)

                    np_notes = np.vstack([np_notes, [int(pitch), int(velocity), duration_int, duration_few,
                                                     begin_time_int, begin_time_few, root_note]])
                    before = note  # 次のループまでnoteを保持
                np_notes = np.vstack([np_notes, [-1, -1, -1, -1, -1, -1, -1]])
        self.np_note = np_notes

    def save(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # ルートディレクトリまでの相対パスを計算
        project_root = os.path.abspath(os.path.join(current_dir, '..'))

        # ルートディレクトリからoutディレクトリへのパスを生成
        out_directory = os.path.join(project_root, 'output')

        split_direc = self.directory.split("/")

        np.savez(out_directory + "/np/" + split_direc[1] + "/" + split_direc[-1], *self.np_note)

    def get_np_notes(self):
        return self.np_note

    @staticmethod
    def get_root_pitch(pitch):
        return pitch % 12

    def get_root_note(self, start: float) -> int:
        bass = self.get_bass()
        if bass is None:
            return -1
        else:
            measure = self.get_measure(start)
            for note in bass.notes:
                if note.start >= self.get_measure_sec() * (measure - 1):
                    return self.get_root_pitch(note.pitch)

            return -1

    def get_bass(self):
        for inst in self.midi_data.instruments:
            if inst.program in range(33, 41):
                return inst

        return None

    def get_measure(self, start):
        measure_sec = self.get_measure_sec()  # 一小節当たりの秒数
        return start / measure_sec + 1

    def get_measure_sec(self):
        time, tempo = self.midi_data.get_tempo_changes()
        s = 60 / tempo[0]  # 一泊当たりの秒数
        return 4 * s

    @staticmethod
    def get_begin_time(before: midi.Note, notes: midi.Note) -> float:
        return notes.start - before.end if before.start != -1000 else -1

    @staticmethod
    def split_float_to_ints(num):
        if num == -1:
            return -1
        else:
            # 整数部を取得
            integer_part = int(num)

            # 少数部を取得し、整数部に変換
            decimal_part_as_str = str(num).split('.')[1]
            decimal_part_as_int = int(decimal_part_as_str)

            return integer_part, decimal_part_as_int


class _ConvertChangeKey(_AbstractConvert):

    def __init__(self, conv_key: str):
        self.conv_key_number = self.get_key_number(conv_key)

    def convert(self, directory: str, midi_data: PrettyMIDI):
        score = m21.converter.parse(directory)
        key = score.analyze("key").tonic.name
        now_key_number = self.get_key_number(key)
        transpose_number = abs(now_key_number - self.conv_key_number)
        for inst in midi_data.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    note.pitch -= transpose_number
        pass

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

import pretty_midi
import pretty_midi as midi
import numpy as np
from convert import AbstractConverter


class MidiConvertToNumPy(AbstractConverter.AbstractConverter):
    def __init__(self, directory, data=None):
        super().__init__(directory, data)

    def convert(self):
        print(f"{self.directory} converting....")
        prog_inst = self.get_melody_midi()
        np_notes = []
        i = 0
        for prog_inst in prog_inst:
            if prog_inst is not None:
                #write = midi.PrettyMIDI()
                #write.instruments.append(prog_inst)
                #write.write(f"output/temp{i}.mid")
                new_notes = self.get_convert_note_to_numpy(prog_inst.notes)
                np_notes.append(np.array(new_notes))
                i += 1
            else:
                print("Not Notes")
                return None
        return np_notes

    def get_convert_note_to_numpy(self, notes):
        np_note = []
        for note in notes:
            start = note.start
            end = note.end
            duration = end - start

            np_note.append([int(note.pitch), int(note.velocity), duration, int(self.get_root_for_note(start))])
        return np_note  # [音高、強さ、長さ, ルート]の１次元配列を作成

    def get_root_for_note(self, start):
        bass = self.get_bass()
        if bass is None:
            return -1
        else:
            measure = self.get_measure(start)
            for note in bass.notes:
                if note.start >= self.get_measure_sec() * (measure - 1):
                    return AbstractConverter.get_root_pitch(note.pitch)

        return -1

    def get_measure(self, start):
        measure_sec = self.get_measure_sec()  # 一小節当たりの秒数
        return start / measure_sec + 1

    def get_measure_sec(self):
        time, tempo = self.midi_data.get_tempo_changes()
        s = 60 / tempo[0]  # 一泊当たりの秒数
        return 4 * s

    def get_melody_midi(self):
        new_instruments = []
        for inst in self.midi_data.instruments:
            if not inst.is_drum and inst.program not in range(33, 41):
                new_inst = pretty_midi.Instrument(program=inst.program)
                new_inst.notes = inst.notes
                new_instruments.append(new_inst)
        return new_instruments

    def get_bass(self):
        for inst in self.midi_data.instruments:
            if inst.program in range(33, 41):
                return inst

        return None

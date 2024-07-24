import numpy as np
import pretty_midi as pm


def sum_begin_time(a_int: int, a_few: int):
    sum_str = f"{a_int}.{a_few}"
    sum_float = float(sum_str)
    return sum_float


def convert(seq: list) -> pm:
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=3, is_drum=False)
    midi.instruments.append(inst)
    before_time = 0
    for note in seq:
        start = before_time + sum_begin_time(note[4], note[5])
        inst.notes.append(pm.Note(pitch=min(note[0], 127), velocity=min(note[1], 127), start=start, end=start + sum_begin_time(note[2], note[3])))
        before_time = start
        pass

    return midi



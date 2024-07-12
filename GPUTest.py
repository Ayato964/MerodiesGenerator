import transformer.AyatoTransFormer as atf
import torch
import pretty_midi as pm
model = atf.AyatoModel()

cuda_ava = torch.cuda.is_available()
print(cuda_ava)
print(torch.cuda.device_count())
print(torch.__version__)

N: int = 1000

midi = pm.PrettyMIDI()
inst = pm.Instrument(1)
bass = pm.Instrument(35)
midi.instruments.append(inst)
midi.instruments.append(bass)
for count in range(N):
    start = 16 * count
    inst.notes.append(pm.Note(100, 64, 0 + start, 2 + start))
    inst.notes.append(pm.Note(100, 66, 2 + start, 3 + start))
    inst.notes.append(pm.Note(100, 68, 3 + start, 4 + start))
    inst.notes.append(pm.Note(100, 69, 4 + start, 5 + start))
    inst.notes.append(pm.Note(100, 71, 5 + start, 6 + start))
    inst.notes.append(pm.Note(100, 73, 6 + start, 7 + start))
    inst.notes.append(pm.Note(100, 75, 7 + start, 8 + start))
    inst.notes.append(pm.Note(100, 76, 8 + start, 9 + start))

    inst.notes.append(pm.Note(100, 76, 9 + start, 10+ start))
    inst.notes.append(pm.Note(100, 75, 10 + start, 11+start))
    inst.notes.append(pm.Note(100, 73, 11 + start, 12+start))
    inst.notes.append(pm.Note(100, 71, 12 + start, 13+start))
    inst.notes.append(pm.Note(100, 69, 13 + start, 14+start))
    inst.notes.append(pm.Note(100, 68, 14 + start, 15+start))
    inst.notes.append(pm.Note(100, 66, 15 + start, 16+start))
    inst.notes.append(pm.Note(100, 64, 16 + start, 17+start))

    bass.notes.append(pm.Note(100, 64 - 12, 0 + start, 2 + start))
    bass.notes.append(pm.Note(100, 64 - 12, 2 + start, 3 + start))
    bass.notes.append(pm.Note(100, 64 - 12, 3 + start, 4 + start))
    bass.notes.append(pm.Note(100, 64 - 12, 4 + start, 5 + start))
    bass.notes.append(pm.Note(100, 64 - 12, 5 + start, 6 + start))
    bass.notes.append(pm.Note(100, 64 - 12, 6 + start, 7 + start))
    bass.notes.append(pm.Note(100, 64 - 12, 7 + start, 8 + start))
    bass.notes.append(pm.Note(100, 64 - 12, 8 + start, 9 + start))

    bass.notes.append(pm.Note(100, 64 - 12, 9 + start, 10+ start))
    bass.notes.append(pm.Note(100, 64 - 12, 10 + start, 11+start))
    bass.notes.append(pm.Note(100, 64 - 12, 11 + start, 12+start))
    bass.notes.append(pm.Note(100, 64 - 12, 12 + start, 13+start))
    bass.notes.append(pm.Note(100, 64 - 12, 13 + start, 14+start))
    bass.notes.append(pm.Note(100, 64 - 12, 14 + start, 15+start))
    bass.notes.append(pm.Note(100, 64 - 12, 15 + start, 16+start))
    bass.notes.append(pm.Note(100, 64 - 12, 16 + start, 17+start))


print(len(midi.instruments[0].notes))

midi.write("data/test/test.mid")

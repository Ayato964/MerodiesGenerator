import transformer.Generate as generate
import torch
from transformer.AyatoTransFormer import AyatoModel
import convert.ConvertMidi as cm
import pretty_midi as pm
from convert import ConvertAyaNodeToMidi as nm
import numpy as np
model_directory = "out/model/"
model = AyatoModel(
    d_model=32,
    dim_feedforward=512,
    trans_layer=3,
    position_length=512,
)
model.load_state_dict(torch.load(model_directory + "AyatoModel.TEST_3.5490127251066013.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("HEEEE")
# メロディ生成の実行
#np_notes = np.load("out/np/test/test.npz")

np_notes = np.load("out/np/datasets/55Dive.npz")

# 生成されたシーケンスの確認

seed = []
for i in range(20):
    seed.append(np_notes[f'arr_{i}'])
print(f"seed is {len(seed)} and {seed[1]}")

gene = model.generate(seed[0:3], max_length=3)

print(gene)

midi: pm.PrettyMIDI = nm.convert(gene[1:])

print(midi.instruments[0].notes[0])

midi.write("out/generated/test.mid")

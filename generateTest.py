import json

import constants
import transformer.Generate as generate
import torch
from transformer.AyatoTransFormer import AyatoModel
import convert.ConvertMidi as cm
import pretty_midi as pm
from convert import ConvertAyaNodeToMidi as nm
import numpy as np
from transformer.tokenizer import Tokenizer

model_directory = "out/model/"

tokenizer = Tokenizer("out/vocab/vocab_list.json")

model = AyatoModel(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    dim_feedforward=4700,
    trans_layer=3,
    position_length=4700,
    dropout=0.4
)
model.load_state_dict(torch.load("out/model/AyatoModel.TEST_0.15547503530979156.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("HEEEE")
# メロディ生成の実行
#np_notes = np.load("out/np/test/test.npz")


gene = model.generate(tokenizer.get(constants.START_SEQ_TOKEN), max_length=3)

print(gene)
'''
midi: pm.PrettyMIDI = nm.convert(gene[1:])

print(midi.instruments[0].notes[0])

midi.write("out/generated/test.mid")
'''
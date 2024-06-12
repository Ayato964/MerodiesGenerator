"""
 このクラスでは、main.pyで生成されたモデルをロードし、そのモデルのテストを行っている。
　ここで行っているテストとは、シードとなる前処理されたデータセットをモデルに読み込み、その続きを生成させるテストである。
"""
import torch
import transformer.AyatoTransFormer as atf
import transformer.Generate as generate
import random
import numpy as np
import torch
from transformer.AyatoTransFormer import AyatoModel
import convert.ConvertMidi as cm
model_directory = "out/model/"


# モデルのロード
model_directory = "out/model/"
model = AyatoModel()
model.load_state_dict(torch.load(model_directory + "JazzAI.0.2.0_20240612.pth"))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# メロディ生成の実行
direct = "data/JazzMidi/2ndMovementOfSinisterFootwear.mid"
conv: cm.ConvertNumPy = cm.ConvertNumPy(direct, cm.ConvertProperties().change_key("C"))
conv.convert()
np_notes = conv.get_np_notes()[0:10]
print("Initial notes:", np_notes)

decorder = generate.CreatingMelodiesContinuation()

output_melodies = decorder.generate(np_notes, 10, model)  # 10ステップのメロディ生成
print("Generated melodies:", output_melodies)

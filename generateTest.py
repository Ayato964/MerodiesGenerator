"""
 このクラスでは、main.pyで生成されたモデルをロードし、そのモデルのテストを行っている。
　ここで行っているテストとは、シードとなる前処理されたデータセットをモデルに読み込み、その続きを生成させるテストである。
"""
import transformer.Generate as generate
import torch
from transformer.AyatoTransFormer import AyatoModel
import convert.ConvertMidi as cm


model_directory = "out/model/"
model = AyatoModel()
model.load_state_dict(torch.load(model_directory + "JazzAI.0.4.0_20240619.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# メロディ生成の実行
direct = "data/JazzMidi/2ndMovementOfSinisterFootwear.mid"
conv: cm.ConvertNumPy = cm.ConvertNumPy(direct, cm.ConvertProperties().change_key("C").sort())
conv.convert()
np_notes = conv.get_np_notes()[0:10]

gene = model.generate(np_notes[1], max_length=50)

print(gene)

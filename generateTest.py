import transformer.Generate as generate
import torch
from transformer.AyatoTransFormer import AyatoModel
import convert.ConvertMidi as cm

model_directory = "out/model/"
model = AyatoModel()
model.load_state_dict(torch.load(model_directory + "JazzAI.0.4.3.1_20240703.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# メロディ生成の実行
direct = "data/JazzMidi/2ndMovementOfSinisterFootwear.mid"
conv = cm.ConvertNumPy(direct, cm.ConvertProperties().change_key("C").sort())
conv.convert()
np_notes = conv.get_np_notes()[0:10]

# 生成されたシーケンスの確認
seed = np_notes[1]
gene = model.generate(seed, max_length=1)

print(gene)
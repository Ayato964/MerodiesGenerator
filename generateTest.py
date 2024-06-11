"""
 このクラスでは、main.pyで生成されたモデルをロードし、そのモデルのテストを行っている。
　ここで行っているテストとは、シードとなる前処理されたデータセットをモデルに読み込み、その続きを生成させるテストである。
"""
import torch
import transformer.AyatoTransFormer as atf
import transformer.Generate
import convert.ConvertMidi as cm
model_directory = "output/model/"

model = atf.AyatoModel()
model.load_state_dict(torch.load(model_directory + "JazzAI.0.1.0_20240523.pth"))
model.eval()

direct = "data/JazzMidi/2ndMovementOfSinisterFootwear.mid"

conv: cm.ConvertNumPy = cm.ConvertNumPy(direct, cm.ConvertProperties().change_key("C"))
conv.convert()
np_notes = conv.get_np_notes()[0:10]
print(np_notes)
decorder = transformer.Generate.CreatingMelodiesContinuation()
output_melodies = decorder.generate(np_notes, 1, model)

print(output_melodies)

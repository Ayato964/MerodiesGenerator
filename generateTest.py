"""
 このクラスでは、main.pyで生成されたモデルをロードし、そのモデルのテストを行っている。
　ここで行っているテストとは、シードとなる前処理されたデータセットをモデルに読み込み、その続きを生成させるテストである。
"""
import torch
import transformer.AyatoTransFormer as atf
import convert.MidiConvertToNumPy as midiNum
import convert.ChangingKey as changeKey
import transformer.Generate

model_directory = "output/model/"

model = atf.AyatoModel()
model.load_state_dict(torch.load(model_directory + "JazzAI.0.1.0_20240324.pth"))
model.eval()

test_np_melody = midiNum.MidiConvertToNumPy("data/JazzMidi/BehindClosedDoors.mid").convert()
change = changeKey.ChangingKey("data/JazzMidi/BehindClosedDoors.mid", np_data=test_np_melody)
change.set_convert_key("C")
test_np_melody = change.convert()
test_np_melody = test_np_melody[0][0:20]

decorder = transformer.Generate.CreatingMelodiesContinuation()
output_melodies = decorder.generate(test_np_melody, 50, model)

print(output_melodies)

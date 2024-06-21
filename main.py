"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import transformer.AyatoTransFormer as atf
import os
import torch
import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model_version = "0.4.0"
today_date = datetime.date.today().strftime('%Y%m%d')

print(f"ToDay is{datetime.date.today()}! start generating JazzAI.{model_version}_{today_date}")

directory = "out/np/JazzMidi/"
datasets = os.listdir(directory)
train_data = atf.set_train_data(directory, datasets)  # 前処理されたデータをTransformerのデータセットクラスに変換する

model = atf.train(train_data, 1)  # 20エポック分機械学習を行う。

torch.save(model.state_dict(), f"out/model/JazzAI.{model_version}_{today_date}.pth")  # できたモデルをセーブする

"""
このクラスでは、ConvertNumPy.py等で変換された前処理したデータセットを用い、機械学習を行うクラスである。

"""
import transformer.AyatoTransFormer as atf
import os
import torch
import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model_version = "TEST"
today_date = datetime.date.today().strftime('%Y%m%d')

print(f"ToDay is{datetime.date.today()}! start generating AyatoModel.{model_version}_{today_date}")

#directory = "out/np/test/"
directory = "out/np/datasets/"
datasets = os.listdir(directory)
train_data = atf.set_train_data(directory, datasets)  # 前処理されたデータをTransformerのデータセットクラスに変換する

model, loss = atf.train(train_data, 30,
                        d_model=32,
                        dim_feedforward=512,
                        trans_layer=3,
                        position_length=512,
                        dropout=0.4
                        )  # 20エポック分機械学習を行う。

torch.save(model.state_dict(), f"out/model/AyatoModel.{model_version}_{loss}.pth")  # できたモデルをセーブする



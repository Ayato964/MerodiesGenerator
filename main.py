import transformer.AyatoTransFormer as atf
import os
import torch
import datetime
from transformer.AyatoTransFormer import AyatoModel
model_version = "0.1.0"
today_date = datetime.date.today().strftime('%Y%m%d')

print(f"ToDay is{datetime.date.today()}! start generating JazzAI.{model_version}_{today_date}")

directory = "output/np/JazzMidi/"
datasets = os.listdir(directory)
train_data = atf.set_train_data(directory, datasets)

model = atf.train(train_data, 20)

torch.save(model.state_dict(), f"output/model/JazzAI.{model_version}_{today_date}.pth")

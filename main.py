import transformer.AyatoTransFormer as atf
import os

from transformer.AyatoTransFormer import AyatoModel

directory = "output/np/JazzMidi/"
datasets = os.listdir(directory)
train_data = atf.set_train_data(directory, datasets)

model = atf.train(train_data, 128)

print(train_data.get_data()[0])

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
from transformers import BertModel
import  numpy as np


def set_train_data(directory, datasets):
    print("generating TrainData.....")
    t_data = AyatoDataSet()
    for dataset in datasets:
        np_load_data = np.load(directory + dataset)
        for i in range(len(np_load_data)):
            t_data.add_data(np_load_data[f'arr_{i}'])

    return t_data


def train(ayato_dataset, num_epochs):
    loader = DataLoader(ayato_dataset, batch_size=16, shuffle=True)
    print("Creating Model....")
    model = AyatoModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Start training...")
    for epoch in range(num_epochs):
        print(f"epoch {epoch} start....")
        for batch in loader:
            input_ids, attention_mask, targets = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    return model


# Transformerの定義
class AyatoModel(nn.Module):

    def __init__(self):
        super(AyatoModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 4)  # 指定したNumPy配列の次元数

    def forward(self, input_ids, attention_mask):
       # print(input_ids, attention_mask)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        outputs = self.fc(pooler_output)  # 1次元に変更してから線形層に渡す
        return outputs


# 一つの楽曲の一つの楽器のNumPy配列のノートをTransFormerのデータセットとして変換する
class AyatoDataSet(Dataset):
    data = np.array([])

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
            sample = self.data[item]
            input_text = ' '.join(map(str, sample))
            encoding = self.tokenizer.encode_plus(input_text, add_special_tokens=True, padding='max_length',
                                                  max_length=128,
                                                  truncation=True, return_attention_mask=True)
            input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)
            target = torch.tensor(sample, dtype=torch.float32)
            return input_ids, attention_mask, target

    def add_data(self, numpy_data):
        if self.data.size == 0:
            self.data = numpy_data.astype(float)
        else:
            self.data = np.concatenate((self.data, numpy_data.astype(float)), axis=0)

    def get_data(self):
        return self.data


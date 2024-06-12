import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
from transformers import BertModel
import numpy as np

IS_DEBUG = False


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# デバイスを取得
def set_train_data(directory, datasets):
    if not IS_DEBUG:
        print("Generating TrainData.....")
        t_data = AyatoDataSet()
        for dataset in datasets:
            print(directory + dataset)
            np_load_data = np.load(directory + dataset)
            for i in range(len(np_load_data) - 1):  # 最後のデータを除く
                input_data = np_load_data[f'arr_{i}']
                target_data = np_load_data[f'arr_{i + 1}']
                t_data.add_data(input_data, target_data)
        return t_data
    else:
        np_load_data = np.load(directory + datasets[0])
        print(np_load_data[f'arr_{3}'])
    return None


def train(ayato_dataset, num_epochs):
    start = time.time()
    loader = DataLoader(ayato_dataset, batch_size=64, shuffle=True, pin_memory=False)
    print("Creating Model....")
    model = AyatoModel().to(get_device())
    criterion = nn.MSELoss().to(get_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Start training...")
    for epoch in range(num_epochs):
        print(f"epoch {epoch} start....")
        print(f"batch size :{len(loader)}")
        for batch in loader:
            batch_start = time.time()
            input_ids, attention_mask, targets = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    return model


device = get_device()


class AyatoModel(nn.Module):
    def __init__(self):
        super(AyatoModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 7)  # 出力次元を9に設定

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        outputs = self.fc(pooler_output)
        return outputs


class AyatoDataSet(Dataset):
    def __init__(self):
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        input_data = sample['input']
        target_data = sample['target']
        input_text = ' '.join(map(str, input_data))
        encoding = self.tokenizer.encode_plus(input_text, add_special_tokens=True, padding='max_length',
                                              max_length=128, truncation=True, return_attention_mask=True)
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long).to(device)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long).to(device)
        target = torch.tensor(target_data, dtype=torch.float).to(device)
        return input_ids, attention_mask, target

    def add_data(self, input_data, target_data):
        self.data.append({'input': input_data, 'target': target_data})

''''
# トレーニングデータの読み込みとトレーニングの実行
directory = 'your_directory_path/'
datasets = ['your_dataset_file.npz']
train_data = set_train_data(directory, datasets)
num_epochs = 10
trained_model = train(train_data, num_epochs)
'''


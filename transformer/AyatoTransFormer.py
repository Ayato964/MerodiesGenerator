import time

import numpy
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
from transformers import BertModel
import numpy as np

import constants
from .PositionalEncoding import PositionalEncoding


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
    model = AyatoModel().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Start training...")
    for epoch in range(num_epochs):
        print(f"epoch {epoch} start....")
        print(f"batch size :{len(loader)}")
        count = 0
        for batch in loader:
            batch_start = time.time()
            input_ids, attention_mask, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            for input in input_ids:
                outputs = model(input, attention_mask=attention_mask)
#                loss = criterion(outputs, targets.float())
#                loss.backward()
                optimizer.step()
            print(f"batch count:{count}")
            count += 1
       # print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    return model


device = get_device()


class AyatoModel(nn.Module):
    def __init__(self, trans_layer=6, num_heads=8, d_model=512, dim_feedforward=1024):
        super(AyatoModel, self).__init__()

        self.trans_layer = trans_layer
        self.num_heads = num_heads
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward

        #位置エンコーディングを作成
        self.positional: PositionalEncoding = PositionalEncoding(self.d_model, 0.1, 2048).to(device)
        #Transformerの設定
        self.transformer: nn.Transformer = nn.Transformer(d_model=self.d_model, nhead=num_heads,  #各種パラメーターの設計
                                                          num_encoder_layers=self.trans_layer, num_decoder_layers=0,
                                                          dropout=0.1, dim_feedforward=dim_feedforward).to(device)

        self.fc: nn.Linear = nn.Linear(self.d_model, constants.VOCAB_SIZE).to(device)  # 出力次元を7に設定

        self.embedding: nn.Embedding = nn.Embedding(constants.VOCAB_SIZE, self.d_model).to(device)

    def forward(self, inputs, attention_mask):
        mask = self.transformer.generate_square_subsequent_mask(inputs.shape[0]).to(device)

        inputs_em: nn.Embedding = self.embedding(inputs)
        inputs_pos = self.positional(inputs_em)
        outputs = self.transformer.encoder(inputs_pos)
        outputs = self.fc(outputs)

        return outputs


    def generate(self, input_seq, max_length=50):
        self.eval()

        generated_seq = np.array([input_seq])

        if max_length >= 1:
            input_tensor = torch.tensor(input_seq, dtype=torch.long).to(device).unsqueeze(1)  # シーケンスをテンソルに変換し、バッチ次元を追加
            attention_mask = torch.ones(input_tensor.shape, dtype=torch.long).to(device)  # 注意マスクを作成
            score = self(input_tensor, attention_mask)

            print(f"Score shape is {score.shape}")

            output_node: Tensor = torch.argmax(score, dim=-1).view(-1)
            print(score[0, 0, output_node[0]])
            output_node_copy = output_node.to('cpu')
            generated_seq = np.concatenate((generated_seq, [output_node_copy]), axis=0)

            print(f"output node:{output_node.squeeze().tolist()}")
            with torch.no_grad():# 勾配計算を無効化（推論モード）
                for _ in range(max_length - 1):
                    score = self(output_node, attention_mask)
                    output_node = self.get_node_by_score(score)
                    generated_seq = np.concatenate((generated_seq, [output_node]), axis=0)
                pass
        return generated_seq  # 生成されたシーケンスを返す


class AyatoDataSet(Dataset):
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        input_data = sample['input']
        target_data = sample['target']
        input_data = torch.tensor(input_data, dtype=torch.int).to(device)
        target = torch.tensor(target_data, dtype=torch.int).to(device)
        return input_data, 0, target

    def add_data(self, input_data, target_data):
        self.data.append({'input': input_data, 'target': target_data})

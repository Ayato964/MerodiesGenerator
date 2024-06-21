import time
import torch
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
    model = AyatoModel().to(get_device())
    criterion = nn.MSELoss().to(get_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Start training...")
    for epoch in range(num_epochs):
        print(f"epoch {epoch} start....")
        print(f"batch size :{len(loader)}")
        count = 0
        for batch in loader:
            batch_start = time.time()
            input_ids, attention_mask, targets = batch
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
        self.positional: PositionalEncoding = PositionalEncoding(self.d_model, 0.1, 2048)
        #Transformerの設定
        self.transformer: nn.Transformer = nn.Transformer(d_model=self.d_model, nhead=num_heads,  #各種パラメーターの設計
                                                          num_encoder_layers=self.trans_layer, num_decoder_layers=0,
                                                          dropout=0.1, dim_feedforward=dim_feedforward)


        self.fc = nn.Linear(self.d_model, constants.VOCAB_SIZE)  # 出力次元を7に設定

        self.embedding = nn.Embedding(constants.VOCAB_SIZE, self.d_model)

    def forward(self, inputs, attention_mask):
        mask = self.transformer.generate_square_subsequent_mask(inputs.shape[0]).to(device)

        # デバッグ用にインデックスの範囲を確認する
        #print(f"Inputs min: {inputs.min()}, max: {inputs.max()}")

        # 埋め込み層の定義時のnum_embeddingsと一致しているか確認
        #print(f"Embedding layer num_embeddings: {self.embedding.num_embeddings}")

        inputs_em: nn.Embedding = self.embedding(inputs)
        inputs_pos = self.positional(inputs_em)
        outputs = self.transformer.encoder(inputs_pos)
        outputs = self.fc(outputs)
        return outputs

    def generate(self, input_seq, max_length=50):
        self.eval()  # モデルを評価モードに切り替え
        generated_seq = list(input_seq)  # 生成シーケンスをリストとして初期化
        input_tensor = torch.tensor(input_seq, dtype=torch.long).to(device).unsqueeze(1)  # シーケンスをテンソルに変換し、バッチ次元を追加
        attention_mask = torch.ones(input_tensor.shape, dtype=torch.long).to(device)  # 注意マスクを作成

        with torch.no_grad():  # 勾配計算を無効化（推論モード）
            for _ in range(max_length):
                outputs = self(input_tensor, attention_mask)  # モデルの出力を取得
                next_token = outputs[-1, :, :].argmax(dim=-1)  # 最新のトークンのスコアから次のトークンを選択
                generated_seq.append(next_token.item())  # 生成シーケンスに次のトークンを追加
                input_tensor = torch.cat((input_tensor, next_token.unsqueeze(0)), dim=0)  # 新しいトークンを入力テンソルに追加

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

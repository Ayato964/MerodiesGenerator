import random
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
        print(f"Token size: {len(t_data.data)}")
        return t_data
    else:
        np_load_data = np.load(directory + datasets[0])
        print(np_load_data[f'arr_{3}'])
    return None


def train(ayato_dataset, num_epochs, trans_layer=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
          position_length=2048):
    start = time.time()
    loader = DataLoader(ayato_dataset, batch_size=512, shuffle=False, pin_memory=False)
    print("Creating Model....")
    model = AyatoModel(trans_layer=trans_layer, num_heads=num_heads,
                       d_model=d_model, dim_feedforward=dim_feedforward,
                       dropout=dropout, position_length=position_length).to(device)

    criterion = nn.CrossEntropyLoss()  # 損失関数を定義
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # オプティマイザを定義

    print("Start training...")
    loss_val = None
    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1} start....")
        print(f"batch size :{len(loader)}")
        count = 0
        loss = None
        model.train()
        for batch in loader:
            input_ids, attention_mask, targets = [x.to(device) for x in batch]

            optimizer.zero_grad()  # 勾配を初期化
            outputs = model(input_ids, attention_mask=attention_mask)

            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1).long()

            loss = criterion(outputs, targets)  # 損失を計算
            loss.backward()  # 逆伝播
            optimizer.step()  # オプティマイザを更新
            count += 1
        print(f"Epoch [{epoch+1}/{num_epochs}],  Loss: {loss.item():.4f}")
        loss_val = loss.item()
    return model, loss_val


device = get_device()


class AyatoModel(nn.Module):
    def __init__(self, trans_layer=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
                 position_length=2048):
        super(AyatoModel, self).__init__()

        self.dummy = DummyDecoder()
        self.trans_layer = trans_layer
        self.num_heads = num_heads
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        #位置エンコーディングを作成
        self.positional: PositionalEncoding = PositionalEncoding(self.d_model, dropout, position_length).to(device)
        #Transformerの設定
        self.transformer: nn.Transformer = nn.Transformer(d_model=self.d_model, nhead=num_heads,  #各種パラメーターの設計
                                                          num_encoder_layers=self.trans_layer, num_decoder_layers=0,
                                                          dropout=self.dropout, dim_feedforward=dim_feedforward,
                                                          custom_decoder=self.dummy
                                                          ).to(device)

        self.Wout = nn.Linear(self.d_model, constants.VOCAB_SIZE).to(device)

        self.embedding: nn.Embedding = nn.Embedding(constants.VOCAB_SIZE, self.d_model).to(device)

        self.softmax: nn.Softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, attention_mask):

        mask = self.transformer.generate_square_subsequent_mask(inputs.shape[1]).to(get_device())

        inputs_em: nn.Embedding = self.embedding(inputs)

        #print(f"inputs EM: {inputs_em.shape}")

        inputs_em = inputs_em.permute(1, 0, 2)

        #print(f"inputs EM Permute: {inputs_em.shape}")

        inputs_pos = self.positional(inputs_em)

        #print(f"positional inputs: {inputs_em.shape}")

        outputs = self.transformer(src=inputs_pos, tgt=inputs_pos, src_mask=mask, tgt_mask=mask)
        outputs = outputs.permute(1, 0, 2)
        score = self.Wout(outputs)

        return score


    def generate(self, input_seq, max_length=50):
        self.eval()
        generated_seq = torch.tensor(input_seq, dtype=torch.long).to(device)

        print(f"first{generated_seq}")

        with torch.set_grad_enabled(False):
            for i in range(max_length):
                score: torch.Tensor = self.softmax(self(generated_seq, 0))

                next_token_score: torch.Tensor = score[-1, :, :]
                next_token_score.flatten()

                dis = torch.distributions.categorical.Categorical(probs=next_token_score)
                output_node = dis.sample().unsqueeze(0)

                generated_seq = torch.cat((generated_seq, output_node), dim=0).to(device)

        #print(f"Finally:{generated_seq}")
        return generated_seq.tolist()  # 生成されたシーケンスを返す


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


class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask,tgt_key_padding_mask,memory_key_padding_mask, **kwargs):

        return memory
    pass

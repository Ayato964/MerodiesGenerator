import random
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from .tokenizer import convert_token
from .tokenizer import Tokenizer
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
        tokenizer = Tokenizer()
        t_data = AyatoDataSet(tokenizer)
        for dataset in datasets:
            print(directory + dataset)
            np_load_data = np.load(directory + dataset)
            train_data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
            train_next_data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
            for i in range(len(np_load_data) - 1):
                np_data = np.expand_dims(np_load_data[f'arr_{i}'], axis=0)
                np_next_data = np.expand_dims(np_load_data[f'arr_{i + 1}'], axis=0)

                train_data = np.concatenate((train_data, np_data), axis=0)
                train_next_data = np.concatenate((train_next_data, np_next_data), axis=0)

            t_data.add_data(train_data[1:].tolist(), train_next_data[1:].tolist())
        print(f"Token size: {sum(len(sub) for sub in t_data.musics_seq)}   Vocab size is : {tokenizer.vocab_size}")
        print("----------------------------------")
        print("Generate Padding...")
        t_data.set_padding()

        return t_data, tokenizer
    else:
        np_load_data = np.load(directory + datasets[0])
        print(np_load_data[f'arr_{3}'])
    return None


def train(ayato_dataset, vocab_size: int, num_epochs: int, trans_layer=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
          position_length=2048):
    loader = DataLoader(ayato_dataset, batch_size=64, shuffle=True, pin_memory=False)
    print("Creating Model....")
    model = AyatoModel(vocab_size=vocab_size, trans_layer=trans_layer, num_heads=num_heads,
                       d_model=d_model, dim_feedforward=dim_feedforward,
                       dropout=dropout, position_length=position_length).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 損失関数を定義
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # オプティマイザを定義

    print("Start training...")
    loss_val = None
    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1} start....")
        print(f"batch size :{len(loader)}")
        count = 0
        epoch_loss = 0.0

        model.train()
        for seq, tgt in loader: # seqにはbatch_size分の楽曲が入っている
            for i in range(len(seq)): #１曲を取り出す
                tune = seq[i].to(device)
                tgt_tune = tgt[i].to(device)

                output = model(tune[1:])

                # 出力とターゲットの形状を一致させるためにリシェイプ
                output = output.view(-1, output.size(-1))
                tgt_tune = tgt_tune[1:].view(-1)

                # 損失計算
                loss = criterion(output, tgt_tune)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                count += 1
                epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}],  Loss: {epoch_loss / count:.4f}")
        loss_val = epoch_loss / count
    return model, loss_val


device = get_device()


class AyatoModel(nn.Module):
    def __init__(self, vocab_size, trans_layer=6, num_heads=8, d_model=512, dim_feedforward=1024, dropout=0.1,
                 position_length=2048):
        super(AyatoModel, self).__init__()

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
                                                          custom_decoder=DummyDecoder(),
                                                          dropout=self.dropout, dim_feedforward=dim_feedforward,
                                                          ).to(device)
        print(vocab_size)
        self.Wout = nn.Linear(self.d_model, vocab_size).to(device)

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, self.d_model).to(device)

        self.softmax: nn.Softmax = nn.Softmax(dim=-1).to(device)

    def forward(self, inputs_seq, tgt_seq, input_mask, tgt_mask, padding_mask):

        inputs_em: nn.Embedding = self.embedding(inputs_seq)
        inputs_em.permute(1, 0, 2)
        inputs_pos: PositionalEncoding = self.positional(inputs_em)

        if tgt_seq is None:
            tgt_pos = inputs_pos
        else:
            tgt_em: nn.Embedding = self.embedding(tgt_seq)
            tgt_em.permute(1, 0, 2)
            tgt_pos: PositionalEncoding = self.positional(tgt_em)

        out: Tensor = self.transformer(inputs_pos, tgt_pos, input_mask, tgt_mask,
                                       src_key_padding_mask=padding_mask, tgt_key_padding_mask=padding_mask)
        score = self.Wout(out)
        return score

    def generate(self, input_seq, max_length=50):
        self.eval()
        generated_seq = torch.tensor(input_seq, dtype=torch.long).to(device)

        print(f"first{generated_seq}")

        with torch.set_grad_enabled(False):
            for i in range(max_length):
                rand = random.randint(0, 100)
                score: torch.Tensor = self.softmax(self(generated_seq))
                print(score.shape)
                next_token_score: torch.Tensor = score[-1, :, :]
                next_token_score.flatten()
                if rand <= 90:
                    dis = torch.distributions.categorical.Categorical(probs=next_token_score)
                    output_node = dis.sample().unsqueeze(0)
                else:
                    output_node = torch.argmax(next_token_score, dim=-1).unsqueeze(0)

                generated_seq = torch.cat((generated_seq, output_node), dim=0).to(device)

        #print(f"Finally:{generated_seq}")
        return generated_seq.tolist()  # 生成されたシーケンスを返す


class AyatoDataSet(Dataset):
    def __init__(self, tokenizer: Tokenizer):
        self.musics_seq = None
        self.tgt_seq = None
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.musics_seq)

    def __getitem__(self, item):
        return self.musics_seq[item], self.tgt_seq[item]

    def add_data(self, music_seq, tgt_seq):

        if self.musics_seq is None and self.tgt_seq is None:
            self.musics_seq = [convert_token(music_seq, self.tokenizer)]
            self.tgt_seq = [convert_token(tgt_seq, self.tokenizer)]
        else:
            self.musics_seq = self.musics_seq + [convert_token(music_seq, self.tokenizer)]
            self.tgt_seq = self.tgt_seq + [convert_token(tgt_seq, self.tokenizer)]

        print(self._get_shape(self.musics_seq), self._get_shape(self.tgt_seq))
        print(self.musics_seq[-1][5])
        pass

    def _get_shape(self, lst):
        if isinstance(lst, list):
            return [len(lst)] + self._get_shape(lst[0]) if lst else []
        return []

    def set_padding(self):
        self._padding(self.tgt_seq)
        self._padding(self.musics_seq)

        print(self._get_shape(self.musics_seq[-1]), self._get_shape(self.musics_seq[0]), self._get_shape(self.tgt_seq[-1]))
        pass

    def _padding(self, target: list):
        max_lengths = []
        for t in target:
            max_lengths.append(len(t))
        max_length = max(max_lengths)

        for t in target:
            if len(t) < max_length:
                for _ in range(max_length - len(t)):
                    t.append(self.tokenizer.get(constants.PADDING_TOKEN))
        pass


class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, **kwargs):
        return memory
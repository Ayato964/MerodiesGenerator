import random
import numpy as np
import torch
from transformers import BertTokenizer
from transformer.AyatoTransFormer import AyatoModel


class CreatingMelodiesContinuation:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def generate(self, input_melodies, generate_time, model):
        output_melodies = input_melodies
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(generate_time):
            input_text = ' '.join(map(str, output_melodies[-10:]))
            encoding = self.tokenizer.encode_plus(input_text, add_special_tokens=True, padding='max_length',
                                                  max_length=128, truncation=True, return_attention_mask=True)
            input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)

            model_output = model(input_ids, attention_mask=attention_mask).detach().cpu().numpy()
            choice_output = np.round(model_output[0]).astype(int)  # 四捨五入して整数に変換

            output_melodies = np.append(output_melodies, [choice_output], axis=0)

        return output_melodies

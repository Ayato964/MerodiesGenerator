import random
from abc import abstractmethod
import numpy as np
import torch
from transformer.AyatoTransFormer import AyatoModel, AyatoDataSet


class CreatingMelodiesContinuation:

    def generate(self, input_melodies, generate_time, model):
        output_melodies = input_melodies
        for _ in range(generate_time):
            tenser = torch.tensor(output_melodies, dtype=torch.long)
            model_output = model(tenser, attention_mask=None).detach()
            model_output_np = model_output.numpy()
            choice_output = np.array(random.choice(model_output_np))

            choice_output = choice_output.tolist()
            output_melodies = output_melodies.tolist()

            output_melodies = output_melodies + [choice_output]
            output_melodies = np.array(output_melodies, dtype=float)
        return output_melodies



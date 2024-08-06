import numpy as np
import json
import constants


class Tokenizer:
    def __init__(self, load_data: str = None):
        if load_data is None:
            self.vocab_size = 3
            self.tokens: dict = dict()
            self.tokens[constants.PADDING_TOKEN] = 0
            self.tokens[constants.START_SEQ_TOKEN] = 1
            self.tokens[constants.END_SEQ_TOKEN] = 2
        else:
            with open(load_data, 'r') as file:
                self.tokens: dict = json.load(file)
            self.vocab_size = len(self.tokens)

    def get(self, a: str) -> int:
        if a in self.tokens:
            return self.tokens[a]
        else:
            self.tokens[a] = self.vocab_size
            self.vocab_size += 1
            return self.tokens[a]

    def save(self):
        json_string = json.dumps(self.tokens)
        with open("out/vocab/vocab_list.json", 'w') as file:
            file.write(json_string)
    pass


def convert_token(data: list, tokenizer: Tokenizer) -> list:
    new_data: list = []
    is_first: bool = True
    for i in range(len(data)):
        if data[i] == [0, 0, 0, 0, 0, 0, 0, 0, 0] and is_first:
            new_data.append(tokenizer.get(constants.START_SEQ_TOKEN))
            is_first = False
        elif data[i] == [0, 0, 0, 0, 0, 0, 0, 0, 0]:
            new_data.append(tokenizer.get(constants.END_SEQ_TOKEN))
            is_first = True
        else:
            str_token = "_".join(map(str, data[i]))
            new_data.append(tokenizer.get(str_token))

    return new_data

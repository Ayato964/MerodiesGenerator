from convert import AbstractConverter
import numpy


class ChangingKey(AbstractConverter.AbstractConverter):
    key = -1

    def convert(self):
        pass

    def __init__(self, directory, data=None):
        super().__init__(directory, data)
        self.set_key()

    def set_key(self):
        pass

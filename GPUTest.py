import transformer.AyatoTransFormer as atf
import torch
model = atf.AyatoModel()

cuda_ava = torch.cuda.is_available()
print(cuda_ava)
print(torch.cuda.device_count())
print(torch.__version__)
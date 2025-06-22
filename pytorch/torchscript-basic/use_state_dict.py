from model import SimpleModel
import torch

model = SimpleModel()

model.load_state_dict(torch.load("state_dict.pth"))

print(model)

# 추론 모드 전환
model.eval()

print(model(torch.tensor([[1.1, 1.0]])).tolist())
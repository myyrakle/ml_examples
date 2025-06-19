import torch

model = torch.load("model_full.pth", weights_only=False)

print(model)

# 추론 모드 전환
model.eval()

print(model(torch.tensor([[1.1, 1.0]])).tolist())
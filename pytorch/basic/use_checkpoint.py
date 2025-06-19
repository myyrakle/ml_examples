import torch
import torch.optim as optim
from model import SimpleModel

model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

checkpoint = torch.load("checkpoint.pth")

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(model)
print(optimizer)
print(epoch)
print(loss)

# 추론 모드 전환
model.eval()

print(model(torch.tensor([[1.1, 1.0]])).tolist())
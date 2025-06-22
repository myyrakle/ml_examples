from model import SimpleModel
import torch

loaded_model = torch.jit.load("traced_model.pt")

print(loaded_model)

print(loaded_model(torch.tensor([[1.0, 1.0]])).tolist())
import trace
from model import SimpleModel
import torch

model = SimpleModel()

model.load_state_dict(torch.load("state_dict.pth"))

print(model)

scripted_model = torch.jit.script(model)

scripted_model.save("scripted_model.pt")
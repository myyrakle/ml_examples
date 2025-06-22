import trace
from model import SimpleModel
import torch

model = SimpleModel()

model.load_state_dict(torch.load("state_dict.pth"))

print(model)

# 추론 모드 전환
model.eval()

# Tracing 방식 - 예제 입력으로 모델 실행 경로 추적
example_input = torch.tensor([[1.0, 2.0]])
traced_model = torch.jit.trace(model, example_input)

traced_model.save("traced_model.pt")
from model import SimpleModel
import torch
import torch.nn as nn
import torch.optim as optim

model = SimpleModel()

print(model)

x = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
y = torch.tensor([[3.0, 5.0, 7.0], [5.0, 7.0, 9.0], [7.0, 9.0, 11.0], [9.0, 11.0, 13.0]])

criterion = nn.MSELoss()  # 평균제곱오차
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 확률적 경사하강법

# 10번 반복해서 학습 (epoch)
for epoch in range(100):
    optimizer.zero_grad()         # 기울기 초기화
    outputs = model(x)            # 모델에 입력값 전달
    loss = criterion(outputs, y)  # 손실 계산
    loss.backward()               # 역전파
    optimizer.step()              # 파라미터 업데이트

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# 결과 확인
print("학습된 가중치:", model.linear.weight.tolist())
print("학습된 편향:", model.linear.bias.tolist())

# 테스트 
print("기대한 출력", y.tolist())
print("실제의 출력",  model(torch.tensor([[1.0, 1.0]])).tolist())

# 3가지 방식으로 모델 저장
torch.save(model.state_dict(), "state_dict.pth")
torch.save(model, "model_full.pth")
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    # ... 기타 필요한 정보
}
torch.save(checkpoint, "checkpoint.pth")

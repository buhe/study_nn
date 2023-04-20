import torch
import numpy as np
import matplotlib.pyplot as plt

xs = torch.tensor([[1.0] ,[2.0] ,[3.0]])
ys = torch.tensor([[0.0] ,[0.0] ,[1.0]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
# trainer
model = LogisticRegressionModel()
loss_fn = torch.nn.BCELoss(reduction='sum')
opt = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(1000):
    for x, y in zip(xs, ys):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        print(f'loss: {loss.item()}')
        opt.zero_grad()
        loss.backward()
        opt.step()
        
print(f'w: {model.linear.weight.item()}')
print(f'b: {model.linear.bias.item()}')

x_test_np = np.linspace(0, 10, 100)
x_test = torch.tensor(x_test_np, dtype=torch.float32).view((100, 1))
y_test = model(x_test)
y_test = y_test.detach().numpy()

plt.plot(x_test_np, y_test, color='red')
plt.plot([0, 10], [0.5, 0.5], color='blue')
plt.xlabel('h')
plt.ylabel('p')
plt.grid()
plt.show()

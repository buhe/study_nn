import torch
import numpy as np
import matplotlib.pyplot as plt
xy = np.loadtxt('./diabetes.csv', delimiter=',', dtype=np.float32)
xs = torch.from_numpy(xy[:,:-1])
# print(xs)
ys = torch.from_numpy(xy[:,[-1]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 4)
        self.linear2 = torch.nn.Linear(4, 1)
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear1(x))
        y_pred = torch.sigmoid(self.linear2(y_pred))
        return y_pred
    
# trainer
model = LogisticRegressionModel()
loss_fn = torch.nn.BCELoss(reduction='mean')
opt = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(100):
    # for x, y in zip(xs, ys):
    y_pred = model(xs)
    loss = loss_fn(y_pred, ys)
    print(f'loss: {loss.item()}')
    opt.zero_grad()
    loss.backward()
    opt.step()
        
# print(f'w: {model.weight.item()}')
# print(f'b: {model.bias.item()}')

# x_test_np = np.linspace(0, 10, 100)
# x_test = torch.tensor(x_test_np, dtype=torch.float32).view((100, 1))
# y_test = model(x_test)
# y_test = y_test.detach().numpy()

# plt.plot(x_test_np, y_test, color='red')
# plt.plot([0, 10], [0.5, 0.5], color='blue')
# plt.xlabel('h')
# plt.ylabel('p')
# plt.grid()
# plt.show()

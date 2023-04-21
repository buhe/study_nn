import torch
import numpy as np
import matplotlib.pyplot as plt
xy = np.loadtxt('./diabetes.csv', delimiter=',', dtype=np.float32)
xs = torch.from_numpy(xy[:,:-1])
# print(xs)
ys = torch.from_numpy(xy[:,[-1]])
# print(ys)
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.active = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.active(self.linear1(x))
        x = self.active(self.linear2(x))
        x = self.active(self.linear3(x))
        return x
    
# trainer
model = LogisticRegressionModel()
loss_fn = torch.nn.BCELoss(reduction='mean')
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for i in range(100):
    # for x, y in zip(xs, ys):
    y_pred = model(xs)
    loss = loss_fn(y_pred, ys)
    print(f'loss: {loss.item()}')
    
    opt.zero_grad()
    loss.backward()

    opt.step()
        
print(f'w1: {model.linear1.weight.data}')
print(f'b1: {model.linear1.bias.data}')
# -0.882353,-0.0653266,0.147541,-0.373737,0,-0.0938897,-0.797609,-0.933333,1
x_test = torch.tensor([[-0.882353,-0.0653266,0.147541,-0.373737,0,-0.0938897,-0.797609,-0.933333]])
print(f'x_test: {x_test.data}')
y_test = model(x_test)
# -0.882353,0.266332,-0.0163934,0,0,-0.102832,-0.768574,-0.133333,0
# 0.0588235,0.708543,0.213115,-0.373737,0,0.311475,-0.722459,-0.266667,0
# -0.882353,-0.145729,0.0819672,-0.414141,0,-0.207153,-0.766866,-0.666667,1
print(f'y_pred: {y_test.item()}')
x_test2 = torch.tensor([[-0.882353,-0.145729,0.0819672,-0.414141,0,-0.207153,-0.766866,-0.666667]])
print(f'x_test 2: {x_test2.data}')
y_test2 = model(x_test2)
print(f'y_pred 2: {y_test2.item()}')
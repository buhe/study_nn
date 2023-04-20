import torch

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

x_test = torch.tensor([[4.0]])
y_test = model(x_test)

print(f'y_pred: {y_test.item()}')

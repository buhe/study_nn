import torch

xs = torch.tensor([[1.0] ,[2.0] ,[3.0]])
ys = torch.tensor([[2.0] ,[4.0] ,[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)
    
# trainer
model = LinearModel()
loss_fn = torch.nn.MSELoss(reduction='sum')
opt = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(1000):
    # for x, y in zip(xs, ys):
    y_pred = model(xs)
    loss = loss_fn(y_pred, ys)
    print(f'loss: {loss.item()}')
    opt.zero_grad()
    loss.backward()
    opt.step()
        
print(f'w: {model.linear.weight.item()}')
print(f'b: {model.linear.bias.item()}')

x_test = torch.tensor([[4.0]])
y_test = model(x_test)

print(f'y_pred: {y_test.item()}')

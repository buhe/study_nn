import torch
xs = [1.0, 2.0, 3.0]
ys = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True)
def forward(x):
    # print('call forward')
    return w * x
def loss(x, y):
    # print('call loss')
    y_hat = forward(x)
    return (y_hat - y) ** 2

print('predict (before training)', 4, forward(4).item())
for epoch in range(100):
    for x, y in zip(xs, ys):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data -= 0.01 * w.grad.data # .data 不会产生计算图
        print(w.grad.item(),'||', w.grad.data, "||")
        w.grad.data.zero_()
    print('progress:', epoch, l.item())

print('predict (after training)', 4, forward(4).item())

    

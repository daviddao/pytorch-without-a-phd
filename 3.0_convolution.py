import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

import pytorchvisu

import numpy as np

# Get data iterators
# We need transforms.ToTensor to convert PIL image into a Torch Tensor

transform = transforms.Compose([transforms.ToTensor()])

mnist_tr = torch.utils.data.DataLoader(
        datasets.MNIST('../datasets', train=True, download=True, transform=transform), 
        batch_size=100, shuffle=True)
mnist_te = torch.utils.data.DataLoader(
        datasets.MNIST('../datasets', train=False, download=True, transform=transform),
        batch_size=1000, shuffle=True)


class ConvNet(torch.nn.Module):
    def __init__(self, output_dim):
        super(ConvNet, self).__init__()

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(1, 10, kernel_size=5))
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("conv_2", torch.nn.Conv2d(10, 20, kernel_size=5))
        self.conv.add_module("dropout_2", torch.nn.Dropout())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_2", torch.nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(320, 50))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout())
        self.fc.add_module("fc2", torch.nn.Linear(50, output_dim))
        self.fc.add_module("relu_4", torch.nn.ReLU())
        self.fc.add_module("softmax", torch.nn.Softmax())

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 320)
        return self.fc.forward(x)

# Build MNIST model
model = ConvNet(10) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.003) 
loss_fn = torch.nn.CrossEntropyLoss(size_average=False)

# Extract the weights and biases
params = model.state_dict()
weights = params['fc.fc2.weight']
biases = params['fc.fc2.bias']

w = weights.numpy().reshape(-1)
b = biases.numpy().reshape(-1)

datavis = pytorchvisu.MnistDataVis()


def weight_init(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.bias.data.zero_()
            m.weight.data.normal_(0, 0.1)

weight_init(model) # initialize the weights

def adjust_lr(optimizer, i):
    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
    lr = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-i/decay_speed)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Visualisation
def training_step(i, update_test_data, update_train_data):
    _, (x, y) = next(enumerate(mnist_tr)) # Get a shuffled batch
    
    optimizer.zero_grad()
    adjust_lr(optimizer, i) # Adjust learning rate
    y_pred = model(Variable(x))

    loss = loss_fn(y_pred, Variable(y))

    loss.backward()
    optimizer.step()

    if update_train_data:
        
        w_ = np.sort(w)
        b_ = np.sort(b)
        x_np = x.numpy()
        y_pred_np = np.argmax(y_pred.data.numpy(), axis=1)
        y_np = y.numpy()
        accuracy = np.count_nonzero(y_np == y_pred_np) / (1.0 * y_np.shape[0])
        datavis.append_training_curves_data(i, accuracy, loss.data[0] / 2)     
        datavis.append_data_histograms(i, w_, b_)        
        im = pytorchvisu.numpy_format_mnist_images(x_np, y_pred_np, y_np)
        datavis.update_image1(im)
        print(str(i) + ": train accuracy: " + str(accuracy) + " training loss: " + str(loss.data[0]))

    if update_test_data:
        _, (xt, yt) = next(enumerate(mnist_te))
        yt_pred = model(Variable(xt))
        loss_t = loss_fn(yt_pred, Variable(yt))
        xt_np = xt.numpy()
        yt_np = yt.numpy()
        yt_pred_np = np.argmax(yt_pred.data.numpy(), axis=1)
        accuracy = np.count_nonzero(yt_np == yt_pred_np) / (1.0 * yt_np.shape[0])
        datavis.append_test_curves_data(i, accuracy, loss_t.data[0] / 20)        
        im = pytorchvisu.numpy_format_mnist_images(xt_np, yt_pred_np, yt_np)
        datavis.update_image2(im)
        print(str(i) + ": test accuracy: " + str(accuracy) + " test loss: " + str(loss_t.data[0]))

datavis.animate(training_step, iterations=10000, train_data_update_freq=20, test_data_update_freq=50, more_tests_at_start=True)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

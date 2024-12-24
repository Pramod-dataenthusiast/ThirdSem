#------------------------------------------------------------------
# PyTorch for Feedforward Networks

# 1. Add an additional layer to the network to change the layer
#    dimensions from 4, 10, 3 to 4, 10, 10, 3.

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)
        ## add a linear layer with 10 inputs and 10 outputs
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        ## add pass fc1 -> fc3 -> fc2
        x = F.relu(self.fc3(x))
        x = self.fc2(x)
        return x

# 2. Replace the ReLU functions with leaky ReLU functions with negative
#    gradient 0.1.

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)
        self.fc3 = nn.Linear(10, 10)
        ## define the leaky ReLU
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        
    def forward(self, x):
        ## replace the relu functions with leaky ReLU
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        x = self.fc2(x)
        return x

#------------------------------------------------------------------
# Convolutional Network

# 1. Add a dropout layer after the image is flattened and before the
#    image reaches the first linear layer.


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # 3 inputs, 6 outputs, kernels are size 5x5
        # provides 18 kernels
        # https://pytorch.org/docs/stable/nn.html#conv2d
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Pool 2x2 sets of pixels with stride size 2
        # https://pytorch.org/docs/stable/nn.html#maxpool2d
        self.pool = nn.MaxPool2d(2, 2)
        # 6 inputs, 16 outputs, with kernel size 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # https://pytorch.org/docs/stable/nn.html?highlight=eval#linear
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # final layer outputs 10 classes
        self.fc3 = nn.Linear(84, 10)

        # Randomly zero p proportion of the edge weights
        # https://pytorch.org/docs/stable/nn.html#dropout
        self.do = nn.Dropout(p=0.1)

        
    def forward(self, x):
        # x begins as a 32x32 pixel, 3 channel (R,G,B)image
        x = self.pool(F.relu(self.conv1(x)))
        # after convolution with 5x5 kernel, image is 28x28, then pooled to 14x14
        x = self.pool(F.relu(self.conv2(x)))
        # after convolution with 5x5 kernel, image is 10x10, then pooled to 5x5
        # giving 16 outputs of size 5x5.  Remember to adjust input size of fc1
        x = x.view(-1, 16 * 5 * 5)
        # flatten 16x5x5 tensor to 16x5x5=400 length vector

        ## use dropout on the flattened vector
        x = self.do(x)
        
        # 400 dimensions to 120
        x = F.relu(self.fc1(x))
        # 120 dimensions to 84
        x = F.relu(self.fc2(x))
        # 84 dimensions to 10
        x = self.fc3(x)
        return x


# 2. Add an additional convolutional layer between the last existing
#    convolutional layer and image flattening layer. The layer has an
#    input size of 16 and set the output size to 8. Use a kernel size of
#    $3\times 3$. Note that adjustment are required to the input size of
#    the linear layer.

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        # 3 inputs, 6 outputs, kernels are size 5x5
        # provides 18 kernels
        # https://pytorch.org/docs/stable/nn.html#conv2d
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Pool 2x2 sets of pixels with stride size 2
        # https://pytorch.org/docs/stable/nn.html#maxpool2d
        self.pool = nn.MaxPool2d(2, 2)
        # 6 inputs, 16 outputs, with kernel size 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 16 inputs, 8 outputs, with kernel size 3x3
        self.conv3 = nn.Conv2d(16, 8, 3)
        # Randomly zero p proportion of the edge weights
        # https://pytorch.org/docs/stable/nn.html#dropout
        self.do = nn.Dropout(p=0.1)
        
        # https://pytorch.org/docs/stable/nn.html?highlight=eval#linear
        self.fc1 = nn.Linear(8 * 3 * 3, 120)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # final layer outputs 10 classes
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x begins as a 32x32 pixel, 3 channel (R,G,B)image
        x = self.pool(F.relu(self.conv1(x)))
        # after convolution with 5x5 kernel, image is 28x28, then pooled to 14x14
        x = self.pool(F.relu(self.conv2(x)))
        # after convolution with 5x5 kernel, image is 10x10, then pooled to 5x5
        x = F.relu(self.conv3(x))
        # after convolution with 3x3 kernel, image is 3x3, no pooling
        # giving 8 outputs of size 3x3.  Remember to adjust input size of fc1
        x = x.view(-1, 8 * 3 * 3)
        # flatten 16x5x5 tensor to 16x5x5=400 length vector
        x = self.do(x)
        # 400 dimensions to 120
        x = F.relu(self.fc1(x))
        # 120 dimensions to 84
        x = F.relu(self.fc2(x))
        # 84 dimensions to 10
        x = self.fc3(x)
        return x


# 3. Add a padding layer to the output of the second convolution layer,
#    such that the third convolution layer uses kernels of size 
#    $3\times 3$ and is able to output images of size $5\times 5$.  Keep the
#    number of inputs and outputs of the third convolutional layer to 16
#    and 8.

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        # 3 inputs, 6 outputs, kernels are size 5x5
        # provides 18 kernels
        # https://pytorch.org/docs/stable/nn.html#conv2d
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Pool 2x2 sets of pixels with stride size 2
        # https://pytorch.org/docs/stable/nn.html#maxpool2d
        self.pool = nn.MaxPool2d(2, 2)
        # 6 inputs, 16 outputs, with kernel size 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 16 inputs, 8 outputs, with kernel size 3x3
        self.conv3 = nn.Conv2d(16, 8, 3)
        # Randomly zero p proportion of the edge weights
        # https://pytorch.org/docs/stable/nn.html#dropout
        self.do = nn.Dropout(p=0.1)
        
        # https://pytorch.org/docs/stable/nn.html?highlight=eval#linear
        self.fc1 = nn.Linear(8 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # final layer outputs 10 classes
        self.fc3 = nn.Linear(84, 10)

        # add 1 pixel padding so that 3x3 convolution does not reduce image
        self.pad = nn.ZeroPad2d(1)
        
    def forward(self, x):
        # x begins as a 32x32 pixel, 3 channel (R,G,B)image
        x = self.pool(F.relu(self.conv1(x)))
        # after convolution with 5x5 kernel, image is 28x28, then pooled to 14x14
        x = self.pool(F.relu(self.conv2(x)))
        # after convolution with 5x5 kernel, image is 10x10, then pooled to 5x5
        x = self.pad(x)
        # add padding to make image 7x7
        x = F.relu(self.conv3(x))
        # after convolution with 3x3 kernel, image is 5x5, no pooling
        # giving 8 outputs of size 5x5. Remember to adjust input size of fc1
        x = x.view(-1, 8 * 5 * 5)
        # flatten 8x5x5 tensor to 8x5x5=200 length vector
        x = self.do(x)
        # 200 dimensions to 120
        x = F.relu(self.fc1(x))
        # 120 dimensions to 84
        x = F.relu(self.fc2(x))
        # 84 dimensions to 10
        x = self.fc3(x)
        return x

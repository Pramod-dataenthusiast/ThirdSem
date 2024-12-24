#-------------------------------------------------------------------
# Example of an autoencoder architecture for the MNIST images.
# This AE encoder maps the 28x28 images to 8 2x2 matrices.
# The decoder maps the 8 2x2 matrices to a 28x28 image.

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        # MNIST are 28x28, single channel
        # this outputs 6 channels using 3x3 kernels
        # padding 1 adjusts to 30x30
        # stride 3 provides images of size 10x10
        self.conv1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
        # 2x2 max pool with stride 2 takes 10x10 images
        # to 5x5 images
        self.pool2 = nn.MaxPool2d(2, stride=2)  # b, 16, 5, 5
        # takes 16 channels to 8 channels, using 3x3 kernels
        # padding 1 adjusts 5x5 to 7x7 images
        # srtide 2 provides 3x3 image
        self.conv2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
        # 2x2 max pool with stride 1 takes 3x3 images
        # to 2x2 images
        self.pool1 = nn.MaxPool2d(2, stride=1)  # b, 16, 5, 5
        # This leaves us with 8 2x2 images as the encoding (8x2x2 = 32 values)
        
        # The encoded imags are 8 channel 2x2 images
        # This transposed convolution takes 8 channels to 16 channels
        # using a 3x3 kernel. This takes each pixel to a 3x3 image
        # Stride 2 provides a 5x5 image.
        self.iconv1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5

        # This transposed convolution takes 16 channels to 8 channels
        # using a 5x5 kernel. This takes each pixel to a 5x5 image
        # Stride 3 provides a 17x17 image.
        # padding 1 adjusts the images from 17x17 to 15x15
        self.iconv2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15

        # This transposed convolution takes 8 channels to 1 channel.
        # using a 2x2 kernel. This takes each pixel to a 2x2 image
        # Stride 2 provides a 30x30 image.
        # padding 1 adjusts the images from 30x30 to 28x28
        self.iconv3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b, 1, 28, 28
        # This provides a 1 channel 28x28 image.

    def encoder(self, x):
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        return(x)
    def decoder(self, x):
        x = F.relu(self.iconv1(x))
        x = F.relu(self.iconv2(x))
        x = torch_tanh(self.iconv3(x))
        return(x)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#-------------------------------------------------------------------
# Adding noise to the input images for the Denoising AE

# Create the function to add noise to the batch of images.
def add_noise(img):
    noise = torch_randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img

# The modified iterations, adding noise to the input images.
for epoch in range(num_epochs):
    # Iterate num_epoch number of times.
    for data in dataloader:
        # data contains the number of images specified in batch_size
        # this will loop untill all 60000 MNIST images are provided to the AE
        img, _ = data
        noise_img = add_noise(img)

        # pass the images throught the AE
        output = model(noise_img)
        # compare the images to the output
        loss = criterion(output, img)
        # update the gradients to miminise the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save some of the images to examine the improvement in the AE
    if epoch % 10 == 0:
        imageGrid = to_img(output.data)
        save_image(imageGrid, './imageGrid_{}.png'.format(epoch))


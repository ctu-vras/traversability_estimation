from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch


class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(inChannels, outChannels, 2)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 2)

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(Module):
    def __init__(self, channels):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            blockOutputs.append(x)
            # x = self.pool(x)
        # return the list containing the intermediate outputs
        return blockOutputs


class Decoder(Module):
    def __init__(self, channels):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
             for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])

    def forward(self, x, encFeatures):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x

    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        H, W = x.shape[-2:]
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures


class TerrainPredictor(Module):
    def __init__(self, encChannels, decChannels,
                 nbClasses=1, retainDim=True):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim

    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(decFeatures)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, x.shape[-2:])
        # return the segmentation map
        return map


class LinearPredictor(Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(1))
        self.b = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        y = self.a * x + self.b
        return y


def main():
    import matplotlib.pyplot as plt

    height_gt = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
                           [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
                           [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5],
                           [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5],
                           [0.5, 0.5, 0.0, 0.5, 0.7, 0.5, 0.5, 0.0, 0.5, 0.7],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    model = TerrainPredictor(encChannels=(1, 2, 4), decChannels=(4, 2))
    # model = LinearPredictor()
    height_init = torch.randn((1, 1, 10, 10))
    # height_init = torch.as_tensor(0.3 * height_gt - 0.5)[None][None]

    # ground truth
    height_gt = height_gt[None][None]

    # # check encoder-decoder output shapes
    # for out in model.encoder(height_init):
    #     print(out.shape)
    # print(model(height_init).shape)

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    model = model.train()

    losses = []
    for i in range(500):
        height_pred = model(height_init)
        loss = loss_fn(height_pred, height_gt)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(loss.item())
        losses.append(loss.item())

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Prediction')
    plt.imshow(height_pred.squeeze().detach().cpu().numpy())

    plt.subplot(1, 3, 2)
    plt.title('GT')
    plt.imshow(height_gt.squeeze().detach().cpu().numpy())

    plt.subplot(1, 3, 3)
    plt.title('Loss')
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()

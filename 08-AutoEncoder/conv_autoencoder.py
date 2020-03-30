
import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST

from DCIGNClamping import DCIGNClamping

import ipdb

num_epochs = 100
batch_size = 200
learning_rate = 1e-3

clamp = False

class AutoEncoder(nn.Module):
    def __init__(self, is_clamping=True, number_processors=1, batch_size=100):
        super(AutoEncoder, self).__init__()

        self.is_clamping = is_clamping

        self.latent_dim = 3
        self.number_processors = number_processors
        DCIGNClamping.latent_dim = self.latent_dim
        self.dcign = DCIGNClamping.apply
        self.batch_size = batch_size

        self.conv2d_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
        self.relu_1 = nn.ReLU(True)
        self.max_pool_2d_1 = nn.MaxPool2d(2, stride=2)  # b, 16, 5, 5
        self.conv2d_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
        self.relu_2 = nn.ReLU(True)
        self.max_pool_2d_2 = nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        self.fc = nn.Linear(int(self.batch_size * 32 / self.number_processors), self.latent_dim)

        self.d_fc = nn.Linear(self.latent_dim, int(self.batch_size * 32 / self.number_processors))
        self.d_conv_trans_2d_1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5
        self.d_relu_1 = nn.ReLU(True)
        self.d_conv_trans_2d_2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15
        self.d_relu_2 = nn.ReLU(True)
        self.d_conv_trans_2d_3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b, 1, 28, 28
        self.d_tanh = nn.Tanh()

    @staticmethod
    def reparameterize(x):
        std = torch.exp(0.5 * x)
        eps = torch.randn_like(std)
        return x + eps * std

    def encode(self, x):
        h1 = self.conv2d_1(x)
        h2 = self.relu_1(h1)
        h3 = self.max_pool_2d_1(h2)
        h4 = self.conv2d_2(h3)
        h5 = self.relu_2(h4)
        h6 = self.max_pool_2d_2(h5)
        h6 = h6.view(-1, int(self.batch_size * 32 / self.number_processors))
        return self.fc(h6)

    def decode(self, x):
        h0 = self.d_fc(x)
        h0 = h0.view([int(self.batch_size / self.number_processors), 8, 2, 2])
        h1 = self.d_conv_trans_2d_1(h0)
        h2 = self.d_relu_1(h1)
        h3 = self.d_conv_trans_2d_2(h2)
        h4 = self.d_relu_2(h3)
        h5 = self.d_conv_trans_2d_3(h4)
        return self.d_tanh(h5)

    def forward(self, x, idx=None):
        x = self.encode(x)

        if self.is_clamping:
            x = self.reparameterize(x)
            if self.training:
                x = self.dcign(x, idx)

        out = self.decode(x)
        return out


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def main():
    if not os.path.exists('./dc_img'):
        os.mkdir('./dc_img')

    number_of_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))
    model = AutoEncoder(
        is_clamping=clamp,
        number_processors=number_of_devices,
        batch_size=batch_size
    ).to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Let's use", number_of_devices, "GPUs!")
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = Variable(img).to(device)
            # ===================forward=====================
            output = model(img)
            # import ipdb
            # ipdb.set_trace()

            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.item()))

        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './dc_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './conv_autoencoder.pth')


if __name__ == "__main__":
    main()

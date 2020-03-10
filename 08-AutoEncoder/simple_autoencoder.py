
import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

num_epochs = 100
batch_size = 128
learning_rate = 1e-3


class DCIGNClamping(torch.autograd.Function):

    @staticmethod
    def forward(ctx, data_in, index):
        mask = torch.ones([3, ], dtype=torch.bool)
        mask[index] = False
        ctx.save_for_backward(data_in, index, mask)
        batch_index = 0
        data_in[:, mask] = torch.mean(data_in[:, mask], dim=batch_index, keepdim=True)
        return data_in

    @staticmethod
    def backward(ctx, grad_output):
        data_in, index, mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        batch_index = 0
        grad_input[:, mask] = data_in[:, mask] - torch.mean(data_in[:, mask], dim=batch_index, keepdim=True)
        return grad_input, None


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.latent_dim = 3
        self.dcign = DCIGNClamping.apply

        self.linear_1 = nn.Linear(28 * 28, 128)
        self.relu_1 = nn.ReLU(inplace=True)
        self.linear_2 = nn.Linear(128, 64)
        self.relu_2 = nn.ReLU(inplace=True)
        self.linear_3 = nn.Linear(64, 12)
        self.relu_3 = nn.ReLU(inplace=True)
        self.linear_4 = nn.Linear(12, self.latent_dim)

        self.d_linear_1 = nn.Linear(self.latent_dim, 12)
        self.d_relu_1 = nn.ReLU(inplace=True)
        self.d_linear_2 = nn.Linear(12, 64)
        self.d_relu_2 = nn.ReLU(inplace=True)
        self.d_linear_3 = nn.Linear(64, 128)
        self.d_relu_3 = nn.ReLU(inplace=True)
        self.d_linear_4 = nn.Linear(128, 28 * 28)
        self.d_tanh = nn.Tanh()

    @staticmethod
    def reparameterize(x):
        std = torch.exp(0.5 * x)
        eps = torch.randn_like(std)
        return x + eps * std

    def encode(self, x):
        h1 = self.linear_1(x)
        h2 = self.relu_1(h1)
        h3 = self.linear_2(h2)
        h4 = self.relu_2(h3)
        h5 = self.linear_3(h4)
        h6 = self.relu_3(h5)
        return self.linear_4(h6)

    def decode(self, x):
        h1 = self.d_linear_1(x)
        h2 = self.d_relu_1(h1)
        h3 = self.d_linear_2(h2)
        h4 = self.d_relu_2(h3)
        h5 = self.d_linear_3(h4)
        h6 = self.d_relu_3(h5)
        h7 = self.d_linear_4(h6)
        return self.d_tanh(h7)

    def forward(self, x, idx=None):
        x = self.encode(x)
        z = self.reparameterize(x)
        if self.training:
            z = self.dcign(z, idx)
        out = self.decode(z)
        return out


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def main():
    if not os.path.exists('./mlp_img'):
        os.mkdir('./mlp_img')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder().to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).to(device)
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './mlp_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './sim_autoencoder.pth')


if __name__ == "__main__":
    main()

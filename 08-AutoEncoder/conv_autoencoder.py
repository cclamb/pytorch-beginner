
import os
from collections import OrderedDict

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


class DecodeAdapter(nn.Module):
    def __init__(self, autoencoder):
        super(DecodeAdapter, self).__init__()
        self.autoencoder = autoencoder

    def forward(self, x):
        output = self.autoencoder.decode(x)
        return output


class AutoEncoder(nn.Module):
    def __init__(self, is_clamping=True, number_processors=1, batch_size=100, latent_dim=16):
        super(AutoEncoder, self).__init__()

        self.is_clamping = is_clamping

        self.latent_dim = latent_dim
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
        self.fc = nn.Linear(32, self.latent_dim)

        self.d_fc = nn.Linear(self.latent_dim, 32)
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
        h6 = h6.view([int(self.batch_size / self.number_processors), 32])
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

def to_contrast_img(x):
    # ipdb.set_trace()
    x = 0.5 * (x + 1)
    x = 255 * x
    x = x.clamp(0, 255)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def run(latent_dim):
    clamp_state = '_clamp' if clamp else '_no_clamp'
    latency_state = '_{}'.format(latent_dim)
    output_directory = './dc_img{}{}'.format(clamp_state, latency_state)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    number_of_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))
    model = AutoEncoder(
        is_clamping=clamp,
        number_processors=number_of_devices,
        batch_size=batch_size,
        latent_dim=latent_dim
    ).to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Let's use", number_of_devices, "GPUs!", flush=True)
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
            loss = criterion(output, img)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.item()), flush=True)

        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './{}/image_{}.png'.format(output_directory, epoch))

    torch.save(model.state_dict(), './{}/conv_autoencoder.pth'.format(output_directory))


def run_all():
    global clamp
    clamp = False
    print('clamping OFF')
    for latent_dim in [128, 64, 32, 16, 8, 4]:
        print('[running convnet with {} latent variables]'.format(latent_dim), flush=True)
        run(latent_dim)
        print('[finished]', flush=True)
    clamp = True
    print('[clamping ON]')
    for latent_dim in [128, 64, 32, 16, 8, 4]:
        print('[running convnet with {} latent variables]'.format(latent_dim), flush=True)
        run(latent_dim)
        print('[finished]', flush=True)


def parallel_to_serial_state(state_dict):
    # State dicts created using dataparallel reference 'module';
    # we need to remove that to load serially
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # odict_keys is not subscriptable, so rather than creating
        # a list just to check for the 'module' prifix, just do it
        # here. If one key has 'module', they all will.
        if 'module' not in k:
            return state_dict
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def run_decode(number_of_devices=1,
               is_clamping=True,
               current_batch_size=1,
               latent_dim=32,
               saved_model='./conv_autoencoder_32.pth'):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))
    model = AutoEncoder(
        is_clamping=is_clamping,
        number_processors=number_of_devices,
        batch_size=current_batch_size,
        latent_dim=latent_dim
    ).to(device)
    state_dict = torch.load(saved_model, map_location=device)
    state_dict = parallel_to_serial_state(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    generator = DecodeAdapter(model)
    data = torch.zeros([latent_dim, 1, 28, 28])
    for i in range(latent_dim):
        latent_vector = torch.zeros(latent_dim)
        latent_vector[i] = 1
        output = generator.forward(latent_vector)
        data[i, :, :, :] = output
    pic = to_img(data.cpu().data)
    save_image(pic, './image_construct_{}_lv.png'.format(latent_dim), padding=1, pad_value=0.5)


def run_all_decode():
    for latent_dim in [128, 64, 32, 16, 8, 4]:
        model_file = 'conv_autoencoder_{}.pth'.format(latent_dim)
        run_decode(latent_dim=latent_dim, saved_model=model_file)


def main():
    run_all_decode()


if __name__ == "__main__":
    main()

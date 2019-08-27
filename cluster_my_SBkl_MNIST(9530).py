from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from mpl_toolkits.mplot3d import axes3d
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from sklearn import mixture
from sklearn.cluster import KMeans

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=60, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=0.002, metavar='N',
                    help='learning rate.')
args = parser.parse_args()

args.cuda = True

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

torch.manual_seed(321894354)
#95.30  bach:100  lr:0.002  structure:784-500--500-500--500-2000--2000-10 2000-200--200-200--200-10  lrgmm:1.05  decay: lr = max(args.lr * 0.9 ** (np.floor(epoch) / 40.0), 0.000001)
# lr = lr * ((np.cos((epoch / (math.pi * 2))) + 1) / 2)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/data/', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/data/', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

class VAE(nn.Module):
    def __init__(self, lr = 0.002,n_gmm = 10, latent_dim=10):
        super(VAE, self).__init__()

        layers = []
        layers += [nn.Linear(784, 500)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(500, 500)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(500, 2000)]
        layers += [nn.ReLU()]
        self.FC = nn.Sequential(*layers)

        self.mean = nn.Linear(2000, latent_dim)
        self.logvar = nn.Linear(2000, latent_dim)

        # layers = []
        # layers += [nn.Linear(latent_dim, 20)]
        # layers += [nn.ReLU()]
        # layers += [nn.Linear(2000, 500)]
        # layers += [nn.ReLU()]
        # layers += [nn.Dropout(0.5)]
        # layers += [nn.Linear(500, 200)]
        # layers += [nn.Sigmoid()]
        # layers += [nn.Dropout(0.5)]
        # layers += [nn.Linear(200, n_gmm)]
        # layers += [nn.Sigmoid()]
        # self.FCGAMMA = nn.Sequential(*layers)

        #layers = []
        #layers += [nn.Linear(20, n_gmm)]
        #layers += [nn.Softplus()]
        #self.a = nn.Sequential(*layers)
        layers = []
        layers += [nn.Linear(2000, 200)]
        layers += [nn.Softplus()]
        layers += [nn.Linear(200, n_gmm)]
        layers += [nn.Softplus()]
        self.b = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(latent_dim, 2000)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(2000, 500)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(500, 500)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(500, 784)]
        layers += [nn.Sigmoid()]
        self.IFC = nn.Sequential(*layers)

        self.pi = torch.nn.Parameter(data=torch.ones(n_gmm)/n_gmm, requires_grad=False)
        self.mu = torch.nn.Parameter(data=torch.randn(n_gmm, latent_dim), requires_grad=False)
        self.var = torch.nn.Parameter(data=torch.ones(n_gmm, latent_dim), requires_grad=False)

        self.lr = lr

    def encode(self, x):
        x1 = self.FC(x.view(x.size(0), -1))
        mu = self.mean(x1)
        logvar = self.logvar(x1)
        # beta_a = self.a(x1)
        beta_b = self.b(x1)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = eps * std + mu

        return z, mu, logvar, beta_b

    def decode(self, z):
        xx = self.IFC(z)
        return xx.view(xx.size(0), 1, 28, 28)

    def gmm(self, z):
        temp_p_c_z = torch.exp(torch.sum(torch.log(self.pi.unsqueeze(0).unsqueeze(-1)+1e-10) - 0.5* torch.log(2*math.pi*self.var.unsqueeze(0)+1e-10) - (z.unsqueeze(-2)-self.mu.unsqueeze(0)).pow(2)/(2*self.var.unsqueeze(0)+ 1e-10), -1)) + 1e-10
        return temp_p_c_z / torch.sum(temp_p_c_z, -1).unsqueeze(-1)
        #return -temp_p_c_z

    def forward(self, x, lr):
        self.lr = lr
        z, mu, logvar, beta_b = self.encode(x)
        # gamma = self.FCGAMMA(z)
        # gamma = torch.softmax(gamma/2, 1)
        #v = self.FCGAMMA(z)
        beta_a = torch.Tensor(beta_b.shape).zero_().cuda()+1
        gamma, remaining_gamma = self.reparameterizebeta(beta_a, beta_b)
        self.compute_gmm_params(z, gamma)
        gamma_l = self.gmm(z)
        xx = self.decode(z)
        return xx, z, gamma, mu, logvar, gamma_l, beta_a, beta_b

    def compute_gmm_params(self, z, gamma):
        # gamma = gamma.clone()
        # pi = torch.mean(gamma,0)
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(-2), 0)/(torch.sum(gamma, 0).unsqueeze(-1)+1e-10)
        var = torch.sum(gamma.unsqueeze(-1) * (z.unsqueeze(-2) - mu.unsqueeze(0)).pow(2), 0)/(torch.sum(gamma, 0).unsqueeze(-1)+1e-10)

        lr = self.lr * 1.05
        # self.pi.data = (1-lr) * self.pi.data + lr * pi.clone().data
        self.var.data = (1-lr) * self.var.data + lr * var.clone().data
        self.mu.data = (1-lr) * self.mu.data + lr * mu.clone().data

    def reparameterizebeta(self, a, b):
        uniform_samples = torch.Tensor(b.shape).uniform_(0.01, 0.99)
        v_samples = (1 - (uniform_samples.cuda() ** (1 / b))) ** (1 / a)
        remaining_stick = torch.Tensor(a.shape[0], 1).zero_().cuda() + 1
        stick_segment = torch.Tensor(a.shape).zero_().cuda()
        for i in range(a.shape[1]):
            stick_segment[:, i] = (v_samples[:, i].unsqueeze(-1) * remaining_stick).squeeze(-1)
            remaining_stick = remaining_stick.clone() * (1 - v_samples[:, i]).unsqueeze(-1)
        stick_segment = stick_segment/torch.sum(stick_segment, 1).unsqueeze(-1)
        return stick_segment, remaining_stick



model = VAE(lr=args.lr).to(device)

optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr)

ac = 0

#不注释就是读取模型
# checkpoint = torch.load('./checkpoint/ckpt8-9one_best.t8-9')#
# model.load_state_dict(checkpoint['net'])

def Beta_fn(a, b):
    return torch.exp(a.lgamma() + b.lgamma() - (a+b).lgamma())

# NOte a = torch.Tensor(0.2)
# Note a.lgamma_()

def calc_kl_divergence(a, b, prior_alpha, prior_beta):
        # compute taylor expansion for E[log (1-v)] term
        # hard-code so we don't have to use Scan()
    kl = 0
    for i in range(1,11,1):
        kl += 1./(i+a*b) * Beta_fn(i/a, b)
    kl *= (prior_beta-1)*b

        # use another taylor approx for Digamma function
    psi_b_taylor_approx = torch.log(b) - 1./(2 * b) - 1./(12 * b**2)
    kl += (a-prior_alpha)/a * (-0.57721 - psi_b_taylor_approx - 1/b) #T.psi(self.posterior_b)

        # add normalization constants
    kl += torch.log(a*b) + torch.log(Beta_fn(prior_alpha, prior_beta))

        # final term
    kl += -(b-1)/b

    return torch.sum(kl)


def train(epoch):
    model.train()
    train_loss = 0
    ACT = 0
    ACTn = 0
    for batch_idx, (data, gt) in enumerate(train_loader):
        lr = args.lr
        if epoch>=0:
            # lr = max(args.lr * 0.9**(np.floor((epoch)*10)/10.0), 0.00002)
            # lr = args.lr * ((np.sin((epoch/(math.pi*2)))+1)/2)
            lr = max(args.lr * 0.9 ** (np.floor(epoch) / 40.0), 0.000001)
            lr = lr * ((np.cos((epoch / (math.pi * 2))) + 1) / 2)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        data = data.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()
        recon_batch, z, gamma, mu, logvar, gamma_l, a, b = model(data, lr=lr)

        #print(np.any(np.isnan(recon_batch.detach().cpu().numpy())), np.any(np.isnan(z.detach().cpu().numpy())), np.any(np.isnan(gamma.detach().cpu().numpy())), np.any(np.isnan(mu.detach().cpu().numpy())), np.any(np.isnan(logvar.detach().cpu().numpy())), np.any(np.isnan(gamma_l.detach().cpu().numpy())))

        BCE = F.binary_cross_entropy(recon_batch, data, reduction='mean') * 784

        KLD = torch.sum(0.5 * gamma.unsqueeze(-1) * (
                        torch.log((torch.zeros_like(gamma.unsqueeze(-1)) + 2) * math.pi+1e-10) + torch.log(
                    model.var.unsqueeze(0)+1e-10) + torch.exp(logvar.unsqueeze(-2)) / (model.var.unsqueeze(0)+1e-10) + (
                                    mu.unsqueeze(-2) - model.mu.unsqueeze(0)).pow(2) / (model.var.unsqueeze(0)+1e-10)),
                            [-1, -2])
        KLD -= 0.5 * torch.sum(logvar + 1, -1)
        KLD += torch.sum(torch.log(gamma+1e-10) * gamma, -1)
        KLD -= torch.sum(torch.log(gamma_l+1e-10) * gamma, -1)
        KLD = torch.mean(KLD)

        prior_alpha = torch.Tensor(1).zero_().cuda() + 1
        prior_beta = torch.Tensor(1).zero_().cuda() + 3
        SBKL = calc_kl_divergence(a, b, prior_alpha, prior_beta)/ data.shape[0]
        loss = BCE + KLD + SBKL*0.005

        if batch_idx==0:
            gt_all = gt
            ret_all = torch.max(gamma, 1)[1]
        else:
            gt_all = torch.cat((gt_all, gt))
            ret_all = torch.cat((ret_all, torch.max(gamma, 1)[1]))

        #ACT += cluster_acc(torch.max(gamma, 1)[1].detach().cpu().numpy(), gt.detach().cpu().numpy())[0]
        #ACTn += 1
        if batch_idx % args.log_interval == 0:
            acc = cluster_acc(torch.max(gamma, 1)[1].detach().cpu().numpy(), gt.detach().cpu().numpy())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBCE: {:.6f}\tKLD: {:.6f}\tKL: {:.6f}\tACC: {:.2f}\tLR: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item(), BCE, KLD, SBKL, acc[0], lr))


        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx == 0:
            for j in range(gamma.size()[1]):
                a = torch.max(gamma, 1)[1]
                n = min(data.size(0), 40)
                m = 0
                samenumber = recon_batch[0, :, :, :].zero_()
                samenumber_org = data[0, :, :, :].zero_()
                for i in range(1, recon_batch.shape[0]):
                    if a[i] == j:
                        xx = recon_batch[i, :, :, :]
                        xx_org = data[i, :, :, :]
                        if m==0:
                            samenumber = xx
                            samenumber_org = xx_org
                        else:
                            samenumber = torch.cat((samenumber, xx))
                            samenumber_org = torch.cat((samenumber_org, xx_org))

                        m = m + 1
                        if m == n-1:
                            break

                samenumber = samenumber.unsqueeze(-3)
                samenumber_org = samenumber_org.unsqueeze(-3)

                samenumber = torch.cat((samenumber_org, samenumber))

                save_image(samenumber.cpu(),
                   'results8-3/'+'c'+ str(j) + '.png', nrow=max(m,1))

    #test set
    for batch_idx, (data, gt) in enumerate(test_loader):

        data = data.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()
        recon_batch, z, gamma, mu, logvar, gamma_l, a, b = model(data, lr=lr)

        #print(np.any(np.isnan(recon_batch.detach().cpu().numpy())), np.any(np.isnan(z.detach().cpu().numpy())), np.any(np.isnan(gamma.detach().cpu().numpy())), np.any(np.isnan(mu.detach().cpu().numpy())), np.any(np.isnan(logvar.detach().cpu().numpy())), np.any(np.isnan(gamma_l.detach().cpu().numpy())))

        BCE = F.binary_cross_entropy(recon_batch, data, reduction='mean') * 784

        KLD = torch.sum(0.5 * gamma.unsqueeze(-1) * (
                        torch.log((torch.zeros_like(gamma.unsqueeze(-1)) + 2) * math.pi+1e-10) + torch.log(
                    model.var.unsqueeze(0)+1e-10) + torch.exp(logvar.unsqueeze(-2)) / (model.var.unsqueeze(0)+1e-10) + (
                                    mu.unsqueeze(-2) - model.mu.unsqueeze(0)).pow(2) / (model.var.unsqueeze(0)+1e-10)),
                            [-1, -2])
        KLD -= 0.5 * torch.sum(logvar + 1, -1)
        KLD += torch.sum(torch.log(gamma+1e-10) * gamma, -1)
        KLD -= torch.sum(torch.log(gamma_l+1e-10) * gamma, -1)
        KLD = torch.mean(KLD)

        prior_alpha = torch.Tensor(1).zero_().cuda() + 1
        prior_beta = torch.Tensor(1).zero_().cuda() + 3
        SBKL = calc_kl_divergence(a, b, prior_alpha, prior_beta)/ data.shape[0]
        loss = BCE + KLD + SBKL*0.005

        gt_all = torch.cat((gt_all, gt))
        ret_all = torch.cat((ret_all, torch.max(gamma, 1)[1]))

        #ACT += cluster_acc(torch.max(gamma, 1)[1].detach().cpu().numpy(), gt.detach().cpu().numpy())[0]
        #ACTn += 1
        if batch_idx % args.log_interval == 0:
            acc = cluster_acc(torch.max(gamma, 1)[1].detach().cpu().numpy(), gt.detach().cpu().numpy())
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBCE: {:.6f}\tKLD: {:.6f}\tKL: {:.6f}\tACC: {:.2f}\tLR: {:.6f}'.format(
                epoch, batch_idx * len(data), len(test_loader.dataset),
                100. * batch_idx / len(test_loader),
                loss.item(), BCE, KLD, SBKL, acc[0], lr))

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    if epoch % 10 == 0:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt8-9mnist4.t8-9')

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(train_loader.dataset)+len(test_loader.dataset))))
    acc = cluster_acc(ret_all.detach().cpu().numpy(), gt_all.detach().cpu().numpy())[0]

    global ac
    print('current:', acc, 'best: ', ac)

    if (acc > ac):
        ac = acc
        print('Saving best..')

        state = {
            'net': model.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt8-9mnist4_best.t8-9')


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
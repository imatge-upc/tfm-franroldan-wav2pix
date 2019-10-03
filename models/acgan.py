import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import Concat_embed
import pdb
from models.spectral_norm import SpectralNorm


class generator(nn.Module):
	def __init__(self, dataset='youtubers'):
		super(generator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.noise_dim = 100
		self.embed_dim = 62
		self.projected_embed_dim = 128
		self.latent_dim = self.noise_dim + self.projected_embed_dim
		self.ngf = 64
		self.dataset_name = dataset
		self.projection = nn.Sequential(
			nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
			nn.BatchNorm1d(num_features=self.projected_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
			)

		# based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
		self.netG = nn.Sequential(
			nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
			#nn.BatchNorm2d(self.ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
			#nn.BatchNorm2d(self.ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
			#nn.BatchNorm2d(self.ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
			#nn.BatchNorm2d(self.ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
			nn.Tanh()
			 # state size. (num_channels) x 64 x 64
			)


	def forward(self, embed_vector, z, project=True):

		if self.dataset_name == 'youtubers' and not project:
			padding = Variable(torch.cuda.FloatTensor(embed_vector.data.shape[0], self.projected_embed_dim - 62).fill_(
				0).float()).cuda()
			#padding = Variable(torch.FloatTensor(embed_vector.data.shape[0], self.projected_embed_dim - 62).fill_(
				0).float())
			projected_embed = torch.cat([embed_vector, padding], 1).unsqueeze(2).unsqueeze(3)
		else:
			projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
		latent_vector = torch.cat([projected_embed, z], 1)
		output = self.netG(latent_vector)

		return output

class discriminator(nn.Module):
	def __init__(self, dataset='youtubers'):
		super(discriminator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.embed_dim = 62
		self.projected_embed_dim = 128
		self.ndf = 64
		self.B_dim = 128
		self.C_dim = 16
		self.dataset_name = dataset
		self.conv1 = SpectralNorm(nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False))
		self.conv2 = SpectralNorm(nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False))
		self.conv3 = SpectralNorm(nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False))
		self.conv4 = SpectralNorm(nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False))
		self.disc_linear = nn.Linear(self.ndf * 1, self.ndf)
		self.disc_linear2 = nn.Linear(31, 31)
		self.aux_linear = nn.Linear(4*4*512, self.embed_dim+1)
		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()
		self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

		self.netD_2 = nn.Sequential(
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
			#nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
			#nn.Sigmoid()
			)

	def forward(self, inp, embed, concat=True, project=True):
		#x_intermediate = self.netD_1(inp)
		x = nn.LeakyReLU(0.2, inplace=True)(self.conv1(inp))
		x = nn.LeakyReLU(0.2, inplace=True)(self.conv2(x))
		x = nn.LeakyReLU(0.2, inplace=True)(self.conv3(x))
		x_intermediate = nn.LeakyReLU(0.2, inplace=True)(self.conv4(x))
		x2 = self.netD_2(x_intermediate)
		x = x2.view(-1, x2.data.shape[0])
		xc = x_intermediate.view(-1, 4*4*512)
		c = self.aux_linear(xc)
		c = self.softmax(c)
		if x2.data.shape[0] == 64:
			s = self.disc_linear(x)
		else:
			s = nn.Linear(x2.data.shape[0] * 1, x2.data.shape[0] * 1).cuda()(x)
			#s = nn.Linear(x2.data.shape[0] * 1, x2.data.shape[0] * 1)(x)

		#s = self.sigmoid(s)
		return s, c, x_intermediate


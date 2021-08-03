import torch
import torch.nn as nn


class VAE(nn.Module):

	def __init__(self, encoder, decoder):
		super(VAE, self).__init__()

		self.enc = encoder
		self.dec = decoder

	def forward(self, x: torch.tensor) -> tuple:

		z, z_mu, z_logvar = self.enc(x)
		x_mu, x_logvar = self.dec(z)

		return x_mu, x_logvar, z_mu, z_logvar
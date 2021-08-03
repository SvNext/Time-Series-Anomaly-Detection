import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

	def __init__(self, input_size: int, hidden_size: int, 
				num_layers: int, seq_length: int, z_size: int):
		super(Encoder, self).__init__()

		self.z_size = z_size
		self.num_layers = num_layers
		self.hidden_size = hidden_size

		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
							batch_first=True, bidirectional=True)

		# mu
		self.m_layers = nn.ModuleList([

			nn.Sequential(

				nn.Linear(2*hidden_size, hidden_size),
				nn.LeakyReLU(0.1),
				nn.Linear(hidden_size, z_size)

			)

			for _ in range(seq_length)
		])

		# logvar
		self.v_layers = nn.ModuleList([ 

			nn.Sequential(

				nn.Linear(2*hidden_size, hidden_size),
				nn.LeakyReLU(0.1),
				nn.Linear(hidden_size, z_size)

			)

			for _ in range(seq_length)
			])


	def forward(self, x: torch.tensor) -> torch.tensor:

		h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size)
		c0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size)

		output, _ = self.lstm(x, (h0, c0))
		z = torch.zeros(x.size(0), x.size(1), self.z_size)

		mu = torch.zeros(x.size(0), x.size(1), self.z_size)
		logvar = torch.zeros(x.size(0), x.size(1), self.z_size)

		for indx, (m, v) in enumerate(zip(self.m_layers, self.v_layers)):
			mu[:, indx, :]  = m(output[:, indx, :])
			logvar[:, indx, :] = F.softplus(v(output[:, indx, :]))

			# reparametrazation trick
			std = torch.exp(logvar[:, indx, :] / 2)
			#eps = torch.randn_like(std)
			eps = 1e-2

			z[:, indx, :] = mu[:, indx, :] + eps+std

		return z, mu, logvar
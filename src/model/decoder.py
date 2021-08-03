import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):

	def __init__(self, z_size:  int, hidden_size: int, 
				num_layers: int, seq_length: int, output_size: int):
		super(Decoder, self).__init__()

		self.num_layers = num_layers
		self.output_size = output_size
		self.hidden_size = hidden_size

		self.lstm = nn.LSTM(z_size, hidden_size, num_layers, 
							batch_first=True, bidirectional=True)


		self.layers = nn.ModuleList([
			nn.Sequential(

				nn.Linear(hidden_size * 2, hidden_size),
				nn.LeakyReLU(0.2)

			)

			for _ in range(seq_length)

		])


		self.h2 = output_size * 2
		self.h4 = output_size * 4

		self.m = nn.Sequential(

			nn.Flatten(),
			nn.Linear(
				seq_length * hidden_size, 
				seq_length * hidden_size, bias=False
			),
			nn.LeakyReLU(0.2),
			nn.BatchNorm1d(seq_length * hidden_size),

			nn.Linear(
				seq_length * hidden_size, 
				seq_length * self.h4
			),
			nn.LeakyReLU(0.2),
		)


		# mu
		self.m_layers = nn.ModuleList([
			nn.Linear(self.h4, output_size) for _ in range(seq_length)
		])

		# logvar
		self.v_layers = nn.ModuleList([
			nn.Linear(self.h4, output_size) for _ in range(seq_length)
		])


	def forward(self, z: torch.tensor) -> torch.tensor:

		h0 = torch.zeros(2*self.num_layers, z.size(0), self.hidden_size)
		c0 = torch.zeros(2*self.num_layers, z.size(0), self.hidden_size)

		output, _ = self.lstm(z, (h0, c0))
		hidden1 = torch.zeros(z.size(0), z.size(1), self.hidden_size)

		for indx, m in enumerate(self.layers):
			hidden1[:, indx, :] = m(output[:, indx, :])

		hidden2 = self.m(hidden1)
		hidden2 = hidden2.view(z.size(0), z.size(1), -1)

		mu = torch.zeros(z.size(0), z.size(1), self.output_size)
		logvar = torch.zeros(z.size(0), z.size(1), self.output_size)


		for indx, (m, v) in enumerate(zip(self.m_layers, self.v_layers)):
			mu[:, indx, :]  = m(hidden2[:, indx, :])
			logvar[:, indx, :] = F.softplus(v(hidden2[:, indx, :]))

		return mu, logvar
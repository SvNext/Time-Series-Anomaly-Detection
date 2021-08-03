import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class NominalDataset(Dataset):

	def __init__(self, x: np.array, seq_length: int, step: int):
		super(NominalDataset, self).__init__()

		if not isinstance(x, np.ndarray):
			raise Exception("X isn't numpy.ndarray ")

		self.n_samples = x.shape[0] - step * (seq_length - 1) - 1
		self.data = [
			(
				torch.from_numpy(x[i:i+(seq_length*step):step, :].astype(np.float32)),
				torch.from_numpy(x[i+((seq_length-1)*step)+1, :].astype(np.float32))
			)

			for i in range(self.n_samples)
		]
	

	def __getitem__(self, index: int) -> tuple:
		return self.data[index][0], self.data[index][1]


	def __len__(self) -> int:
		return self.n_samples
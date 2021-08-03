import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Animator:
	
	
	def plot(self, data: pd.DataFrame, cnames: list, 
			 start_seq: int, end_seq: int, ls = '--'):

		n_streams = len(cnames)
		fig, axes = plt.subplots(n_streams, 1, figsize=(20, n_streams*4))

		plot_data = data.loc[start_seq:end_seq, :]
		for i, cname in enumerate(cnames):
			axes[i].plot(plot_data.index, plot_data.loc[:,  cname], 
						 color='grey', ls=ls, lw=1.5, zorder=1)

			anomaly = [0, 1]
			colors  = ['green', 'yellow']
			labels  = ['normality', 'anomaly']

			anomaly_index = plot_data.loc[plot_data.anomaly == 0, :].index
			y_min = np.min(plot_data.loc[anomaly_index, cname])
			y_max = np.max(plot_data.loc[anomaly_index, cname])

			for a, c, l in zip(anomaly, colors, labels):
				indexes = plot_data.loc[plot_data.anomaly == a, :].index
				axes[i].scatter(indexes, np.min([1.25*y_min, -0.05*y_max])*np.ones_like(indexes),
								marker='D', s=15, color=c, zorder = 2, label = l)

				axes[i].set_xlabel('Time', fontsize = 15)
				axes[i].set_ylim(np.min([1.25*y_min, -0.15*y_max]), 1.1*y_max)

				axes[i].legend(loc ='upper right')
				axes[i].set_ylabel('Stream №{}'.format(cname), fontsize = 15)
				axes[i].grid(color='blue', which='major', linestyle=':',
							 linewidth=0.5)

		return fig

	
	def plot_results(self, data: pd.DataFrame, cnames: list):
		num_streams = len(cnames)
		fig, axes = plt.subplots(num_streams + 2, 1, figsize=(30, 6 * num_streams))
		for i, cname in enumerate(cnames):
			c1, c2 = 'mu_{}'.format(cname), 'std_{}'.format(cname)

			min_bound = data.loc[:, c1] - data.loc[:, c2]
			max_bound = data.loc[:, c1] + data.loc[:, c2]
			axes[i].plot(data.index, data.loc[:, cname], color='blue')
			axes[i].fill_between(data.index, min_bound, max_bound, color='grey')

			anomaly = [0, 1]
			colors  = ['green', 'yellow']
			labels  = ['normality', 'anomaly']

			anomaly_index = data.loc[data.anomaly == 0, :].index
			y_min = np.min(data.loc[anomaly_index, cname])
			y_max = np.max(data.loc[anomaly_index, cname])

			for a, c, l in zip(anomaly, colors, labels):

				indexes = data.loc[data.anomaly == a, :].index
				axes[i].scatter(indexes, np.min([2*y_min, -0.5*y_max])*np.ones_like(indexes),
								marker='D', s=15, color=c, zorder = 2, label = l)

				axes[i].legend(loc ='upper right')
				axes[i].set_ylabel('Stream №{}'.format(cname), fontsize = 15)
				axes[i].set_ylim(np.min([2.1*y_min, -0.75*y_max]), 1.5*y_max)
				axes[i].grid(color='blue', which='major', linestyle=':',linewidth=0.5)


		for i, cname in enumerate(['log_prob_z', 'log_prob_x']):

			y_min = np.min(data.loc[:, cname])
			axes[num_streams + i].plot(data.index, data.loc[:, cname], color='red')

			for a, c, l in zip(anomaly, colors, labels):

				indexes = data.loc[data.anomaly == a, :].index
				axes[num_streams + i].scatter(indexes, 0.05*np.abs(y_min) * np.ones_like(indexes),
											  marker='D', s=20, color=c, zorder = 2, label = l)

				axes[num_streams + i].legend(loc ='upper right')
				axes[num_streams + i].set_ylim(None, 0.1*np.abs(y_min))
				axes[num_streams + i].set_ylabel(cname, fontsize = 15)
				axes[num_streams + i].grid(color='blue', which='major', linestyle=':',linewidth=0.5)


		return fig


	def plot_spot(self, data:pd.DataFrame, cname: str):

		fig, ax = plt.subplots(1, 1, figsize=(16, 8))
		ax.plot(data.index, data.loc[:, cname], color='green', zorder = 1)
		ax.plot(data.index, data.loc[:, 'thresholds'], color='blue', ls='--', lw=1.5, zorder = 1)
		
		anomaly = [0, 1]
		colors  = ['green', 'yellow']
		labels  = ['normality', 'anomaly']
		
		y_min = np.min(data.loc[:, cname])
		for a, c, l in zip(anomaly, colors, labels):

			indexes = data.loc[data.anomaly == a, :].index
			ax.scatter(indexes, 0.01*np.abs(y_min) * np.ones_like(indexes),
					marker='D', s=20, color=c, zorder = 2, label = l)



		ax.scatter(data.loc[data.pred_anomaly == 1, :].index, 
				   data.loc[data.pred_anomaly == 1, cname], marker='D', s=30, color='red', zorder = 2)


		ax.set_xlabel('Time', fontsize = 12)
		ax.set_ylabel(cname, fontsize = 12)
		ax.set_ylim(0.2*y_min, None)
		ax.grid(color='blue', which='major', linestyle=':',linewidth=0.5)

		return fig
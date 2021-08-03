import os
import numpy as np
import pandas as pd

from PIL import Image
import streamlit as st

from src.model.spot import biSPOT
from src.utils.animator import Animator

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix




data_path = dict({

	'Machine1': 'machine-1-1.csv',
	'Machine2': 'machine-1-2.csv',
	'Machine3': 'machine-1-3.csv'

})


# fix bags streamlit
st.cache()
def check_files():
	if os.path.exists('cfile.csv'):
		return (pd.read_csv('cfile.csv'), 
				pd.read_csv('cresults.csv'))

	return (None, None)


data, results = None, None
st.header('Time Series Anomaly Detection')


st.write('---')
with st.beta_container():
	col1, col2 = st.beta_columns(2)	

	with col1:

		image = Image.open('./images/data.png')
		st.image(image, use_column_width = True)


	with col2:
		st.subheader('Load Data')
		with st.form(key='load_dataset'):

			option = st.selectbox(
				'Select one of the datasets',
				list(data_path.keys()))
			
			
			download_button = st.form_submit_button(label='Download')



data, results = check_files()
if not (isinstance(data, pd.DataFrame) or download_button):
	st.stop()

elif download_button:
	data = pd.read_csv('./data/datasets/{}'.format(data_path[option]))
	results = pd.read_csv('./data/cache/cache_{}'.format(data_path[option]))

	data.to_csv('./cfile.csv', index=False)
	results.to_csv('./cresults.csv', index=False)



st.cache()
def vae_plot(animator, data, results):
	
	start_seq, end_seq = 0, data.shape[0]
	fig_data = animator.plot(

		data, data.columns.drop('anomaly'),
		start_seq, end_seq, ls='-'

	)

	fig_vae = animator.plot_results(

		results, data.columns.drop('anomaly')

	)

	return fig_data, fig_vae




animator = Animator()
fig1, fig2 = vae_plot(animator, data, results)


st.write('---')
with st.beta_container():

	#st.subheader('Outside view of the data')
	st.pyplot(fig1)


st.write('---')
with st.beta_container():
	st.pyplot(fig2)


st.write('---')
with st.beta_container():
	col1, col2 = st.beta_columns(2)	

	with col1:

		image = Image.open('./images/robot.png')
		st.image(image, use_column_width = True)


	with col2:
		
		st.subheader('Streaming Peaks-Over-Threshold (SPOT) Algorithm')
		with st.form(key='spot_parameters'):
			
			log_prob_name = st.radio(
				'Type of log_prob', 
				('log_prob_x', 'log_prob_z')
			)

			degree = st.slider(
				'Degree of risk parameter', -5, -1
			)


			n_train = st.slider(
				'The dimension of the training samples',
				int(0.5 * results.shape[0]),  results.shape[0]

			)


			n_init = st.slider(
				'The dimension of the calibration samples',
				int(0.05 * results.shape[0]), int(0.25 * results.shape[0])
			)
			
			run_button = st.form_submit_button(label='Run')


if not run_button:
 	st.stop()


def work_spot(results, log_prob_name, degree, n_train, n_init):

	q = 10 ** degree
	spot_init = results.loc[:n_init, log_prob_name]
	spot_data = results.loc[n_init:n_train, log_prob_name]

	spot = biSPOT(q)
	spot.fit(spot_init, spot_data)
	spot.initialize()


	#print(spot_init.shape, spot_data.shape)
	return spot, spot.run()



st.write('---')
with st.beta_container():
	
	
	spot, generator = work_spot(
		results, log_prob_name, 
		degree, n_train, n_init
	)
	
	progress_bar = st.progress(0)
	n_iterations = n_train - n_init + 1

	#print(results.shape)
	#print(n_iterations, n_train, n_init)
	
	for i in generator:
		progress_bar.progress(
			(i + 1) / n_iterations
		)


	plot_data = results.loc[n_init:n_train, [log_prob_name, 'anomaly']]
	plot_data.reset_index(drop=True, inplace=True)

	plot_data['thresholds'] = spot.markers['lower_thresholds']
	condition = plot_data.thresholds > plot_data.loc[:, log_prob_name]

	plot_data['pred_anomaly'] = 0
	plot_data.loc[condition, 'pred_anomaly'] = 1

	st.pyplot(animator.plot_spot(
		plot_data, log_prob_name
	))
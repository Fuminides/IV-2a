#!/usr/bin/env python3

'''	Loads the dataset 2a of the BCI Competition IV
available on http://bnci-horizon-2020.eu/database/data-sets
'''

import numpy as np
import scipy.io as sio

__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"

def get_data_graz(subject,training,PATH):
	'''	Loads the dataset 2a of the BCI Competition IV
	available on http://bnci-horizon-2020.eu/database/data-sets

	Keyword arguments:
	subject -- number of subject in [1, .. ,9]
	training -- if True, load training data
				if False, load testing data
	
	Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
			class_return 	numpy matrix 	size = NO_valid_trial
	'''
	NO_channels = 22
	NO_tests = 6*48 	
	Window_Length = 7*250 

	class_return = np.zeros(NO_tests)
	data_return = np.zeros((NO_tests,NO_channels,Window_Length))

	NO_valid_trial = 0
	try:
		#BCI IV dataset
		if training:
			a = sio.loadmat(PATH+'A0'+str(subject)+'T.mat')
		else:
			a = sio.loadmat(PATH+'A0'+str(subject)+'E.mat')
	except FileNotFoundError:
		try:
			#BCI IIIa dataset
			subject_iii = ['l1b', 'k3b', 'k6b'][subject]
			a = sio.loadmat(PATH+'train_subject'+str(subject_iii)+'_fft.mat')
		except (FileNotFoundError, IndexError):
			#Graz dataset
			if int(subject) < 10:
				subject = '0' + str(subject)
				
			if training:
				a = sio.loadmat(PATH+'S'+str(subject)+'T.mat')
			else:
				a = sio.loadmat(PATH+'S'+str(subject)+'E.mat')
			
			NO_channels = 15
			data_return = np.zeros((NO_tests,NO_channels,Window_Length))
		
	a_data = a['data']
	total_trials = 0
	for ii in range(0,a_data.size):
		a_data1	 = a_data[0,ii]
		a_data2	 = [a_data1[0,0]]
		a_data3	 = a_data2[0]
		a_X 		= a_data3[0]
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
		a_fs 		= a_data3[3]
		a_classes 	= a_data3[4]
		try:
			a_artifacts = a_data3[5]
			a_gender 	= a_data3[6]
			a_age 		= a_data3[7]
		except IndexError:
			a_artifacts = np.zeros(a_trial.size)
			a_trial 	= np.ravel(a_trial)
			a_y 		= np.ravel(a_y)
			
		for trial in range(0,a_trial.size):
			if(a_artifacts[trial]==0):
				data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
				class_return[NO_valid_trial] = int(a_y[trial])
				NO_valid_trial +=1
				
		   
	col_mean = np.nanmean(data_return, axis=0)
	inds = np.where(np.isnan(data_return))
	data_return[inds] = np.take(col_mean, inds[1])
	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]


def get_data_wcci(subject, training=True, PATH='./'):
	import scipy.io as sio
	Window_Length = 7*250 
	#subjects = 8
	res = []; y_res=[]
	global_NO_valid_trial = 0
    
	NO_channels = 12
	NO_tests = 20
	Window_Length = 7*250 

	class_return = np.zeros(NO_tests)
	data_return = np.zeros((NO_tests,NO_channels,Window_Length))

	NO_valid_trial = 0

	if subject < 10:
		subject = '0' + str(subject)

	if training:
		a = sio.loadmat(PATH+'parsed_P'+str(subject)+'T.mat')
	else:
		a = sio.loadmat(PATH+'B'+str(subject)+'E.mat')
        
	a = sio.loadmat(PATH+'parsed_P'+str(subject)+'T.mat')
	X = a['RawEEGData']
	y = a['Labels']
    
	return X, np.ravel(y) #np.eye(2)[np.array(np.ravel(y)-1, dtype='int32')]
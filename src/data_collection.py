# The data collection script for the Myo armband 
# was edited to provide a consistent 100 sample 
# window. At 50Hz, this is 2 seconds of data.
# Author: Tyler Talarico

import multiprocessing
from pyomyo import Myo, emg_mode
import os
import sys
import traceback
import queue
import numpy as np
import pandas as pd

def cls():
	# Clear the screen in a cross platform way
	# https://stackoverflow.com/questions/517970/how-to-clear-the-interpreter-console
    os.system('cls' if os.name=='nt' else 'clear')

# ------------ Myo Setup ---------------
imu_q = multiprocessing.Queue()
emg_q = multiprocessing.Queue()

def worker(imu_q, emg_q):
	m = Myo(mode=emg_mode.PREPROCESSED)
	m.connect()
 
	def add_to_imu_queue(quat, gyro, acc):
		imu_data = [quat, acc, gyro]
		imu_q.put(imu_data)
  
	def add_to_emg_queue(emg, movement):
		emg_q.put(emg)

	m.add_imu_handler(add_to_imu_queue)

	m.add_emg_handler(add_to_emg_queue)
	
	# Orange logo and bar LEDs
	m.set_leds([128, 128, 0], [128, 128, 0])

	
	"""worker function"""
	print("Collecting data...")
	while True:
		m.run()
	print("Worker Stopped")

# -------- Main Program Loop -----------
if __name__ == "__main__":
	# 2 seconds of data at 50Hz = 100 samples
	QUEUE_SIZE = 100

	data_dir = './data'

	# Arguments to provide easy data collection
	# python data_collection.py <participant_name> <gesture_name>
	args = sys.argv
	if len(args) != 3:
		print("Invalid number of args: must include participant name and gesture")
		exit()

	participant = args[1]
	action = args[2]
	
	prev_participants = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
	data_dir = os.path.join(data_dir, participant)

	if participant not in prev_participants:
		os.mkdir(data_dir)
	
	data_files = os.listdir(data_dir)
	data_files = [file for file in data_files if (participant + '_' + action) in file]

	# The sample count is iterated automatically
	data_point_idx = len(data_files) + 1
	file = participant + '_' + action + '_' + str(data_point_idx) + '.csv'


	p = multiprocessing.Process(target=worker, args=(imu_q,emg_q))
	p.start()

	imu_window = queue.Queue(QUEUE_SIZE)
	emg_window = queue.Queue(QUEUE_SIZE)

	try:
		while not(imu_window.full()) or not(emg_window.full()):
			while not (imu_q.empty()):
				imu = list()
				raw = list(imu_q.get())
				for el in raw:
					imu.extend(el)
				if not(imu_window.full()):
					imu_window.put(imu)

			while not (emg_q.empty()):
				emg = list(emg_q.get())
				if not(emg_window.full()):
					emg_window.put(emg)

		labels = [
			"EMG_Ch1","EMG_Ch2","EMG_Ch3","EMG_Ch4",
			"EMG_Ch5","EMG_Ch6","EMG_Ch7","EMG_Ch8",
			"Quat_1", "Quat_2", "Quat_3", "Quat_4",
			"Acc_x", "Acc_y", "Acc_z",
			"Gyro_x", "Gyro_y", "Gyro_z"
		]
		imu_data = np.array(imu_window.queue)

		emg_data = np.array(emg_window.queue)
		data = np.hstack((emg_data, imu_data))
		# Combine label names for the header of the csv file
		data = np.vstack((labels, data.astype(str)))

		np.savetxt(os.path.join(data_dir, file), data, delimiter=',', fmt='%s')
		p.terminate()
		quit()

	except KeyboardInterrupt:
		p.terminate()
		print("Quitting")
		quit()

	except Exception as e:
		traceback.print_exc()
		p.terminate()
		quit()

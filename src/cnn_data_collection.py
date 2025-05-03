import multiprocessing
from pyomyo import Myo, emg_mode
import os
import sys
import traceback
import queue
import numpy as np
import pandas as pd
import onnxruntime as rt
import vgamepad as vg
import tensorflow as tf

REVERSE_CLASSES = {0:'idle', 1:'kick', 2:'pass', 3:'walk'}  

def preprocess_data(data_array):
    """
    Preprocess the data by normalizing it
    :param data_array: numpy array of shape (n_samples, 100, 18)
    :return: normalized numpy array of shape (n_samples, 100, 18)
    """
    # Normalize the data
    data_array = (data_array - np.mean(data_array, axis=0)) / np.std(data_array, axis=0)

    return data_array

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
	# Vibrate to know we connected okay

	"""worker function"""
	print("Collecting data...")
	while True:
		m.run()
	print("Worker Stopped")

# -------- Main Program Loop -----------
if __name__ == "__main__":
	# gamepad = vg.VX360Gamepad()
	QUEUE_SIZE = 100
	SLIDE_SIZE = 10

	p = multiprocessing.Process(target=worker, args=(imu_q,emg_q))
	p.start()

	imu_window = queue.Queue(QUEUE_SIZE)
	emg_window = queue.Queue(QUEUE_SIZE)

	# load the model
	cwd = os.getcwd()
	saved_models_dir = os.path.join(cwd, 'src\SavedModels')
	model = tf.keras.models.load_model(os.path.join(saved_models_dir, '2sTimeWindow_CNN_model.keras'))
	# create a random input

	while True:
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
			
			# convert to float32
			data = data.astype(np.float32)

			preprocessed_data = preprocess_data(data)

			# convert to 3D array (100, 18, 1)
			data = np.reshape(preprocessed_data, (1, 100, 18, 1))
			# run the model
			preds = model.predict(data, batch_size=1, verbose=0)
			prediction_class = np.argmax(preds, axis=1)
			print(REVERSE_CLASSES[prediction_class[0]])
			# match preds[0]:
			# 	case 'Idle':
			# 		gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
			# 		gamepad.update()
			# 	case 'Walk':
			# 		gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
			# 		gamepad.update()
			# clean queue
			for i in range(SLIDE_SIZE):
				imu_window.queue.pop()
				emg_window.queue.pop()

		except KeyboardInterrupt:
			p.terminate()
			print("Quitting")
			quit()

		except Exception as e:
			traceback.print_exc()
			p.terminate()
			quit()

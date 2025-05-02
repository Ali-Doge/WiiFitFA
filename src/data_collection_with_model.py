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
	QUEUE_SIZE = 5

	p = multiprocessing.Process(target=worker, args=(imu_q,emg_q))
	p.start()

	imu_window = queue.Queue(QUEUE_SIZE)
	emg_window = queue.Queue(QUEUE_SIZE)

	# load the model
	cwd = os.getcwd()
	model_path = os.path.join(cwd, 'svm_model.onnx')
	sess = rt.InferenceSession(model_path)
	print(sess)
	input_name = sess.get_inputs()[0].name
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

			# # average the data over the queue
			# # convert to dataframe
			data = pd.DataFrame(data, columns=labels)
			# average the data over the queue

			# data = data.mean(axis=0)
			data = data.astype(np.float32)
			data = data.values.reshape(1, -1)

			# convert to numpy array
			# run the model
			preds = sess.run(None, {input_name: data})
			print('{}\x1b[2K\r'.format(preds[0]))
			# match preds[0]:
			# 	case 'Idle':
			# 		gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
			# 		gamepad.update()
			# 	case 'Walk':
			# 		gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
			# 		gamepad.update()
			# clean queue
			imu_window.queue.clear()
			emg_window.queue.clear()

		except KeyboardInterrupt:
			p.terminate()
			print("Quitting")
			quit()

		except Exception as e:
			traceback.print_exc()
			p.terminate()
			quit()

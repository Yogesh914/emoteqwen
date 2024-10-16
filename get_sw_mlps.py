import numpy as np
import pandas as pd 
import os
from tqdm import tqdm

fps_data = pd.read_csv("../data/all_metadata.csv")
landmarks_folder = "../data/facial_landmarks/"

all_landmarks = []
all_file_paths = []
all_landmarks_fourier = []
all_blendshapes = []
fft_path = []

def average_landmarks_per_second(landmarks, fps):
    num_frames = landmarks.shape[0]
    seconds = int(np.ceil(num_frames / fps))
    averaged_landmarks = []
    for i in range(seconds):
        start_frame = int(i * fps)
        end_frame = int(min((i+1) * fps, num_frames))
        averaged_landmarks.append(np.mean(landmarks[start_frame:end_frame], axis=0))
    return np.array(averaged_landmarks)


def sliding_window(data, window_size=30, step_size=1, padding=True):
    data_len = data.shape[0]
    windows = []

    if padding and data_len < window_size:
        padded_data = np.pad(data, ((0, window_size - data_len), (0, 0), (0, 0)), 'constant')
        #padded_data = np.pad(data, ((0, window_size - data_len), (0, 0)), 'constant')
        windows.append(padded_data)
    else: 
        for start in range(0, data_len - window_size + 1, step_size):
            windows.append(data[start:start + window_size, :])
    return np.array(windows)

def check(window):
    if window.ndim != 3:
        return False
    if window.shape[1] == 30 and window.shape[2] == 52:
        return True
    else:
        return False

npy_files = [f for f in os.listdir(landmarks_folder) if f.endswith('.npy')]

for file_name in tqdm(npy_files, desc="Processing data", unit="file"):

    file_path = os.path.join(landmarks_folder, file_name)
    landmarks_data = np.load(file_path, allow_pickle=True)

    for entry in landmarks_data:
        file_path = entry["file_path"]
        landmarks = entry["landmarks"]
        #blendshapes = entry["blendshapes"]

        if file_path in fps_data["file_path"].values:
            fps = fps_data[fps_data["file_path"] == file_path]['FPS'].values[0]
            averaged_landmarks = average_landmarks_per_second(landmarks, fps)
            #averaged_blendshapes = average_landmarks_per_second(blendshapes, fps)
            windows = sliding_window(averaged_landmarks, step_size=10)
            # fourier_windows = np.array([np.fft.fft(window, axis=0) for window in windows])
            # all_landmarks_fourier.append(fourier_windows)

            #all_blendshapes.append(windows)
            all_file_paths.extend([file_path] * windows.shape[0])
            all_landmarks.append(windows)
            
            #fft_path.append(file_path)
    

all_landmarks = np.concatenate(all_landmarks, axis=0)
all_file_paths = np.array(all_file_paths)
#all_blendshapes = np.concatenate(all_blendshapes, axis=0)
#fft_path = np.array(fft_path)

np.save("../data/facial_landmarks_dataset/window40_step10/landmarks.npy", all_landmarks, allow_pickle=True)
# np.save("../data/facial_landmarks_dataset/window30_step10/blendshapes.npy", all_blendshapes, allow_pickle=True)
np.save("../data/facial_landmarks_dataset/window40_step10/filepaths.npy", all_file_paths, allow_pickle=True)
# np.save("../data/facial_landmarks_dataset/window30_step10/filepaths_fft.npy", fft_path, allow_pickle=True)
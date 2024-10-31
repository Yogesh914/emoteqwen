import torch
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Initialize the FaceLandmarker
base_options = python.BaseOptions(model_asset_path='../models/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Function to process video and extract facial landmarks and blend shapes
def extract_landmarks_and_blendshapes_from_video(video_path, sample_rate=1):
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []
    all_blendshapes = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * sample_rate)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #if frame_count % frame_interval == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect face landmarks and blend shapes from the frame
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in face_landmarks]
            all_landmarks.append(landmarks)

        if detection_result.face_blendshapes:
            blendshapes = detection_result.face_blendshapes[0]
            blendshapes_scores = [bs.score for bs in blendshapes]
            all_blendshapes.append(blendshapes_scores)
            
        frame_count += 1

    cap.release()

    return np.array(all_landmarks), np.array(all_blendshapes)

# note: change the input file path to finish the rest
def main(input_csv_path="../data/celebv-data/split_celeb_df_10.csv", output_np_path="../data/celebv-landmarks/facial_landmarks_blendshapes_10.npy"):
    
    existing_dataset = pd.read_csv(input_csv_path)

    new_data = []
    for index, row in tqdm(existing_dataset.iterrows(), total=existing_dataset.shape[0]):
        video_path = row['file_path'] 
        #print(f"Reading {video_path}")
        try:
            
            all_landmarks, all_blendshapes = extract_landmarks_and_blendshapes_from_video(video_path)
        
            data_row = {
                "file_path": video_path,
                "landmarks": all_landmarks,
                "blendshapes": all_blendshapes,
            }
            new_data.append(data_row)
            
        except Exception as e:
            print(f"Error processing video: {video_path}")
            data_row = {
                "file_path": "error",
                "landmarks": "error",
                "blendshapes": "error",
            }
            new_data.append(data_row)
            continue
        
        # if (index + 1) % 1000 == 0:
        #      partial_output_path = f"../data/facial_landmarks/facial_landmarks_blendshapes_partial_{index+65001}.npy"
        #      np.save(partial_output_path, new_data)

        #      new_data = []
    
    np.save(output_np_path, new_data)
    
if __name__ == "__main__":
    main()
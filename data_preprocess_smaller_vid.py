import gc
from numpy import uint8
import os
import numpy as np
import cv2
from decord import cpu, gpu
from decord import VideoReader
import random

from tqdm import tqdm

FPS = 30  # frames/sec
EACH_VID_LEN = 10*FPS  # frames

def smaller_using_opencv():
    i = 0
    for vid_path in video_paths:
        cap = cv2.VideoCapture(vid_path)

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = 448 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = 448 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)

        path = os.path.join(processed_dataset_path, f"{i}.mp4")
        out = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        c = 0

        if not cap.isOpened():
            print("Something went wrong")
            exit()

        tqdm_loop = tqdm(total=int(num_frames/EACH_VID_LEN), desc=vid_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (H, W))
            out.write(frame)
            c += 1
            if c % EACH_VID_LEN == 0:
                i += 1
                c = 0
                out.release()

                path = os.path.join(processed_dataset_path, f"{i}.mp4")
                out = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

                tqdm_loop.update()

        cap.release()
        out.release()
        cv2.destroyAllWindows()


dataset_path = "datasets/MIX_DATA/"
processed_dataset_path = "datasets/VI_dataset_mix_448_10s/"
if not os.path.exists(processed_dataset_path):
    print('Making the directory', processed_dataset_path)
    os.makedirs(processed_dataset_path)

video_paths = [os.path.join(dataset_path, vid_path)
               for vid_path in os.listdir(dataset_path)]


# using_opencv()
smaller_using_opencv()
# input('Enter to end :')
print('Pre-processing Done')

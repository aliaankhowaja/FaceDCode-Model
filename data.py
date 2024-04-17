import cv2
import os
from utils.data_process import crop_video

# cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
# fourcc = cv2.VideoWriter_fourcc(*'hvc1')
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

for file in os.listdir("data/videos"):
    print(file)
    out = cv2.VideoWriter(f"data/final_data/{file}",fourcc, 30, (200, 200))
    vidObj = cv2.VideoCapture(f"data/videos/{file}")
    crop_video(out, vidObj)
    
    
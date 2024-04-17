import argparse
from utils.predict_vitals import predict_vitals
from utils.data_process import sample_data
from keras.models import load_model
import cv2
from utils.data_process import crop_video
import numpy as np

def inference(args):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vidObj = cv2.VideoCapture(args.video_path)
    args.video_path = f"inference_data/cropped/{args.video_path.split('/')[-1]}"
    out = cv2.VideoWriter(args.video_path,fourcc, 30, (200, 200))
    
    crop_video(out, vidObj)
    pulse, resp = predict_vitals(args)
    if len(pulse) < 150 :
        print("video must be greater than 5 seconds")
        return
    x, n = sample_data(pulse)
    x = np.array(x)
    model1 = load_model("model1.keras")
    model2 = load_model("model2.keras")
    y_2 = model2.predict(x)
    y_1 = model1.predict(x)
    print(y_1.mean(axis=0))
    print(y_2.mean(axis=0))
    print()
    print(y_1)
    print(y_2)
    """
        model.keras - 60 data points, 20 ep, 10 bs 
        pulse.keras - 50 data points, 10 ep, 10 bs
    """
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
    parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    args = parser.parse_args()
    inference(args)
    
import numpy as np
import cv2
from skimage.util import img_as_float
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import time
import scipy.io
from scipy.sparse import spdiags

def preprocess_raw_video(videoFilePath, dim=36):
    # while True:
    #     status, photo = cap.read()
    #     photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    #     face = model.detectMultiScale(photo, 1.1, 4)
    #     if len(face) >= 1:
    #         x = face[0][0]
    #         y = face[0][1]
    #         w = face[0][2]
    #         h = face[0][3]
    #         crop_img = photo[y:y+h, x:x+w]
    #         cropped = cv2.resize(crop_img, (200, 200))
    #     cv2.imshow("this is a test", cropped)
    #     # cv2.imshow("this is a test", photo)

    #     if cv2.waitKey(1) == 13:
    #         break
    #########################################################################
    # set up
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath)
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT)) # get total frame size
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype = np.float32)
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = vidObj.read()
    dims = img.shape
    print (success)
    # print("Orignal Height", height)
    # print("Original width", width)
    # cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    
    #########################################################################
    # Crop each frame size into dim x dim
    # while success:
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     # face = cascade.detectMultiScale(photo, 1.1, 4)
    #     # if len(face) >= 1:
    #     #     x = face[0][0]
    #     #     y = face[0][1]
    #     #     w = face[0][2]
    #     #     h = face[0][3]
    #     #     crop_img = img[y:y+h, x:x+w]
    #     #     cropped = cv2.resize(crop_img, (200, 200))
        
    #     t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))# current timestamp in milisecond
        
    #     vidLxL = cv2.resize(img_as_float(img[ int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]), (dim, dim), interpolation = cv2.INTER_AREA)
    #    # vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE) # rotate 90 degree
    #     vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
    #     vidLxL[vidLxL > 1] = 1
    #     vidLxL[vidLxL < (1/255)] = 1/255
    #     # cv2.imshow("fs", vidLxL)
    #     Xsub[i, :, :, :] = vidLxL
    #     success, img = vidObj.read() # read the next one
    #     i = i + 1
    #     print(i)
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))# current timestamp in milisecond
        # vidLxL = cv2.resize(img_as_float(img[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]), (dim, dim), interpolation = cv2.INTER_AREA)
        # vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE) # rotate 90 degree
        vidLxL = cv2.resize(img_as_float(img[:, :, :]), (dim, dim), interpolation = cv2.INTER_AREA)
        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1/255)] = 1/255
        Xsub[i, :, :, :] = vidLxL
        success, img = vidObj.read() # read the next one
        i = i + 1
    
    # plt.imshow(Xsub[0])
    # plt.title('Sample Preprocessed Frame')
    # plt.show()
    #########################################################################s
    # Normalized Frames in the motion branch
    normalized_len = len(t) - 1
    dXsub = np.zeros((normalized_len, dim, dim, 3), dtype = np.float32)
    for j in range(normalized_len - 1):
        dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j+1, :, :, :] + Xsub[j, :, :, :])
    dXsub = dXsub / np.std(dXsub)
    #########################################################################
    # Normalize raw frames in the apperance branch
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub  / np.std(Xsub)
    Xsub = Xsub[:totalFrames-1, :, :, :]
    #########################################################################
    # Plot an example of data after preprocess
    dXsub = np.concatenate((dXsub, Xsub), axis = 3)
    return dXsub

def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

# if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--video_path', type=str, help='processed video path')
    # parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
    # parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    # args = parser.parse_args()

    # preprocess_raw_video("C:\\Users\\Aliaan\\Development\\FaceDcode\\MTTS-CAN\\a copy.mp4")

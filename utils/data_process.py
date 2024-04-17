import numpy as np
import cv2
def sample_data(data, sample_size=150):
    data_points = len(data)
    num_samples = int(data_points/sample_size)
    # int((data_points - (data_points % sample_size)) /sample_size)
    data = data[:num_samples*sample_size]
    data = np.array_split(data, num_samples)
    return (data, num_samples)
def crop_video(out, vidObj):
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    success, img = vidObj.read()
    # i=0
    num = 1
    while success:
        photo = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = cascade.detectMultiScale(photo, 1.1, 4)
        if len(face) >= 1:
            x = face[0][0]
            y = face[0][1]
            w = face[0][2]
            h = face[0][3]
            crop_img = img[y:y+h, x:x+w]
            cropped = cv2.resize(crop_img, (200, 200))
            print(".", end="")
            out.write(cropped)
        # cv2.imshow("fsdf", cropped)
        success, img = vidObj.read() 
        # cv2.waitKey(1)
    
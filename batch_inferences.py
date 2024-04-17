import os
from utils import predict_vitals
import pandas as pd
import matplotlib.pyplot as plt

from argparse import Namespace
# from .code import predict_vitals
sys =   [125, 125, 110, 101, 122, 131, 115, 122]

dia =   [81, 81, 77, 67, 86, 86, 81, 75]
ht_ra = [76, 76, 94, 84, 85, 79, 69, 80]

data = {
    "1a":  [125, 81, 76],
    "1b":  [125, 81, 76],
    "2":   [110, 77, 94],
    "3":   [101, 67, 84],
    "4":   [122, 86, 85],
    "5":   [131, 86, 79],
    "6":   [115, 81, 69],
    "7":   [122, 75, 80],
    "8":   [136, 85, 76],
    "9":   [108, 59, 67],
    "10":  [128, 62, 76],
    "11":  [133, 81, 82],
    "12":  [122, 80, 81],
    "13":  [114, 76, 86],
    "14":  [153, 100, 93],
    "15":  [115, 77, 84],
    "16":  [127, 89, 66],
    "17":  [159, 87, 72],
    "18a": [106, 76, 94],
    "18b": [106, 76, 94],
    "19":  [162, 100, 66],
    "20":  [113, 70, 72],
    "21":  [136, 76, 64],
    "23":  [135, 77, 65],
    "24":  [142, 76, 93],
    "25":  [120, 67, 73],
    "26":  [112, 72, 78],
    "27":  [121, 63, 81],
    "28":  [130, 77, 62],
    "29":  [134, 87, 83],
    "30a": [129, 85, 73],
    "30b": [129, 85, 73],
    "32":  [137, 89, 78],
    "33a": [118, 79, 60],
    "33b": [118, 79, 60],
    "33c": [118, 79, 60],
    "34":  [126, 76, 69],
    "37":  [131, 70, 94],
    "38":  [125, 87, 64],
    "39":  [149, 80, 68],

}

i = 0
for key in data.keys():
    print(key)
for file in os.listdir("data/final_data"):
    print(file)
    file_name = file.split('.')[0]
    args = Namespace(video_path = f'C:/Users/Aliaan/Development/FaceDcode/MTTS-CAN/data/final_data/{file}', sampling_rate = 30, batch_size = 100)
    
    pulse_pred, resp_pred = predict_vitals.predict_vitals(args)
    data = {"pulse": pulse_pred, "resp": resp_pred, "sys": data[file_name][0], "dia": data[file_name][1], "hr": data[file_name][2]}
    df = pd.DataFrame(data)
    df.to_csv(f"data/preds/{file_name}.csv")
    print()
    # plt.subplot(211)
    # plt.plot(pulse_pred)
    # plt.title('Pulse Prediction')
    # plt.subplot(212)
    # plt.plot(resp_pred)
    # plt.title('Resp Prediction')
    # plt.show()
    i+=1
from sklearn.decomposition import FastICA
import heartpy as hp
from peakdetect import peakdetect
# from scipy.signal import find_peaks
# import plotly.graph_objects as go
# import plotly
from numpy.fft import fft, fftshift
# import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
# from matplotlib import pyplot as plt
# import numpy as np
# import pandas as pd


# video1 = "uploads/sample.mp4"

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def predictionmodel(video):
    cap = cv2.VideoCapture(video)  # considered a video

    co = 1
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            if (co >= 150):  # considering only the frames after first 10 seconds
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame)  # storing all the frames
            co += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    red = []
    green = []
    blue = []
    for i in range(len(all_frames)):
        r, g, b = (all_frames[i].mean(axis=0)).mean(axis=0)
        red.append(r)
        green.append(g)
        blue.append(b)

    t = [(i)/30 for i in range(0, len(all_frames))]  # time in seconds
    r_detrended = signal.detrend(red)
    g_detrended = signal.detrend(green)
    b_detrended = signal.detrend(blue)

    # if __name__ == "__main__":
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 30
    lowcut = 0.5
    highcut = 3.667

    datar = r_detrended
    datag = g_detrended
    datab = b_detrended

    yr = butter_bandpass_filter(datar, lowcut, highcut, fs, order=6)
    yg = butter_bandpass_filter(datag, lowcut, highcut, fs, order=6)
    yb = butter_bandpass_filter(datab, lowcut, highcut, fs, order=6)
    # =======================================================================
    # In[22]:

    S = np.c_[yr, yg, yb]

    S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations

    # In[23]:
    # Compute ICA
    ica = FastICA(n_components=3)
    S_ = ica.fit_transform(X)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix

    # We can `prove` that the ICA model applies by reverting the unmixing.
    assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

    models = [S_]
    names = [
        "ICA recovered signals"
    ]
    colors = ["red", "green", "blue"]

    for ii, (model, name) in enumerate(zip(models, names), 1):

        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.figure(figsize=(30, 10))
            plt.plot(sig, color=color)

    window = np.hamming(51)

    A = fft(window, 2048) / 25.5
    mag = np.abs(fftshift(S_))
    freq = np.linspace(-0.5, 0.5, len(S_))
    response = 20 * np.log10(mag)
    response = np.clip(response, -100, 100)

    models = [response]
    names = [
        "ICA recovered signals"
    ]
    colors = ["red", "green", "blue"]

    for ii, (model, name) in enumerate(zip(models, names), 1):

        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.figure(figsize=(30, 10))
            plt.plot(sig, color=color)

    ini_array1 = np.array(S_)

    # printing initial arrays
    # print("initial array", str(ini_array1))

    # Multiplying arrays
    result1 = ini_array1.flatten()

    # printing result
    # print("New resulting array: ", result1)

    # In[29]:

    ini_array1 = np.array(response)

    # printing initial arrays
    # print("initial array", str(ini_array1))

    # Multiplying arrays
    result2 = ini_array1.flatten()

    # printing result
    # print("New resulting array: ", result2)

    # In[30]:

    peaks = peakdetect(result1, lookahead=20)
    # Lookahead is the distance to look ahead from a peak to determine if it is the actual peak.
    # Change lookahead as necessary
    higherPeaks = np.array(peaks[0])
    lowerPeaks = np.array(peaks[1])
    peaks = peakdetect(result2, lookahead=20)
    # Lookahead is the distance to look ahead from a peak to determine if it is the actual peak.
    # Change lookahead as necessary
    higherPeaks = np.array(peaks[0])
    lowerPeaks = np.array(peaks[1])

    # run the analysis
    wd, m = hp.process(result1, sample_rate=4.5)

    print("\nHR: ", '%.3f' % m['bpm'])
    print("ibi: ", '%.3f' % m['ibi'])
    print("sdnn: ", '%.3f' % m['sdnn'])
    print("Breathing Rate in Hz: ", '%.3f\n' % m['breathingrate'])
    # print(m)
    return m['bpm'], m['breathingrate'], m['ibi'], m['sdnn']


# predictionmodel(video1)

import numpy as np



if __name__ == "__main__":
    anomalyScore = np.load('/media/dataHD3/kpugdeet/Correlation/data/anomalyScore.npy')
    print(anomalyScore.shape)

    divPoint = int(anomalyScore.shape[1]/2)

    filter1 = np.squeeze(anomalyScore[0])
    ori = filter1[:divPoint]
    adv = filter1[divPoint:]

    print(ori.shape)
    print(adv.shape)
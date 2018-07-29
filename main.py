import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib as mpl
mpl.use('Agg')
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
import tensorflow as tf
from graphviz import Digraph
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn import metrics

# For plotting
import plotly.plotly as py
import plotly.graph_objs as go
py.sign_in('krittaphat.pug', 'oVTAbkhd2RQvodGOwrwp') # G-mail link
# py.sign_in('amps1', 'Z1KAk8xiPUyO2U58JV2K') # kpugdeet@syr.edu/12345678
# py.sign_in('amps2', 'jGQHMBArdACog36YYCAI') # yli41@syr.edu/12345678
# py.sign_in('amps3', '5geLaNJlswmzDucmKikR') # liyilan0120@gmail.com/12345678

from alexnet import *
# from loadData import loadData
# from attribute import attribute
# from classify import classify
# from softmax import softmax

def loadAPYData():
    trainClass = pickle.load(open(FLAGS.BASEDIR + 'data/trainClass.pkl', 'rb'))
    trainX = np.load(FLAGS.BASEDIR + 'data/trainX.pkl.npy')
    trainY = np.load(FLAGS.BASEDIR + 'data/trainY.pkl.npy')
    valClass = pickle.load(open(FLAGS.BASEDIR + 'data/valClass.pkl', 'rb'))
    valX = np.load(FLAGS.BASEDIR + 'data/valX.pkl.npy')
    valY = np.load(FLAGS.BASEDIR + 'data/valY.pkl.npy')
    valY += len(trainClass)
    testClass = pickle.load(open(FLAGS.BASEDIR + 'data/testClass.pkl', 'rb'))
    testX = np.load(FLAGS.BASEDIR + 'data/testX.pkl.npy')
    testY = np.load(FLAGS.BASEDIR + 'data/testY.pkl.npy')
    testY += len(trainClass) + len(valClass)
    className = trainClass + valClass + testClass
    x = np.concatenate((trainX, valX, testX), axis=0)
    y = np.concatenate((trainY, valY, testY), axis=0)
    return className, x, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset Path
    parser.add_argument('--BASEDIR', type=str, default='/media/dataHD3/kpugdeet/Correlation/', help='Base folder for dataset and logs')

    # Image size
    parser.add_argument('--batchSize', type=int, default=100, help='Batch size')
    parser.add_argument('--width', type=int, default=227, help='Width')
    parser.add_argument('--height', type=int, default=227, help='Height')

    FLAGS, _ = parser.parse_known_args()

    # # Load Data
    # mapClass, imageX, imageY = loadAPYData()
    #
    # x = tf.placeholder(tf.float32, name='inputImage', shape=[None, FLAGS.height, FLAGS.width, 3])
    # cnnFeature = alexnet(x)
    #
    # # Tensorflow Session
    # tfConfig = tf.ConfigProto(allow_soft_placement=True)
    # tfConfig.gpu_options.allow_growth = True
    # sess = tf.Session(config=tfConfig)
    # sess.run(tf.global_variables_initializer())
    #
    # # # Choose index class 0 imagess
    # # chooseIndex = 0
    # # print('Choose: ' + mapClass[chooseIndex])
    # # eachInputX = []
    # # for i in range(0, imageX.shape[0]):
    # #     if imageY[i] == chooseIndex:
    # #         eachInputX.append(imageX[i])
    # # eachInputX = np.array(eachInputX)
    # eachInputX = imageX
    #
    # # Run output
    # outputCNN = sess.run(cnnFeature, feed_dict={x: eachInputX[:FLAGS.batchSize]})
    # for j in range(FLAGS.batchSize, eachInputX.shape[0], FLAGS.batchSize):
    #     xBatch = eachInputX[j:j + FLAGS.batchSize]
    #     outputCNN = np.concatenate((outputCNN, sess.run(cnnFeature, feed_dict={x: xBatch})), axis=0)
    #
    # # Visualization
    # reducedDim = 16
    # reducedOutput = np.zeros((outputCNN.shape[0], outputCNN.shape[1], outputCNN.shape[2], reducedDim))
    #
    # # For PCA
    # pca_data = outputCNN[:, 0, 1, :]
    # for i in range(outputCNN.shape[1]):
    #     for j in range(outputCNN.shape[2]):
    #         if i != 0 and j != 1:
    #             pca_data = np.concatenate((pca_data, outputCNN[:, i, j, :]), axis=0)
    # pca_data = np.array(pca_data)
    # print(pca_data.shape)
    # pca = PCA(n_components=reducedDim)
    # pca.fit(pca_data)
    #
    # for i in range(outputCNN.shape[1]):
    #     for j in range(outputCNN.shape[2]):
    #         reducedOutput[:, i, j, :] = pca.transform(outputCNN[:, i, j, :])
    #
    # np.save(FLAGS.BASEDIR + 'data/outputCNN.npy', outputCNN)
    # np.save(FLAGS.BASEDIR + 'data/reducedOutput.npy', reducedOutput)
    # np.save(FLAGS.BASEDIR + 'data/imageY.npy', imageY)

    # Load
    outputCNN = np.load(FLAGS.BASEDIR + 'data/outputCNN.npy')
    reducedOutput = np.load(FLAGS.BASEDIR + 'data/reducedOutput.npy')
    imageY = np.load(FLAGS.BASEDIR + 'data/imageY.npy')

    print(outputCNN.shape)
    print(reducedOutput.shape)
    print(imageY.shape)

    # flatten_reducedOutput = []
    # for i in range(reducedOutput.shape[1]):
    #     for j in range(reducedOutput.shape[2]):
    #         for k in range(reducedOutput.shape[3]):
    #             flatten_reducedOutput.append(reducedOutput[:, i, j, k].flatten())
    # flatten_reducedOutput = np.array(flatten_reducedOutput)
    # print(flatten_reducedOutput.shape)
    # correlateMatrix = np.zeros((flatten_reducedOutput.shape[0], flatten_reducedOutput.shape[0]))
    # for i in range(flatten_reducedOutput.shape[0]):
    #     for j in range(i+1, flatten_reducedOutput.shape[0]):
    #         correlateMatrix[i][j] = np.corrcoef(flatten_reducedOutput[i], flatten_reducedOutput[j])[0][1]
    # np.save(FLAGS.BASEDIR + 'data/reducePearson.npy', correlateMatrix)

    correlateMatrix = np.load(FLAGS.BASEDIR + 'data/reducePearson.npy')

    print(correlateMatrix.shape)
    correlateMatrix = np.round(correlateMatrix*2)/2


    plot_data = np.ma.masked_equal(correlateMatrix, 0)
    plt.figure(figsize=(20, 20))
    fig, ax = plt.subplots()
    cax = ax.imshow(plot_data, cmap=plt.cm.seismic, vmin=-1, vmax=1, interpolation="nearest")
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['-1', '0', '1'])  # vertically oriented colorbar
    plt.savefig('reducePearson.png')


    # # Heat Map
    # trace = go.Heatmap(z=correlateMatrix,
    #                    x=np.arange(correlateMatrix.shape[0]),
    #                    y=np.arange(correlateMatrix.shape[0]),
    #                    zmax=1.0,
    #                    zmin=-1.0
    #                    )
    # data = [trace]
    # layout = go.Layout(title='reducePearson', width=1920, height=1080)
    # fig = go.Figure(data=data, layout=layout)
    # py.image.save_as(fig, filename='reducePearson.png')

    # x = np.array([0, 0, 0, 0, 0, 0])
    # y = np.array([1, 2, 3, 4, 5, 6])

    # print(np.corrcoef(x, y))
    # print(metrics.normalized_mutual_info_score(x, y))


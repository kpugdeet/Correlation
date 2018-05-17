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
    parser.add_argument('--width', type=int, default=227, help='Width')
    parser.add_argument('--height', type=int, default=227, help='Height')

    FLAGS, _ = parser.parse_known_args()

    # Load Data
    mapClass, imageX, imageY = loadAPYData()

    x = tf.placeholder(tf.float32, name='inputImage', shape=[None, FLAGS.height, FLAGS.width, 3])
    cnnFeature = alexnet(x)

    # Tensorflow Session
    tfConfig = tf.ConfigProto(allow_soft_placement=True)
    tfConfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfConfig)
    sess.run(tf.global_variables_initializer())

    # Choose index class 0 images
    chooseIndex = 0
    print('Choose: ' + mapClass[chooseIndex])
    eachInputX = []
    for i in range(0, imageX.shape[0]):
        if imageY[i] == chooseIndex:
            eachInputX.append(imageX[i])
    eachInputX = np.array(eachInputX)

    # Run output
    outputCNN = sess.run(cnnFeature, feed_dict={x: eachInputX[:10]})

    # Visualization
    print(outputCNN.shape)
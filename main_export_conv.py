import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import pickle
import numpy as np
import tensorflow as tf
from alexnet import *

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

    # Load Data
    mapClass, imageX, imageY = loadAPYData()
    print(len(mapClass), imageX.shape, imageY.shape)
    print(mapClass)
    imageX = imageX[imageY == 20]
    imageY = imageY[imageY == 20]
    print(imageX.shape, imageY.shape)

    x = tf.placeholder(tf.float32, name='inputImage', shape=[None, FLAGS.height, FLAGS.width, 3])
    cnnFeature = alexnet(x, convs=True)

    # Tensorflow Session
    tfConfig = tf.ConfigProto(allow_soft_placement=True)
    tfConfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfConfig)
    sess.run(tf.global_variables_initializer())

    # Run
    outputCNN = sess.run(cnnFeature, feed_dict={x: imageX[:FLAGS.batchSize]})
    for j in range(FLAGS.batchSize, imageX.shape[0], FLAGS.batchSize):
        xBatch = imageX[j:j + FLAGS.batchSize]
        outputCNN = np.concatenate((outputCNN, sess.run(cnnFeature, feed_dict={x: xBatch})), axis=0)

    np.save(FLAGS.BASEDIR + 'data/outputCNN_maxpool5_monkey.npy', outputCNN)
    np.save(FLAGS.BASEDIR + 'data/imageY_maxpool5_monkey.npy', imageY)

    # # Load
    # outputCNN = np.load(FLAGS.BASEDIR + 'data/outputCNN_conv5.npy')
    # imageY = np.load(FLAGS.BASEDIR + 'data/imageY_conv5.npy')

    print(outputCNN.shape)
    print(imageY.shape)


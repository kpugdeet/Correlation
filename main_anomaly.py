import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import multiprocessing as mp

from alexnet import *
from caffe_classes import class_names

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

def fgm(model, x, eps=0.01, epochs=1, sign=True, clip_min=0., clip_max=1.):
    """
    Fast gradient method.
    See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533
    for details.  This implements the revised version since the original FGM
    has label leaking problem (https://arxiv.org/abs/1611.01236).
    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise use gradient value.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :return: A tensor, contains adversarial samples for each input.
    """
    xadv = tf.identity(x)
    xnoise = tf.identity(x)

    ybar = model(xadv)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = tf.abs(eps)

    def _cond(xadv, xnoise, i):
        return tf.less(i, epochs)

    def _body(xadv, xnoise, i):
        ybar, logits = model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, noise_fn(dy_dx), i+1

    xadv, xnoise, _ = tf.while_loop(_cond, _body, (xadv, xnoise, 0), back_prop=False,
                            name='fast_gradient')
    return xadv, xnoise

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.
    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid

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


    x = tf.placeholder(tf.float32, name='inputImage', shape=[None, FLAGS.height, FLAGS.width, 3])
    cnnProb = alexnet(x)


    x_fgm, x_noise = fgm(alexnet, x, epochs=1, eps=0.5, clip_min=0., clip_max=255.)
    convOutput = alexnet(x, convs=True)


    # Tensorflow Session
    tfConfig = tf.ConfigProto(allow_soft_placement=True)
    tfConfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfConfig)
    sess.run(tf.global_variables_initializer())


    # # Run Noise
    # advImg = sess.run(x_fgm, feed_dict={x: imageX[:FLAGS.batchSize]})
    # for j in range(FLAGS.batchSize, imageX.shape[0], FLAGS.batchSize):
    #     xBatch = imageX[j:j + FLAGS.batchSize]
    #     advImg = np.concatenate((advImg, sess.run(x_fgm, feed_dict={x: xBatch})), axis=0)
    #
    #
    # # Run conv output
    # imageX_advImg = np.concatenate((imageX, advImg), axis=0)
    # output_conv = sess.run(convOutput, feed_dict={x: imageX_advImg[:FLAGS.batchSize]})
    # for j in range(FLAGS.batchSize, imageX_advImg.shape[0], FLAGS.batchSize):
    #     xBatch = imageX_advImg[j:j + FLAGS.batchSize]
    #     output_conv = np.concatenate((output_conv, sess.run(convOutput, feed_dict={x: xBatch})), axis=0)
    #
    #
    # # Run prediction
    # output_prob = sess.run(cnnProb, feed_dict={x: imageX_advImg[:FLAGS.batchSize]})
    # for j in range(FLAGS.batchSize, imageX_advImg.shape[0], FLAGS.batchSize):
    #     xBatch = imageX_advImg[j:j + FLAGS.batchSize]
    #     output_prob = np.concatenate((output_prob, sess.run(cnnProb, feed_dict={x: xBatch})), axis=0)
    #
    # np.save(FLAGS.BASEDIR + 'data/output_conv.npy', output_conv)
    # np.save(FLAGS.BASEDIR + 'data/output_prob.npy', output_prob)

    output_conv = np.load(FLAGS.BASEDIR + 'data/output_conv.npy')
    output_prob = np.load(FLAGS.BASEDIR + 'data/output_prob.npy')

    print(output_conv.shape)
    print(output_prob.shape)

    import sys
    sys.exit(0)



    # Binary
    bins = np.array([0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0])
    ocb = np.digitize(output_conv, bins, right=True)
    ocb = ocb.astype(np.int32)

    anomalyScore = []
    for z in range(256):
        start_time = time.time()
        klDict0 = pickle.load(open('/media/dataHD3/kpugdeet/Correlation/data/klDict_conv5_' + str(z) + '.pickle', 'rb'), encoding='bytes')
        tmpArray = np.ascontiguousarray(ocb[:, :, :, z].reshape(ocb.shape[0], -1), dtype=np.int32)

        # Count Anomaly
        maxProcess = 11
        output = mp.Queue()

        def count_process(data, output_q):
            countTmp = []
            for choose_0 in range(data.shape[0]):
                count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for ti in range(data.shape[1]):
                    if data[choose_0][ti] != 0:
                        count[9] += 1
                        ansActivation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
                        for ai in range(1, ansActivation.shape[0]):
                            for fi in range(data.shape[1]):
                                keyIndex = fi + ti * data.shape[1] + data[choose_0][fi] * data.shape[1] * 9 + ai * data.shape[1] * 9 * 9
                                try:
                                    ansActivation[ai] += klDict0[keyIndex]
                                except KeyError:
                                    continue
                        elBest = max(ansActivation)
                        elT = ansActivation[data[choose_0][ti]]
                        anomaly = (elBest - elT) / elBest
                        if anomaly > 0.9:
                            count[0] += 1
                        elif anomaly > 0.8:
                            count[1] += 1
                        elif anomaly > 0.7:
                            count[2] += 1
                        elif anomaly > 0.6:
                            count[3] += 1
                        elif anomaly > 0.5:
                            count[4] += 1
                        elif anomaly > 0.4:
                            count[5] += 1
                        elif anomaly > 0.3:
                            count[6] += 1
                        elif anomaly > 0.2:
                            count[7] += 1
                        elif anomaly > 0.1:
                            count[8] += 1
                countTmp.append(count)
            countTmp = np.array(countTmp)
            output_q.put(countTmp)

        div = int(tmpArray.shape[0] / maxProcess)
        processes = [mp.Process(target=count_process, args=(tmpArray[l:l + div], output)) for l in range(0, tmpArray.shape[0], div)]
        for p in processes:
            p.start()
        results = [output.get() for p in processes]
        countArray = results[0]
        for i_1 in range(1, len(results)):
            countArray = np.concatenate((countArray, results[i_1]), axis=0)
        for p in processes:
            p.join()
        anomalyScore.append(countArray)
        elapsed_time = time.time() - start_time
        print(z, elapsed_time)

    anomalyScore = np.array(anomalyScore)
    print(anomalyScore.shape)
    np.save(FLAGS.BASEDIR + 'data/anomalyScore.npy', anomalyScore)
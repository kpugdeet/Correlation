import numpy as np
from numba import cuda
import pickle
import multiprocessing as mp
import time

if __name__ == "__main__":
    bin_outputCNN = np.load('/media/dataHD3/kpugdeet/Correlation/data/bin_outputCNN.npy')

    # sumCount = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    # for i in range(0, bin_outputCNN.shape[0], 1000):
    #     unique, counts = np.unique(bin_outputCNN[i:i + 1000], return_counts=True)
    #     sumCount += counts
    #     print(i)
    # print (np.asarray((unique, sumCount)).T)
    print(bin_outputCNN.shape, bin_outputCNN.dtype)

    @cuda.jit
    def countKlLinks(an_array, result):
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        bw = cuda.blockDim.x
        pos = tx + ty * bw

        while pos < an_array.shape[0]*an_array.shape[1]:
            x = pos / an_array.shape[1]
            y = pos % an_array.shape[1]
            for k in range(an_array.shape[1]):
                if y != k and an_array[x][y] != 0 and an_array[x][k] != 0:
                    calIndex =  y + k * an_array.shape[1] + an_array[x][y] * an_array.shape[1] * 9 + an_array[x][k] * an_array.shape[1] * 9 * 9
                    cuda.atomic.add(result, calIndex, 1)
            pos += cuda.gridDim.x * cuda.blockDim.x

    @cuda.jit
    def divKlLinks(an_array, ans_array, D1, D2, D3, D4):
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        bw = cuda.blockDim.x
        pos = tx + ty * bw

        while pos < ans_array.shape[0]:
            x = pos % D1
            y = ((pos - x) / D1) % D2
            z = ((pos - y * D1 - x) / (D1 * D2)) % D3
            t = ((pos - z * D2 * D1 - y * D1 - x) / (D1 * D2 * D3)) % D4
            sumLow = 0
            for j in range(D3):
                tmpIndex = x + y * D1 + j * D1 * D2 + t * D1 * D2 * D3
                sumLow += an_array[tmpIndex]

            ans_array[pos] = an_array[pos] / (sumLow * 1.0)
            pos += cuda.gridDim.x * cuda.blockDim.x

    for z in range(bin_outputCNN.shape[3]):
        start_time = time.time()

        # Select features layers to calculate KL links
        tmpArray = np.ascontiguousarray(bin_outputCNN[:,:,:,z].reshape(bin_outputCNN.shape[0], -1), dtype=np.int32)

        # Count KL Links
        count_klValue = np.zeros(tmpArray.shape[1] * tmpArray.shape[1] * 9 * 9, dtype=np.int32)
        countKlLinks[65535, 1024](tmpArray, count_klValue)
        print(count_klValue.shape, np.count_nonzero(count_klValue == 0), count_klValue.dtype)

        # Calculate posterior probability KL Links
        div_klValue = np.zeros(tmpArray.shape[1] * tmpArray.shape[1] * 9 * 9, dtype=np.float32)
        divKlLinks[65535, 1024](count_klValue, div_klValue, tmpArray.shape[1], tmpArray.shape[1], 9, 9)
        print(div_klValue.shape, np.count_nonzero(div_klValue == 0), count_klValue.dtype)

        # Reduce KL links
        maxProcess = 10
        output = mp.Queue()
        def reduce_process (data, output):
            tmpDict = dict()
            i = 0
            while i < data.shape[0]:
                if data[i] != 0:
                    tmpDict[i] = data[i]
                i += 1
            output.put(tmpDict)
        div = div_klValue.shape[0] / maxProcess
        processes = [mp.Process(target=reduce_process, args=(div_klValue[l:l+div], output)) for l in range (0, div_klValue.shape[0], div)]
        for p in processes:
            p.start()
        results = [output.get() for p in processes]
        klDict = dict()
        for eachDict in results:
            klDict.update(eachDict)
        for p in processes:
            p.join()
        pickle.dump(klDict, open('/media/dataHD3/kpugdeet/Correlation/data/klDict_' + str(z) + '.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        elapsed_time = time.time() - start_time
        print(z, len(klDict), elapsed_time)


    # outputCNN = np.load('/media/dataHD3/kpugdeet/Correlation/data/outputCNN_conv1.npy')
    # reducedOutput = np.load('/media/dataHD3/kpugdeet/Correlation/data/reducedOutput_conv1.npy')
    # imageY = np.load('/media/dataHD3/kpugdeet/Correlation/data/imageY_conv1.npy')
    #
    # print(outputCNN.shape)
    # print(reducedOutput.shape)
    # print(imageY.shape)
    # #
    # flatten_outputCNN = outputCNN.reshape(outputCNN.shape[0], -1)
    # flatten_outputCNN = flatten_outputCNN/np.amax(flatten_outputCNN)
    # print(flatten_outputCNN.shape)
    #
    # # Quantization testYAtt
    # bins = np.array([0.2, 0.4, 0.6, 0.8])
    # quant_outputCNN = np.digitize(flatten_outputCNN, bins, right=True)
    #
    #
    # @cuda.jit
    # def calDict(an_array, klDict):
    #     tx = cuda.threadIdx.x
    #     ty = cuda.blockIdx.x
    #     bw = cuda.blockDim.x
    #     pos = tx + ty * bw
    #
    #     while pos < an_array.shape[0]*an_array.shape[1]:
    #         x = pos / an_array.shape[1]
    #         y = pos % an_array.shape[1]
    #
    #         # Computation
    #         for k in range(an_array.shape[1]):
    #             if y != k:
    #                 calIndex =  y + k * an_array.shape[1] + an_array[x][y] * an_array.shape[1] * 5 + an_array[x][k] * an_array.shape[1] * 5 * 5
    #                 cuda.atomic.add(klDict, calIndex, 1)
    #
    #         pos += cuda.gridDim.x * cuda.blockDim.x
    #
    # klDict = np.zeros(quant_outputCNN.shape[1]*quant_outputCNN.shape[1]*5*5, dtype=np.int32)
    # calDict[65535, 1024](quant_outputCNN, klDict)
    # print(klDict.shape)
    # klDict = klDict.astype(np.float16)
    # np.save('/media/dataHD3/kpugdeet/Correlation/data/klDict_tmp.npy', klDict)

    # klDict_tmp = np.load('/media/dataHD3/kpugdeet/Correlation/data/klDict_tmp.npy')
    # klDict_tmp = klDict_tmp.astype(np.int32)
    # klDict = np.zeros(klDict_tmp.shape[0])
    # print(klDict_tmp.shape)
    # print(klDict.shape)
    #
    # @cuda.jit('void(int32[:], float64[:], int32, int32, int32, int32)')
    # def calDict(an_array, ans_array, D1, D2, D3, D4):
    #     tx = cuda.threadIdx.x
    #     ty = cuda.blockIdx.x
    #     bw = cuda.blockDim.x
    #     pos = tx + ty * bw
    #
    #     while pos < ans_array.shape[0]:
    #         x = pos % D1
    #         y = ((pos - x) / D1) % D2
    #         z = ((pos - y * D1 - x) / (D1 * D2)) % D3
    #         t = ((pos - z * D2 * D1 - y * D1 - x) / (D1 * D2 * D3)) % D4
    #         sumLow = 0
    #         for j in range(D3):
    #             tmpIndex = x + y * D1 + j * D1 * D2 + t * D1 * D2 * D3
    #             sumLow += an_array[tmpIndex]
    #         if sumLow == 0:
    #             ans_array[pos] = 0
    #         else:
    #             ans_array[pos] = an_array[pos]/(sumLow*1.0)
    #         pos += cuda.gridDim.x * cuda.blockDim.x
    #
    # step = int(1e8)
    # for i in range(0, klDict_tmp.shape[0], step):
    #     calDict[65535, 1024](klDict_tmp, klDict[i:step], 9216, 9216, 5, 5)
    # print(klDict[:20])
    # print(np.count_nonzero(klDict_tmp == 0))
    # np.save('/media/dataHD3/kpugdeet/Correlation/data/klDict.npy', klDict)

    # klDict = np.load('/media/dataHD3/kpugdeet/Correlation/data/klDict.npy')
    # print(klDict.shape, klDict.dtype, klDict.nbytes)
    #
    # outputCNN = np.load('/media/dataHD3/kpugdeet/Correlation/data/outputCNN_maxPool5.npy')
    # flatten_outputCNN = outputCNN.reshape(outputCNN.shape[0], -1)
    # flatten_outputCNN = flatten_outputCNN/np.amax(flatten_outputCNN)
    # bins = np.array([0.2, 0.4, 0.6, 0.8])
    # quant_outputCNN = np.digitize(flatten_outputCNN, bins, right=True)
    # print(quant_outputCNN.shape, quant_outputCNN.dtype)
    #
    #
    # @cuda.jit('void(int32[:], float64[:,:], int8[:], float64[:], int32, int32, int32, int32)')
    # def calActivation(input_array, ans_array, kl_i, kl_r, D1, D2, D3, D4):
    #     tx = cuda.threadIdx.x
    #     ty = cuda.blockIdx.x
    #     bw = cuda.blockDim.x
    #     pos = tx + ty * bw
    #
    #     while pos < ans_array.shape[0]*ans_array.shape[1]:
    #         ans_x = pos % D1
    #         ans_y = ((pos - ans_x) / D1) % D3
    #
    #         sumProb = 0
    #         for j in range(D1):
    #             if j != ans_x:
    #                 tmpIndex = j + ans_x * D1 + input_array[j] * D1 * D2 + ans_y * D1 * D2 * D3
    #                 sumProb += kl_r[kl_i[tmpIndex]]
    #         ans_array[ans_x, ans_y] = sumProb
    #         pos += cuda.gridDim.x * cuda.blockDim.x


    # klDict_index = np.zeros(klDict.shape[0], np.int8)
    # klDict_Reduce = []
    # print(klDict_index.shape, klDict_index.dtype)
    #
    # i = 0
    # while i < klDict.shape[0]:
    #     if klDict[i] != 0:
    #         klDict_index[i] = len(klDict_Reduce)
    #         klDict_Reduce.append(klDict[i])
    #     i += 1
    # klDict_Reduce = np.array(klDict_Reduce)
    # print(klDict_Reduce.shape, klDict_Reduce.dtype)
    #
    # np.save('/media/dataHD3/kpugdeet/Correlation/data/klDict_index.npy', klDict_index)
    # np.save('/media/dataHD3/kpugdeet/Correlation/data/klDict_Reduce.npy', klDict_Reduce)

    # klDict_index = np.load('/media/dataHD3/kpugdeet/Correlation/data/klDict_index.npy')
    # klDict_Reduce = np.load('/media/dataHD3/kpugdeet/Correlation/data/klDict_Reduce.npy')
    #
    # for k in range(quant_outputCNN.shape[0]):
    #     selectPic = k
    #     actArray = np.zeros(shape=(9216, 5))
    #     calActivation[65535, 1024](quant_outputCNN[selectPic], actArray, klDict_index, klDict_Reduce, 9216, 9216, 5, 5)
    #     for i in range(9216):
    #         anomaly = (max(actArray[i]) - actArray[i][quant_outputCNN[selectPic][i]])/max(actArray[i])
    #         if anomaly > 0.5:
    #             print(k, i, anomaly)





    # testAtt_D = np.zeros((tmp_Q.shape[0], tmp_Q.shape[1], bins.shape[0]))
    # for i in range(tmp_Q.shape[0]):
    #     testAtt_D[i][np.arange(tmp_Q.shape[1]), tmp_Q[i]] = 1
    # testAtt_D = testAtt_D.reshape(tmp_Q.shape[0], -1)


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
    # np.save(FLAGS.BASEDIR + 'data/13x13Pearson.npy', correlateMatrix)
    # correlateMatrix = np.load(FLAGS.BASEDIR + 'data/13x13Pearson.npy')
    # print(correlateMatrix.shape)
    # correlateMatrix = np.round(correlateMatrix*2)/2


    # flatten_Output = []
    # for i in range(outputCNN.shape[0]):
    #     tmpOut = []
    #     for j in range(outputCNN.shape[1]):
    #         for k in range(outputCNN.shape[2]):
    #             tmpOut.append(outputCNN[i, j, k, :])
    #     tmpOut = np.array(tmpOut)
    #     flatten_Output.append(tmpOut)
    # flatten_Output = np.array(flatten_Output)
    # print(flatten_Output.shape)
    # correlateMatrix = np.zeros((flatten_Output.shape[0], flatten_Output.shape[1], flatten_Output.shape[1]))
    # for i in range(outputCNN.shape[0]):
    #     print(i)
    #     for j in range(flatten_Output.shape[1]):
    #         for k in range(j + 1, flatten_Output.shape[1]):
    #             correlateMatrix[i][j][k] = np.corrcoef(flatten_Output[i][j], flatten_Output[i][k])[0][1]
    # correlateMatrix = np.nan_to_num(correlateMatrix)
    # np.save(FLAGS.BASEDIR + 'data/13x13Pearson.npy', correlateMatrix)
    # correlateMatrix = np.load(FLAGS.BASEDIR + 'data/13x13Pearson.npy')
    # print(correlateMatrix.shape)
    # correlateMatrix = np.round(correlateMatrix*2)/2

    # for i in range(10):
    #     plt.clf()
    #     plot_data = np.ma.masked_equal(correlateMatrix[i], 0)
    #     plt.figure(figsize=(20, 20))
    #     fig, ax = plt.subplots()
    #     cax = ax.imshow(plot_data, cmap=plt.cm.seismic, vmin=-1, vmax=1, interpolation="nearest")
    #     cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    #     cbar.ax.set_yticklabels(['-1', '0', '1'])  # vertically oriented colorbar
    #     plt.savefig('13x13Pearson_'+str(i)+'.png')
    #     plt.close()


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


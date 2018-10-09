import numpy as np
from numba import cuda
import pickle
import multiprocessing as mp
import time

if __name__ == "__main__":
    bin_outputCNN = np.load('/media/dataHD3/kpugdeet/Correlation/data/bin_outputCNN_maxpool5_monkey.npy')
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
                if y != k and not (an_array[x][y] == 0 and an_array[x][k] == 0):
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
        maxProcess = mp.cpu_count()
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
        pickle.dump(klDict, open('/media/dataHD3/kpugdeet/Correlation/data/klDict_maxpool5_monkey_' + str(z) + '.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        elapsed_time = time.time() - start_time
        print(z, len(klDict), elapsed_time)
        print('')


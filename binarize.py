import numpy as np

# Load
outputCNN = np.load('/media/dataHD3/kpugdeet/Correlation/data/outputCNN_conv1.npy')
imageY = np.load('/media/dataHD3/kpugdeet/Correlation/data/imageY_conv1.npy')
print(outputCNN.shape)
print(imageY.shape)

selectImage = []
for i in range(outputCNN.shape[0]):
	if imageY[i] == 20:
		selectImage.append(outputCNN[i])
selectImage = np.array(selectImage)

bins = np.array([0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0])
bin_outputCNN = np.digitize(selectImage, bins, right=True)
bin_outputCNN = bin_outputCNN.astype(np.int32)
unique, counts = np.unique(bin_outputCNN, return_counts=True)
print (np.asarray((unique, counts)).T)
print(bin_outputCNN.shape, bin_outputCNN.dtype)
np.save('/media/dataHD3/kpugdeet/Correlation/data/bin_outputCNN_monkey.npy', bin_outputCNN)
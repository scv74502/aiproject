import cv2
import matplotlib.pyplot as plt

im = cv2.imread('./Mel-Spectrogram example5.png')

plt.imshow(im[:,:100,:])
plt.show()

print(im.shape)
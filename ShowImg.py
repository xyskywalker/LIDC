import matplotlib.pyplot as plt
import numpy as np
import pickle

output_path = "output\\"
'''
imgs = np.load(output_path+'images_0002_0025.npy')
masks = np.load(output_path+'masks_0002_0025.npy')
for i in range(len(imgs)):
    print("image %d" % i)
    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    ax[0, 0].imshow(imgs[i], cmap='gray')
    ax[0, 1].imshow(masks[i],cmap='gray')
    ax[1,0].imshow(imgs[i]*masks[i],cmap='gray')
    plt.show()
'''


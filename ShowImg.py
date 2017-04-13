import matplotlib.pyplot as plt
import numpy as np
import pickle

output_path = "traindata\\"
imgfilename = output_path+'images_0025_0349.npy'
v_centerfilename = imgfilename.replace("images", "v_center")
img = np.load(imgfilename)
v_center = np.load(v_centerfilename)
plt.imshow(img)
plt.plot(int(v_center[0]), int(v_center[1]), 'ro', color='red')
plt.show()

plt.imshow(img)
plt.show()
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


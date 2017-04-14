import matplotlib.pyplot as plt
import numpy as np
import pickle
from glob import glob

train_file_path = "traindata\\"
train_file_list = glob(train_file_path+"images_0004_0086.npy")

print(train_file_list.__len__())

test_imgfilename = train_file_list[0]
test_v_centerfilename = test_imgfilename.replace("images", "v_center")
test_img = np.load(test_imgfilename)[0]
test_v_center = np.load(test_v_centerfilename)
print(test_v_center)
plt.imshow(test_img)
plt.plot(int(test_v_center[0]), int(test_v_center[1]), 'ro', color='red')
plt.show()

plt.imshow(test_img)
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


import matplotlib.pyplot as plt
import numpy as np
import pickle
from glob import glob

train_file_path = "traindata\\"
# train_file_list = glob(train_file_path+"images_0004_0086.npy")
train_file_list = glob(train_file_path+"images_????_????.npy")

def cutimg(img, centerX, centerY):
    if centerX < 106:
        startX = int(centerX) - 20
    elif centerX > 405:
        startX = int(centerX) + 20 - 299
    else:
        startX = 106

    if centerY < 106:
        startY = int(centerY) - 20
    elif centerY > 405:
        startY = int(centerY) + 20 - 299
    else:
        startY = 106

    maxX = startX + 299
    maxY = startY + 299

    # 中心点太靠上
    if centerY - startY < 20:
        startY -= 20
        maxY -= 20
    # 中心点太靠左
    if centerX - startX < 20:
        startX -= 20
        maxX -= 20
    # 中心点太靠下
    if maxY - centerY < 20:
        startY += 20
        maxY += 20
    # 中心点太靠右
    if maxX - centerX < 20:
        startX += 20
        maxX += 20

    return img[startY:maxY, startX:maxX], startX, startY

for i in range(9):
    test_imgfilename = train_file_list[1060 + i]
    test_v_centerfilename = test_imgfilename.replace("images", "v_center")
    test_img = np.load(test_imgfilename)[1]
    test_v_center = np.load(test_v_centerfilename)
    test_x = np.array(np.zeros(shape=[1, 299, 299], dtype=np.float32))
    test_y = np.array(np.zeros(shape=[1, 2], dtype=np.float32))

    test_x[0], startX, startY = cutimg(test_img, test_v_center[0], test_v_center[1])
    print(test_v_center[0], "-S-", test_v_center[1])
    print(startX, "-C-", startY)
    test_y[0][0] = test_v_center[0] - startX
    test_y[0][1] = test_v_center[1] - startY

    plt.imshow(test_x[0])
    plt.plot(int(test_y[0][0]), int(test_y[0][1]), 'ro', color='yellow')
    plt.show()

    plt.imshow(test_x[0])
    plt.show()

    plt.imshow(test_img)
    plt.plot(int(test_v_center[0]), int(test_v_center[1]), 'ro', color='yellow')
    plt.show()

    plt.imshow(test_img)
    plt.show()


'''
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


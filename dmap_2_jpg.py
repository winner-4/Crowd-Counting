import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

def dmap_npy_jpg(npy_path, set_xy=False):
    npy = np.load(npy_path, encoding='bytes', allow_pickle=True)
    # npy = cv2.cvtColor(npy, cv2.COLOR_RGB2BGR)
    text = 'GT Count:' + str(round(npy.sum()))
    npy = npy * 255
    h, w = npy.shape[0], npy.shape[1]

    # cv2 版本，有点问题
    # cv2.putText(npy, text, org=(10, h-10), fontScale=0.6,
    #             color=(1, 1, 1), thickness=1, fontFace=cv2.LINE_AA)
    # cv2.imshow('dmap', npy)
    # cv2.waitKey(20)

    # plt版本，图层会一直跳出来
    plt.ion()
    plt.text(20, h - 20, text, color=(1, 1, 1))
    if not set_xy:
        # plt.xticks([])
        # plt.yticks([])
        plt.axis('off')
    plt.imshow(npy, cmap=plt.cm.jet)
    plt.show()

    # save image
    plt.savefig(npy_path.replace('.npy', '.jpg'))
    plt.pause(0.2)
    plt.close()

def dmap_h5_jpg(h5_path, set_xy=False):


    h5 = h5py.File(h5_path, 'r')
    density = h5['density'][()]
    text = 'GT Count:' + str(round(density.sum()))
    density = density * 255
    h, w = density.shape[0], density.shape[1]
    plt.text(20, h - 20, text, color=(1, 1, 1))
    if not set_xy:
        # plt.xticks([])
        # plt.yticks([])
        plt.axis('off')
    plt.imshow(density, cmap=plt.cm.jet)
    plt.show()

    # save image
    plt.savefig(h5_path.replace('.h5', '.jpg'))
    plt.pause(0.2)
    plt.close()


if __name__ == '__main__':

    # ShangTechA 密度图的npy格式转换为jpg,并标注GT Count,并保存图片
    npy_dir =  r'data\part_A_final\train_data\density_map'
    npy_dir =  r'data\part_A_final\test_data\density_map'
    npy_paths = os.listdir(npy_dir)
    for npy_path in npy_paths:
        if '.npy' in npy_path:
            dmap_npy_jpg(os.path.join(npy_dir, npy_path), set_xy=False)


    # ShanghaiTechB密度图的h5格式转换为jpg,并标注GT Count,并保存图片
    train_path = r'data\part_B_final\train_data\density_map'
    test_path = r'data\part_B_final\test_data\density_map'
    h5_dir = [train_path, test_path]
    for dir in h5_dir:
        h5_paths = os.listdir(dir)
        for h5_path in h5_paths:
            if '.h5' in h5_path:
                dmap_h5_jpg(os.path.join(dir, h5_path),set_xy=False)


    # h5_path = r'E:\Crowd Counting\data\part_B_final\train_data\density_map\IMG_1.h5'
    # h5 = h5py.File(h5_path, 'r')
    # density = h5['density'][()]
    # text = 'GT Count:' + str(round(density.sum()))
    # density = density * 255
    # h, w = density.shape[0], density.shape[1]
    # plt.text(20, h - 20, text, color=(1, 1, 1))
    # plt.imshow(density, cmap=plt.cm.jet)
    # plt.show()

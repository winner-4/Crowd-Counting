import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import scipy.io as sio
from PIL import Image
def pmap_mat_jpg(mat_path, set_xy=False):

    # Shanghai Tech A train + Shanghai Tech B train / test
    mat = h5py.File(mat_path, 'r')
    pmap = np.transpose(mat['pmap'])

    # Shanghai Tech A test
    # mat = sio.loadmat(mat_path)
    # pmap = mat['pmap']


    if not set_xy:
        # plt.xticks([])
        # plt.yticks([])
        plt.axis('off')
    plt.imshow(pmap, cmap=plt.cm.jet)


    # save image
    plt.savefig(mat_path.replace('perspective_map', 'pmap_imgs').replace('.mat', '.jpg'))
    plt.pause(0.2)
    plt.close()


if __name__ == '__main__':

    # ShangTechA 透视图的mat格式转换为jpg,并保存图片
    mat_dir =  r'E:\Crowd Counting\data\part_A_final\train_data\perspective_map'
    # mat_dir =  r'E:\Crowd Counting\data\part_A_final\test_data\perspective_map'
    mat_paths = os.listdir(mat_dir)
    for mat_path in mat_paths:
        if '.mat' in mat_path:
            pmap_mat_jpg(os.path.join(mat_dir, mat_path), set_xy=False)
    plt.show()

    # ShangTechB 透视图的mat格式转换为jpg,并保存图片
    # mat_dir =  r'E:\Crowd Counting\data\part_B_final\train_data\perspective_map'
    # mat_dir = r'E:\Crowd Counting\data\part_B_final\test_data\perspective_map'
    # mat_paths = os.listdir(mat_dir)
    # for mat_path in mat_paths:
    #     if '.mat' in mat_path:
    #         pmap_mat_jpg(os.path.join(mat_dir, mat_path), set_xy=False)
    # plt.show()


import numpy as np
import argparse
import os
import sys
from ntu_read_skeleton import read_xyz
from numpy.lib.format import open_memmap
import pickle

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30


def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            ignored_sample_path=None):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
        
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
    # https://blog.csdn.net/u014630431/article/details/72844501

    for i, s in enumerate(os.listdir(data_path)):
        if s in ignored_samples:
            continue
        print (s)
        # print_toolbar(i * 1.0 / len(os.listdir(data_path)),
        #               '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
        #                   i + 1, len(os.listdir(data_path))))
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        fp = open_memmap(
            '{}/{}_data.npy'.format(out_path, s.split('.')[-2]),
            dtype='float32',
            mode='w+',
            shape=(3, max_frame, num_joint, max_body))
        fl = open_memmap(
            '{}/{}_num_frame.npy'.format(out_path, s.split('.')[-2]),
            dtype='int',
            mode='w+',
            shape=(1,))

        fp[:, 0:data.shape[1], :, :] = data
        fl[0] = data.shape[1] # num_frame

    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='/data0/NTU-RGB-D/nturgb+d_skeletons')
    parser.add_argument(
        '--ignored_sample_path',
        default='resource/NTU-RGB-D/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='/data0/NTU-RGB-D')
    arg = parser.parse_args()

    out_path = arg.out_folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    gendata(
        arg.data_path,
        out_path,
        arg.ignored_sample_path)
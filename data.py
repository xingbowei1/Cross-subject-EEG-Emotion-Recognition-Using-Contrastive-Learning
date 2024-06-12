import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio


def load_data():
    xtrain = np.random.random(size=[1000, 62, 5])
    xtest = np.random.random(size=[200, 62, 5])
    ytrain = np.random.randint(3, size=[1000])
    ytest = np.random.randint(3, size=[200])
    return xtrain, xtest, ytrain, ytest


def load_data_SEED_dependent(flag=1, flag_band=5, channel_t=1):
    # aaa=[[3,4],[0,3,4],[1,3,4],[2,3,4]]
    # channel=[[5,14,23,31],[0,5,14,22,23,31,32,44],[0,1,5,6,14,21,22,23,27,31,32,44],[0,1,2,5,6,14,15,21,22,23,26,27,31,32,44,58],[16,17,18,19,20,25,26,27,28,29,34,35,36,37,38,43,44,45,46,47]]
    features = ['psd', 'de', 'dasm', 'rasm', 'dcau']
    band = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'all']
    test_id = [9, 10, 11, 12, 13, 14]

    feature_data_train = np.zeros(shape=(1, 45), dtype=object)
    feature_data_test = np.zeros(shape=(1, 45), dtype=object)

    for i in range(15):  # 15
        for j in range(3):  # 2
            path = '/home/xingbowei/Preprocessed_SEED_Database/features_lds/' + \
                str(i + 1) + '_' + str(j + 1) + '.mat'
            data = scio.loadmat(path)
            data = data[features[flag]]
            data_test = data[:, test_id]

            data_train = np.delete(data, test_id, 1)
            p_data_train = np.concatenate((data_train[0, :]), axis=1)
            p_data_train = p_data_train.transpose([1, 0, 2])

            # p_data_train=p_data_train[:,channel[channel_t],:]

            if flag_band == 5:
                feature_data_train[0, i * 3 + j] = p_data_train
            else:
                feature_data_train[
                    0, i * 3 + j] = p_data_train[:, :, flag_band:flag_band + 1]
                # feature_data_train[0,i*2+j]=p_data_train[:,:,aaa[flag_band]]

            p_data_test = np.concatenate((data_test[0, :]), axis=1)
            p_data_test = p_data_test.transpose([1, 0, 2])

            # p_data_test=p_data_test[:,channel[channel_t],:]

            if flag_band == 5:
                feature_data_test[0, i * 3 + j] = p_data_test
            else:
                feature_data_test[
                    0, i * 3 + j] = p_data_test[:, :, flag_band:flag_band + 1]
                # feature_data_test[0,i*2+j]=p_data_test[:,:,aaa[flag_band]]

            if i == 0 & j == 0:
                sh_train = [data_train[0, m].shape[1]
                            for m in range(data_train.shape[1])]
                sh_test = [data_test[0, m].shape[1]
                           for m in range(data_test.shape[1])]

    path = '/home/xingbowei/Preprocessed_SEED_Database/label.mat'
    label = scio.loadmat(path)
    label = label['label']
    label = label.reshape(15, 1)

    label_train = np.delete(label, test_id, 0)
    label_train = label_train + 1
    label_train = label_train.repeat(sh_train, axis=0).flatten()

    label_test = label[test_id]
    label_test = label_test + 1
    label_test = label_test.repeat(sh_test, axis=0).flatten()
    return feature_data_train, feature_data_test, label_train, label_test

'''
def load_data_SEED_IV_dependent(flag_band=5, channel_t=1):
    band = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'all']
    test_id = [[12, 13, 16, 17, 20, 21, 22, 23], [12, 16, 18, 19, 20, 21, 22, 23], [10, 14, 16, 17, 20, 21, 22, 23]]

    feature_data_train = np.zeros(shape=(1, 45), dtype=object)
    feature_data_test = np.zeros(shape=(1, 45), dtype=object)
    label_train = np.zeros(shape=(1, 3), dtype=object)
    label_test = np.zeros(shape=(1, 3), dtype=object)
    sh_train = []
    sh_test = []

    for i in range(3):  # 15
        for j in range(15):  # 2
            path = 'D:/EEG-Datesets/SEED-IV/eeg_feature_smooth/1' + \
                str(i + 1) + '_' + str(j + 1) + '.mat'
            data = scio.loadmat(path)
            data = data['de']
            data_test = data[:, test_id[i]]

            data_train = np.delete(data, test_id[i], 1)
            p_data_train = np.concatenate((data_train[0, :]), axis=1)
            p_data_train = p_data_train.transpose([1, 0, 2])

            # p_data_train=p_data_train[:,channel[channel_t],:]

            if flag_band == 5:
                feature_data_train[0, j * 3 + i] = p_data_train
            else:
                feature_data_train[
                    0, j * 3 + i] = p_data_train[:, :, flag_band:flag_band + 1]
                # feature_data_train[0,i*2+j]=p_data_train[:,:,aaa[flag_band]]

            p_data_test = np.concatenate((data_test[0, :]), axis=1)
            p_data_test = p_data_test.transpose([1, 0, 2])

            # p_data_test=p_data_test[:,channel[channel_t],:]

            if flag_band == 5:
                feature_data_test[0, j * 3 + i] = p_data_test
            else:
                feature_data_test[
                    0, j * 3 + i] = p_data_test[:, :, flag_band:flag_band + 1]
                # feature_data_test[0,i*2+j]=p_data_test[:,:,aaa[flag_band]]

            if j == 0:
                sh_train.append([data_train[0, m].shape[1] for m in range(data_train.shape[1])])
                sh_test.append([data_test[0, m].shape[1]for m in range(data_test.shape[1])])

    for i in range(3):
        path = 'F:/desktop/eeggcn/data/SEEDIV Database/features_lds/label' + str(i + 1) + '.mat'
        label = scio.loadmat(path)
        label = label['label' + str(i + 1)]
        label = label.reshape(24, 1)

        label_train_t = np.delete(label, test_id[i], 0)
        label_train_t = label_train_t.repeat(sh_train[i], axis=0).flatten()

        label_test_t = label[test_id[i]]
        label_test_t = label_test_t.repeat(sh_test[i], axis=0).flatten()
        label_train[0, i] = label_train_t
        label_test[0, i] = label_test_t

    return feature_data_train, feature_data_test, label_train, label_test

'''

def load_data_SEED_independent(flag=1, flag_band=5, channel_t=1):
    # aaa=[[3,4],[0,3,4],[1,3,4],[2,3,4]]
    # channel=[[5,14,23,31],[0,5,14,22,23,31,32,44],[0,1,5,6,14,21,22,23,27,31,32,44],[0,1,2,5,6,14,15,21,22,23,26,27,31,32,44,58],[16,17,18,19,20,25,26,27,28,29,34,35,36,37,38,43,44,45,46,47]]
    features = ['psd', 'de', 'dasm', 'rasm', 'dcau']
    band = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'all']
    SUB = [1, 2, 1, 1, 1, 1, 1, 2, 1, 3, 3, 3, 2, 2, 3]

    feature_data = np.zeros(shape=(1, 15), dtype=object)

    for i in range(15):  # 15
        path = '/home/xingbowei/Preprocessed_SEED_Database/features_lds/' + \
            str(i + 1) + '_' + str(SUB[i]) + '.mat'
        data = scio.loadmat(path)
        data = data[features[flag]]

        p_data = np.concatenate((data[0, :]), axis=1)
        #exchange axis 1 and axis 0
        p_data = p_data.transpose([1, 0, 2])

        # p_data_train=p_data_train[:,channel[channel_t],:]

        if flag_band == 5:
            feature_data[0, i] = p_data

        if i == 0:
            sh = [data[0, m].shape[1]
                  for m in range(data.shape[1])]

    path = '/home/xingbowei/Preprocessed_SEED_Database/label.mat'
    label = scio.loadmat(path)
    label = label['label']
    label = label.reshape(15, 1)
    label = label + 1
    label = label.repeat(sh, axis=0).flatten()
    #flatten is
    return feature_data, label


def load_data_MPED(num=2, flag=4, flag_band=5, flag_class=2, channel_t=0):
    # emotion list:[neural,joy,funny,angry,fear,disgust,sadness],[positive,neutral,negative]
    # num:refer to protocol
    # flag:refer to features
    # flag_band:refer to band

    with open("D:/EEG-Datesets/MPED.txt", "r") as f:
        name = np.array([])
        for line in f.readlines():
            line = line.strip('\n')
            name = np.append(name, line)

    features = ['HHS', 'Hjorth', 'HOC', 'PSD', 'STFT']
    band = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'all']

    if num == 1:
        emotion = {1: [2, 8, 16, 21],  # neutral
                   2: [1, 9, 19, 20],  # joy
                   3: [4, 10, 26, 28],  # funny
                   4: [3, 13, 24, 25],  # angry
                   5: [5, 14, 17, 27],  # fear
                   6: [6, 11, 18, 23],  # disgust
                   7: [7, 12, 22, 29]}  # sadness
        class_index = [[2, 1, 4],
                       [2, 1, 5],
                       [2, 1, 6],
                       [2, 1, 7],
                       [3, 1, 4],
                       [3, 1, 5],
                       [3, 1, 6],
                       [3, 1, 7]]
        Pos, N, Neg = class_index[flag_class]
        train_id = emotion[Pos][:3] + emotion[N][:3] + emotion[Neg][:3]
        test_id = emotion[Pos][3:] + emotion[N][3:] + emotion[Neg][3:]

        feature_data_train = np.zeros(shape=(1, 23), dtype=object)
        feature_data_test = np.zeros(shape=(1, 23), dtype=object)

        for i in range(name.shape[0]):
            path = 'D:/EEG-Datesets/MPED/GSR_RSP_ECG_features/' + \
                features[flag] + '/' + name[i] + '.mat'  # F:/Desktop/毕设/
            data = scio.loadmat(path)

            if flag == 1 or flag == 2:
                data = data[band[flag_band]]
            elif flag == 3 or flag == 4:
                data = data[features[flag]]
            else:
                data = data['hhs_A']

            data_test = data[:, test_id]
            data_train = data[:, train_id]

            p_data_train = np.concatenate((data_train[0, :]), axis=1)
            p_data_train = p_data_train.transpose([1, 0, 2])
            feature_data_train[0, i] = p_data_train

            p_data_test = np.concatenate((data_test[0, :]), axis=1)
            p_data_test = p_data_test.transpose([1, 0, 2])
            feature_data_test[0, i] = p_data_test

            if i == 0:
                sh_train = [data_train[0, j].shape[1]
                            for j in range(data_train.shape[1])]
                sh_test = [data_test[0, j].shape[1]
                           for j in range(data_test.shape[1])]

    else:
        test_id = [20, 21, 23, 25, 27, 28, 29]

        feature_data_train = np.zeros(shape=(1, 23), dtype=object)
        feature_data_test = np.zeros(shape=(1, 23), dtype=object)

        for i in range(name.shape[0]):  # name.shape[0]
            path = 'D:/EEG-Datesets/MPED/GSR_RSP_ECG_features/' + \
                features[flag] + '/' + name[i] + '.mat'  # F:/Desktop/毕设/
            data = scio.loadmat(path)

            if flag == 1 or flag == 2:
                data = data[band[flag_band]]
            elif flag == 3 or flag == 4:
                data = data[features[flag]]
            else:
                data = data['hhs_A']

            data_test = data[:, test_id]
            data_train = np.delete(data, test_id + [0, 15], 1)

            p_data_train = np.concatenate((data_train[0, :]), axis=1)
            p_data_train = p_data_train.transpose([1, 0, 2])

            # p_data_train=p_data_train[:,channel[channel_t],:]
            feature_data_train[0, i] = p_data_train

            p_data_test = np.concatenate((data_test[0, :]), axis=1)
            p_data_test = p_data_test.transpose([1, 0, 2])

            # p_data_test=p_data_test[:,channel[channel_t],:]
            feature_data_test[0, i] = p_data_test

            if i == 0:
                sh_train = [data_train[0, j].shape[1]
                            for j in range(data_train.shape[1])]
                sh_test = [data_test[0, j].shape[1]
                           for j in range(data_test.shape[1])]

    # 协议一、二
    if num == 2 or num == 1:
        path = 'F:/desktop/eeggcn/data/MPED Database/label_3.mat'  # F:/Desktop/毕设/
        label = scio.loadmat(path)
        label = label['label_3']
        label = label.reshape(30, 1)

    # 协议三
    else:
        path = 'F:/desktop/eeggcn/data/MPED Database/label.mat'  # F:/Desktop/毕设/
        label = scio.loadmat(path)
        label = label['label']
        label = label.reshape(30, 1)

    if num == 1:
        label_train = label[train_id]
    else:
        label_train = np.delete(label, test_id + [0, 15], 0)
    label_train = label_train - 1
    label_train = label_train.repeat(sh_train, axis=0).flatten()

    label_test = label[test_id]
    label_test = label_test - 1
    label_test = label_test.repeat(sh_test, axis=0).flatten()

    return feature_data_train, feature_data_test, label_train, label_test

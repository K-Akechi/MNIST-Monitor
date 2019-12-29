import numpy as np
import matplotlib.pyplot as plt
import xlwt

if __name__ == '__main__':
    interValues_train = np.load('training_set_neuron_outputs.npy')
    interValues_test = np.load('test_set_neuron_outputs.npy')
    labels_train = np.load('training_set_labels.npy')
    weights_fc2 = np.load('weights_fc2.npy')
    weights_fc3 = np.load('weights_fc3.npy')
    conv1 = np.load('conv1.npy')
    # conv2 = np.load('conv2.npy')
    print(conv1.shape)
    print(conv1[0, 0, :, 0])
    print(np.mean(conv1[0], axis=(0, 1)).shape)
    print(np.mean(conv1[0], axis=(0, 1)))
    # print(conv1[1])

    # print(interValues_train.shape, interValues_test.shape)
    # max_train = np.max(interValues_train, axis=0)
    # min_train = np.min(interValues_train, axis=0)
    # max_test = np.max(interValues_test, axis=1)
    # min_test = np.min(interValues_test, axis=1)
    # print(max_train.shape, max_test.shape)
    # print(max_train, '\n', min_train)
    # print(max_test, '\n', min_test)
    # print(weights_fc2, '\n', weights_fc3)
    # vc = []
    # for i in range(10):
    #     vc.append(1)
    #     vc[i] = []
    #
    # for i in range(interValues_train.shape[0]):
    #     if labels_train[i] == 0:
    #         vc[0].append(interValues_train[i])
    #     elif labels_train[i] == 1:
    #         vc[1].append(interValues_train[i])
    #     elif labels_train[i] == 2:
    #         vc[2].append(interValues_train[i])
    #     elif labels_train[i] == 3:
    #         vc[3].append(interValues_train[i])
    #     elif labels_train[i] == 4:
    #         vc[4].append(interValues_train[i])
    #     elif labels_train[i] == 5:
    #         vc[5].append(interValues_train[i])
    #     elif labels_train[i] == 6:
    #         vc[6].append(interValues_train[i])
    #     elif labels_train[i] == 7:
    #         vc[7].append(interValues_train[i])
    #     elif labels_train[i] == 8:
    #         vc[8].append(interValues_train[i])
    #     else:
    #         vc[9].append(interValues_train[i])
    #
    # workbook1 = xlwt.Workbook()
    # sheet1 = workbook1.add_sheet('weights')
    # print(weights_fc3.shape)
    # for i in range(84):
    #     for j in range(10):
    #         sheet1.write(i, j, str(weights_fc3[i][j]))
    #
    #
    # sheet2 = workbook1.add_sheet('test range')
    # print(np.max(interValues_test, axis=0))
    # for i in range(84):
    #     sheet2.write(i, 0, str(np.max(interValues_test, axis=0)[i]))
    #     sheet2.write(i, 1, str(np.min(interValues_test, axis=0)[i]))
    #
    # sheet3 = workbook1.add_sheet('train range')
    # for i in range(84):
    #     for j in range(10):
    #         sheet3.write(i, j*2, str(np.max(vc[j], axis=0)[i]))
    #         sheet3.write(i, j*2+1, str(np.min(vc[j], axis=0)[i]))
    #
    # workbook1.save('values.xls')

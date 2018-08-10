
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, Conv1D, MaxPool1D, Lambda
from keras.layers import *
import pandas as pd

data = np.load('11785-hw2pt2/train-features.npy', encoding='bytes')
labels = np.load('11785-hw2pt2/train-labels.npy', encoding='bytes')
print('data and labels loaded')

def preprocess_single_label(label):  # input single label which is an array containing phoneme for each frame
    one_hot = []
    zeros = np.zeros((46, 1))
    for i in label:
        zeros = np.zeros((46))
        zeros[i] = 1
        one_hot.append(zeros)
    one_hot = np.array(one_hot)
    return one_hot



#
# dev_data = np.load('11785-hw2pt2/dev-features.npy', encoding='bytes')
# dev_labels = np.load('11785-hw2pt2/dev-labels.npy', encoding='bytes')

# dev_x_train = []
# dev_label = []
# for i in range(dev_data.shape[0]):
#     x = preprocess_single_x(dev_data[i])
#     y = preprocess_single_label(dev_labels[i])
#     x_train, y_train = final_data(x, y)
#     #     print(x_train.shape)
#
#     dev_x_train.append(x_train)
#     dev_label.append(y_train)
#
# dev_x_train = np.vstack(dev_x_train)
# # print(final_x_train.shape)
# dev_label = np.vstack(dev_label)


data_1 = np.array([data[i] for i in range(data.shape[0]) if data[i][0].shape[0] <= 1350])
label_1 = np.array([labels[i] for i in range(data.shape[0]) if data[i][0].shape[0] <= 1350])

def variable_len_pool(x, data):
    frame_list = []
    time_info = data[3][1]
    for i in range(len(time_info) - 1):  ### need to take into account that time info dont have the last frame number
        frame_list.append(np.mean(x[:, time_info[i]:time_info[i + 1]]))
    return np.asarray(frame_list)

#
# phoneme_list = ["+BREATH+", "+COUGH+", "+NOISE+", "+SMACK+", "+UH+", "+UM+", "AA", "AE", "AH", "AO", "AW",
#                 "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L",
#                 "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "SIL", "T", "TH", "UH", "UW", "V", "W", "Y",
#                 "Z", "ZH"]

def pad_cnn(x, max_len=1350):
    diff = max_len - x.shape[0]
    if diff == 0:
        return x
    if diff > 0:
        zeros = np.zeros((diff, 40))  # 40 mel spec
        return np.concatenate((x, zeros))


def mask(label, time_boundary, max_len=1350):
    one_hot_label = preprocess_single_label(label)
    mask_array = []
    for boundary in range(time_boundary.shape[0] - 1):
        mask_zeros = np.zeros(max_len)
        mask_zeros[time_boundary[boundary]:time_boundary[boundary + 1]] = 1
        mask_array.append(mask_zeros)
    return mask_array

def time_boundary_pair(data):
    # making pairs of start stop time boundary
    time_pair = []
    t = data[1]
    for i in range(t.shape[0]):
        if i == t.shape[0] - 1:
            time_pair.append((t[i], data[0].shape[0]))
        if i != t.shape[0] - 1:
            time_pair.append((t[i], t[i + 1]))
    return time_pair


def process_data(data, idx_1, idx_2, labels):
    time_list = []
    for i in range(data.shape[0]):
        if data[i][1].shape[0] > idx_2:
            time_list.append((data[i][1][idx_1], data[i][1][idx_2]))

    q = [(x, time_list.count(x)) for x in set(time_list)]
    q.sort(key=lambda x: x[1], reverse=True)
    batch_list = []
    for l in q:
        r = [(data[i], labels[i], l[0]) for i in range(data.shape[0]) if
             data[i][1].shape[0] > idx_2 and data[i][1][idx_1] == l[0][0] and data[i][1][idx_2] == l[0][1]]
        yield r
    #     batch_list.append(r)
    # return batch_list  # it contains mel spec,time_boundary,labels,where to cut


# defining global variables for slicing output of cnn
import time
import keras.backend as K
a = K.variable(0, dtype=np.int32)
b = K.variable(8, dtype=np.int32)

def crop(x):
    print(x[:, a:b, :, :].shape)
    return x[:, a:b, :, :]


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=(1350, 40, 1), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Lambda(crop, output_shape=(None, 40, 64)))
model.add(GlobalAveragePooling2D())
model.add(Dense(46, activation='softmax'))

model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy')



import gc

print('training started')
####  batched approach ##########
####  batched approach ##########
# for i in range(1):
gc.collect()
model.load_weights('2d_CNN_dot_weights_')
for i in range(1,800):
    print("CURRENTLY ON INDEX ",i)
    batch_gen=process_data(data_1,i,i+1,label_1)
    c=0
    q=0
    for batch in batch_gen: #range(len(w))
    #     print(len(batch))
    #     print('q',q)
    #     q+=len(batch)
        x=np.array([pad_cnn(instance[0][0]) for instance in batch])
        x=x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
        y=np.array([preprocess_single_label(instance[1])[i] for instance in batch])
        z=0
        for instance in batch:
            crop=instance[2]
            z+=1
            if z>=1:
                break
        K.set_value(a,crop[0])
        K.set_value(b,crop[1])
        model.fit(x,y,batch_size=16)
        c+=1

        gc.collect()
        if x.shape[0]<20:
            break
    # model.save('2d_CNN_dot_save_{}'.format(i))
    model.save_weights('2d_CNN_dot_weights_{}'.format(i))

# In[ ]:
#############################################################################################################333

# saving previous work before making new changes
####  batched approach ##########
# for i in range(1):
# i = 0
# batch_list = process_data(data_1, i, i + 1, label_1)
# for _ in range(len(batch_list)):  # range(len(w))
#     K.set_value(a, batch_list[_][0][2][0])
#     K.set_value(b, batch_list[_][0][2][1])
#     # print(a, b)
#     print('currently fitting {} batch for index {},{} and slice index are {},{}'.format(_,i,i+1,a,b))
#     x = np.array([pad_cnn(batch_list[0][i][0][0]) for i in range(len(batch_list[_]))])
#     x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
#     lab = np.vstack([preprocess_single_label(batch_list[0][i][1])[1] for i in range(len(batch_list[_]))])
#     model.fit(x, lab, batch_size=4)
#############################################################################################################333
#
# t1 = time.time()
# for _ in range(5):
#     for q in range(700, data_1.shape[0]):
#         if q % 100 == 0:
#             print("CURRENTLY ON {}TH UTTERANCE OUT OF {},EPOCH:{}".format(q, data_1.shape[0], _))
#             print("TOTAL TIME ELAPSED {} sec".format(time.time() - t1))
#         if q % 1000 == 0:
#             model.save("2d_cnn_model_dot_save")
#             model.save_weights("2d_cnn_model_dot_save_weights_{}_utt".format(q))
#         label_len = data_1[q][1].shape[0]
#         x_train = pad_cnn(data_1[q][0])
#         x_ = np.repeat(x_train.reshape(1, 1350, 40), label_len, axis=0)
#         y_ = preprocess_single_label(label_1[q])  # 57,46
#         for i, j, p in zip(x_, y_, time_boundary_pair(data_1[q])):
#             #     print(i.shape)
#             K.set_value(a, p[0])
#             K.set_value(b, p[1])
#             #             print(p)
#             model.fit(x=i.reshape(1, 1350, 40, 1), y=j.reshape(1, 46), verbose=0)
##########################################################################
#### PREDICTION PART  ###################################################
#     all_pred=[]
#     for q in range(dev_data.shape[0]):
#         pred_list=[]
#         label_len=dev_data[q][1].shape[0]
#         x_dev=pad_cnn(dev_data[q][0])
#         x_=np.repeat(x_dev.reshape(1,1350,40),label_len,axis=0)
#         y_=preprocess_single_label(dev_labels[q])  # 57,46
#         for i,j,p in zip(x_,y_,time_boundary_pair(dev_data[q])):
#             K.set_value(a,p[0])
#             K.set_value(b,p[1])
#     #         print(p)
#             pred_list.append(model.predict(x=i.reshape(1,2732,40)))
#         all_pred.append(pred_list)

#     acc=[]
#     for i in range(len(all_pred)):
#         p=np.argmax(np.vstack(all_pred[i]),axis=1)
#         t=dev_labels[i]
#         if p.shape == t.shape:
#             acc.append(np.sum(np.equal(p,t))/np.equal(p,t).shape[0])
#     print("accuracy is {}".format(np.array(acc).mean()))

#     model.save("1d_cnn_model_dot_save")
#     model.save_weights("1d_cnn_model_dot_save_weights_{}_acc".format(np.array(acc).mean()))


# In[ ]:


# model.save_weights("2d_cnn_model_dot_save_weights_5700_utt")

# In[ ]:

#
# dev_data_1 = np.array([dev_data[i] for i in range(dev_data.shape[0]) if dev_data[i][0].shape[0] <= 1350])
# dev_labels_1 = np.array([dev_labels[i] for i in range(dev_data.shape[0]) if dev_data[i][0].shape[0] <= 1350])

# In[ ]:


# dev_data_1.shape, dev_labels_1.shape

# In[ ]:


# all_pred = []
# for q in range(dev_data_1.shape[0]):
#     if q % 50 == 0:
#         print("predicting {} th dev".format(q))
#     pred_list = []
#     label_len = dev_data_1[q][1].shape[0]
#     x_dev = pad_cnn(dev_data_1[q][0])
#     x_ = np.repeat(x_dev.reshape(1, 1350, 40, 1), label_len, axis=0)
#     y_ = preprocess_single_label(dev_labels_1[q])  # 57,46
#     for i, j, p in zip(x_, y_, time_boundary_pair(dev_data_1[q])):
#         K.set_value(a, p[0])
#         K.set_value(b, p[1])
#         #         print(p)
#         pred_list.append(model.predict(x=i.reshape(1, 1350, 40, 1)))
#     all_pred.append(pred_list)

# In[ ]:


# np.argmax(pred)


# In[ ]:


# len(all_pred)

# In[ ]:

#
# acc = []
# for i in range(len(all_pred)):
#     p = np.argmax(np.vstack(all_pred[i]), axis=1)
#     t = dev_labels[i]
#     if p.shape == t.shape:
#         acc.append(np.sum(np.equal(p, t)) / np.equal(p, t).shape[0])
# print("accuracy is {}".format(np.array(acc).mean()))

# # accuracy is 0.579 for only 9500 utt of training data on dev set (1d cnn)
#

# # #accuracy is 0.4117 for only 5700 utt of training data on dev set (2d cnn) 15 hr training
#
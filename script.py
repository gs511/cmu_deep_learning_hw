import gc
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, Conv1D, MaxPool1D, Lambda
from keras.layers import *
import time
from keras import optimizers
import tensorflow as tf

def preprocess_single_label(label):  # input single label which is an array containing phoneme for each frame
    one_hot = []
    zeros = np.zeros((46, 1))
    for i in label:
        zeros = np.zeros((46))
        zeros[i] = 1
        one_hot.append(zeros)
    one_hot = np.array(one_hot)
    return one_hot


def variable_len_pool(x, data):
    frame_list = []
    time_info = data[3][1]
    for i in range(len(time_info) - 1):  ### need to take into account that time info dont have the last frame number
        frame_list.append(np.mean(x[:, time_info[i]:time_info[i + 1]]))
    return np.asarray(frame_list)

def pad_cnn(x, max_len=1350):
    diff = max_len - x.shape[0]
    if diff == 0:
        return x
    if diff > 0:
        zeros = np.zeros((diff, 40))  # 40 mel spec
        return np.concatenate((x, zeros))
# 

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

# 
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
a = K.variable(-2, dtype=np.int32)  # type should be int because we will use a and b for slicing tensors
b = K.variable(-1, dtype=np.int32)   # using -ve values just to be sure that these are changing durinng training .if not then it will throw error

# def crop(x):    #use this for 2d cnn
#     import tensorflow as tf
#     print(x[:, a:b, :, :].shape)
#     return x[:, a:b, :, :]

def crop(x):  #use this for 1d cnn
    import tensorflow as tf

#     print(tf.reshape(tf.reduce_mean(x[:,a:b,:],axis=1),(1,64)))
#     print(a,b)1
    return tf.reduce_mean(x[:,a:b,:],axis=1)
    

def two_d_cnn():
	model = Sequential()
	model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=(1350, 40, 1), activation='relu'))
	model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
	model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))

	# model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(Lambda(crop, output_shape=(None, 40, 256)))
	model.add(GlobalAveragePooling2D())
	model.add(Dense(46, activation='softmax'))
	return model

def one_d_cnn():
	model=Sequential()
	model.add(Conv1D(filters=32,kernel_size=11,padding='same',input_shape=(1350,40),activation='relu'))
	model.add(Conv1D(filters=64,kernel_size=7,padding='same',activation='relu'))
	model.add(Conv1D(filters=128,kernel_size=5,padding='same',activation='relu'))
	model.add(Conv1D(filters=256,kernel_size=3,padding='same',activation='relu'))
	model.add(Conv1D(filters=512,kernel_size=3,padding='same',activation='relu'))

	model.add(Lambda(crop,output_shape=(512,)))
	model.add(Dense(128,activation='relu'))
	model.add(Dense(46,activation='softmax'))
	return model

def predict(dev_data_1,dev_labels_1):
	all_pred=[]
	for q in range(dev_data_1.shape[0]):
	    if q%50==0:
	        print("predicting {} th dev".format(q))
	    pred_list=[]
	    label_len=dev_data_1[q][1].shape[0]
	    x_dev=pad_cnn(dev_data_1[q][0])
	    x_=np.repeat(x_dev.reshape(1,1350,40),label_len,axis=0)   # for 1d cnn
	    # x_=np.repeat(x_dev.reshape(1350,40,1),label_len,axis=0)  

	    y_=preprocess_single_label(dev_labels_1[q])  # 57,46

	    for i,j,p in zip(x_,y_,time_boundary_pair(dev_data_1[q])):
	        K.set_value(a,p[0])
	        K.set_value(b,p[1])
	#         print(p)
	        pred_list.append(model.predict(x=i.reshape(1,1350,40)))   # change shape to 1,1350,40 for 1d cnn
	    all_pred.append(pred_list)
	acc=[]
	for i in range(len(all_pred)):
	    p=np.argmax(np.vstack(all_pred[i]),axis=1)
	    t=dev_labels_1[i]
	    if p.shape == t.shape:
	        acc.append(np.sum(np.equal(p,t))/np.equal(p,t).shape[0])
    
	print("accuracy is {}".format(np.array(acc).mean()))
	return np.array(acc).mean()


########## main starts ###################3

model=one_d_cnn()
model.summary()
sgd = optimizers.SGD(lr=0.001)
model.compile(optimizer=sgd,loss='categorical_crossentropy')


data = np.load('11785-hw2pt2/train-features.npy', encoding='bytes')
labels = np.load('11785-hw2pt2/train-labels.npy', encoding='bytes')
dev_data=np.load('11785-hw2pt2/dev-features.npy',encoding='bytes')
dev_labels=np.load('11785-hw2pt2/dev-labels.npy',encoding='bytes')

dev_data_1=np.array([dev_data[i] for i in range(dev_data.shape[0]) if dev_data[i][0].shape[0]<=1350])
dev_labels_1=np.array([dev_labels[i] for i in range(dev_data.shape[0]) if dev_data[i][0].shape[0]<=1350])
data_1 = np.array([data[i] for i in range(data.shape[0]) if data[i][0].shape[0] <= 1350])
label_1 = np.array([labels[i] for i in range(data.shape[0]) if data[i][0].shape[0] <= 1350])

print('data and labels loaded')
print('training started')
####  batched approach ##########

gc.collect()
model.load_weights('1d_CNN_dot_weights_8')
# file=open("train_log.txt","w")
t=time.time()
for i in range(9,150):
    print("CURRENTLY ON INDEX ",i)
    batch_gen=process_data(data_1,i,i+1,label_1)
    c=0
    q=0
    batch_num=0
    for batch in batch_gen: #range(len(w))
    #     print(len(batch))
    #     print('q',q)
        q+=len(batch)
        batch_num+=1
        # if batch_num>20:
        # 	break
        x=np.array([pad_cnn(instance[0][0]) for instance in batch])
        # x=x.reshape(x.shape[0],x.shape[1],x.shape[2],1)   #uncomment for 2d cnn
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
        if i<10 and x.shape[0]==13:
            break
        if i<20 and x.shape[0]==8:
            break  
        if i<30 and x.shape[0]==5:
            break 
        if i<40 and x.shape[0]==2:
            break  
                  
        print("total time elapsed {} sec".format(time.time()-t))
        print("trained on {} utterances".format(q))
    with open("train_log_2d_cnn.txt","a") as file:
        file.write("trained on index {},using {} utterances.total time elapsed : {} sec +\n".format(i,q,time.time()-t))    

    if i%4==0:
        accuracy=predict(dev_data_1[:150],dev_labels_1[:150])

        with open("train_log_2d_cnn.txt","a") as file:
            file.write("trained on index {},using {} utterances.total time elapsed : {} sec,accuracy is {}+\n".format(i,q,time.time()-t,accuracy))    
    model.save('1d_CNN_dot_save_')
    model.save_weights('1d_CNN_dot_weights_{}'.format(i))


##########################################################################
#### PREDICTION PART  ###################################################
#
# dev_data = np.load('11785-hw2pt2/dev-features.npy', encoding='bytes')
# dev_labels = np.load('11785-hw2pt2/dev-labels.npy', encoding='bytes')


# dev_data_1 = np.array([dev_data[i] for i in range(dev_data.shape[0]) if dev_data[i][0].shape[0] <= 1350])
# dev_labels_1 = np.array([dev_labels[i] for i in range(dev_data.shape[0]) if dev_data[i][0].shape[0] <= 1350])




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

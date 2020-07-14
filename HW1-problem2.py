#!/usr/bin/env python
# coding: utf-8

# In[305]:


import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_filename = train_df['filename']
train_xmin = train_df['xmin']
train_xmax = train_df['xmax']
train_ymin = train_df['ymin']
train_ymax = train_df['ymax']

test_filename = test_df['filename']
test_xmin = test_df['xmin']
test_xmax = test_df['xmax']
test_ymin = test_df['ymin']
test_ymax = test_df['ymax']


# In[306]:


#處理train資料切割(3528)

#圖片壓縮至同一大小
IMG_SIZE = 80


train_x = []
train_index = []
#for i in range(len(train_filename)):
for i in range(3528):
    try:
        filepath = 'C:\\Users\\user\\Desktop\\images\\'+ train_filename[i]
        x1 = train_xmin[i]
        x2 = train_xmax[i]
        y1 = train_ymin[i]
        y2 = train_ymax[i]

#         img = cv2.imread(filepath)
        img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        name = 'img'+ str(i)
        sub_img = img [x1:x2, y1:y2]
        
        resized_sub_img = cv2.resize(sub_img, (IMG_SIZE, IMG_SIZE))
        train_x.append(resized_sub_img)
        train_index.append(i)
        
        save_path = 'C:\\Users\\user\\Desktop\\train_sub_images\\'+ name + '.jpg'
        cv2.imwrite(save_path, resized_sub_img)
    except:
        pass


# In[285]:


#處理test_data(393)
test_x = []
test_index = []
for i in range(len(test_filename)):
    try:
        filepath = 'C:\\Users\\user\\Desktop\\images\\'+ test_filename[i]
        x1 = test_xmin[i]
        x2 = test_xmax[i]
        y1 = test_ymin[i]
        y2 = test_ymax[i]

#         img = cv2.imread(filepath)
        img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        name = 'img'+ str(i)
        sub_img = img [x1:x2, y1:y2]
        
        resized_sub_img = cv2.resize(sub_img, (IMG_SIZE, IMG_SIZE))
        test_x.append(resized_sub_img)
        test_index.append(i)
        
        save_path = 'C:\\Users\\user\\Desktop\\test_sub_images\\'+ name + '.jpg'
        cv2.imwrite(save_path, resized_sub_img)
    except:
        pass


# In[312]:


# train / test data normailzation 
x_train = []
x_test = []
for x in train_x:
    x_normalize = x/255
    x_normalize = np.float32(x_normalize)
    x_train.append(x_normalize)
for x in test_x:
    x_normalize = x/255
    x_normalize = np.float32(x_normalize)
    x_test.append(x_normalize)


# encoding label to 0.1.2
y_train = []
y_test = []
train_label = train_df['label']
test_label = test_df['label']

for x in train_index:
    if train_label[x] == 'good':
        y_train.append(0)
    elif train_label[x] == 'bad':
        y_train.append(1)
    else:
        y_train.append(2)
for x in test_index:
    if test_label[x] == 'good':
        y_test.append(0)
    elif test_label[x] == 'bad':
        y_test.append(1)
    else:
        y_test.append(2)
        


# In[313]:


# onehot encode
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

onehot = OneHotEncoder(handle_unknown='ignore')
onehot.fit(y_train)
y_train = onehot.transform(y_train).toarray()
y_test = onehot.transform(y_test).toarray()


# In[314]:


y_train.shape


# In[315]:


# x_train分三類
x_train_good = []
x_train_bad = []
x_train_none = []
for x in range(len(train_index)):
    if train_label[train_index[x]] == 'good':
        x_train_good.append(x_train[x])
    elif train_label[train_index[x]] == 'bad':
        x_train_bad.append(x_train[x])
    else:
        x_train_none.append(x_train[x])

        
# x_test分三類
x_test_good = []
x_test_bad = []
x_test_none = []
for x in range(len(test_index)):
    if test_label[test_index[x]] == 'good':
        x_test_good.append(x_test[x])
    elif test_label[test_index[x]] == 'bad':
        x_test_bad.append(x_test[x])
    else:
        x_test_none.append(x_test[x])


# In[316]:


# reshape train, test good bad none
x_train_good = np.array(x_train_good).reshape(-1,IMG_SIZE, IMG_SIZE, 1)
x_train_bad = np.array(x_train_bad).reshape(-1,IMG_SIZE, IMG_SIZE, 1)
x_train_none = np.array(x_train_none).reshape(-1,IMG_SIZE, IMG_SIZE, 1)

x_test_good = np.array(x_test_good).reshape(-1,IMG_SIZE, IMG_SIZE, 1)
x_test_bad = np.array(x_test_bad).reshape(-1,IMG_SIZE, IMG_SIZE, 1)
x_test_none = np.array(x_test_none).reshape(-1,IMG_SIZE, IMG_SIZE, 1)


# In[317]:


# y_train分三類
y_train_good = []
y_train_bad = []
y_train_none = []
for x in range(len(train_index)):
    if train_label[train_index[x]] == 'good':
        y_train_good.append(y_train[x])
    elif train_label[train_index[x]] == 'bad':
        y_train_bad.append(y_train[x])
    else:
        y_train_none.append(y_train[x])

        
# y_test分三類
y_test_good = []
y_test_bad = []
y_test_none = []
for x in range(len(test_index)):
    if test_label[test_index[x]] == 'good':
        y_test_good.append(y_test[x])
    elif test_label[test_index[x]] == 'bad':
        y_test_bad.append(y_test[x])
    else:
        y_test_none.append(y_test[x])


# In[318]:


# onehot encode good bad none
y_train_good = np.array(y_train_good).reshape(-1, 3)
y_train_bad = np.array(y_train_bad).reshape(-1, 3)
y_train_none = np.array(y_train_none).reshape(-1, 3)
y_test_good = np.array(y_test_good).reshape(-1, 3)
y_test_bad = np.array(y_test_bad).reshape(-1, 3)
y_test_none = np.array(y_test_none).reshape(-1, 3)


# In[319]:


# reshape train, test data
x_train = np.array(x_train).reshape(-1,IMG_SIZE, IMG_SIZE, 1)
x_test = np.array(x_test).reshape(-1,IMG_SIZE, IMG_SIZE, 1)


# In[349]:


x_train.shape


# In[444]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# tf.set_random_seed(1)
# np.random.seed(1)

batch_size = 32
LR = 0.001

def next_batch(train_data, train_target, batch_size):  
    index = [ i for i in range(0,len(train_target)) ]  
    np.random.shuffle(index);  
    batch_data = []; 
    batch_target = [];  
    for i in range(0,batch_size):  
        batch_data.append(train_data[index[i]]);  
        batch_target.append(train_target[index[i]]) 
#     print("batch_data",batch_data[0].shape,"batch_target", batch_target[0].shape )
    return batch_data, batch_target 




# # define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1])/255. 
ys = tf.placeholder(tf.float32, [None, 3])
keep_prob = tf.placeholder(tf.float32)
# # print(x_image.shape)  # [n_samples, 100,100,1] [n_samples, 80,80,1]





# # CNN1
# conv1 = tf.layers.conv2d(inputs=xs, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)           
# pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)           # -> (50,50, 32)
# conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu)    # -> (50,50, 64)
# pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (25,25, 64)


# flat = tf.reshape(pool2, [-1, int(IMG_SIZE/4*IMG_SIZE/4*64)])
# dropout = tf.layers.dropout(flat, rate=0.5, noise_shape=None, seed=None, training=False, name=None)


# # CNN2 dropout
# conv1 = tf.layers.conv2d(inputs=xs, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)           
# pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)          
# conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=tf.nn.relu)   
# pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    


# flat = tf.reshape(pool2, [-1, int(IMG_SIZE/4*IMG_SIZE/4*64)])
# dropout = tf.layers.dropout(flat, rate=0.7, noise_shape=None, seed=None, training=False, name=None)


# # CNN3 stride
# conv1 = tf.layers.conv2d(inputs=xs, filters=32, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)           
# pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)         
# conv2 = tf.layers.conv2d(pool1, 64, 5, 2, 'same', activation=tf.nn.relu)   
# pool2 = tf.layers.max_pooling2d(conv2, 2, 2)   


# flat = tf.reshape(pool2, [-1, int(IMG_SIZE/16*IMG_SIZE/16*64)])
# dropout = tf.layers.dropout(flat, rate=0.5, noise_shape=None, seed=None, training=False, name=None)



# CNN4 filter
conv1 = tf.layers.conv2d(inputs=xs, filters=64, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)           
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)         
conv2 = tf.layers.conv2d(pool1, 64, 5, 2, 'same', activation=tf.nn.relu)   
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)   
flat = tf.reshape(pool2, [-1, int(IMG_SIZE/16*IMG_SIZE/16*64)])



dropout = tf.layers.dropout(flat, rate=0.5, noise_shape=None, seed=None, training=False, name=None)



# -> (25*25*32, )
output = tf.layers.dense(flat, 3)              # output layer

# compute cost
loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=output)           
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

# return (acc, update_op), and create 2 local variables
accuracy = tf.metrics.accuracy(labels=tf.argmax(ys, axis=1), predictions=tf.argmax(output, axis=1))[1]
confusion = tf.confusion_matrix(labels=tf.argmax(ys, axis=1), predictions=tf.argmax(output, axis=1), num_classes=3)


sess = tf.Session()
# the local var is for accuracy_op
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
# initialize var in graph
sess.run(init_op)     



# Parameters
training_iters = len(x_train)
epochs = 50


# In[445]:


y_train_pred_ans = []
y_test_pred_ans = []
learning_curve = []
training_acc = []
testing_acc = []


for epoch in range(epochs):
    for step in range(70):
        b_x, b_y = next_batch(x_train, y_train, batch_size)
        _, loss_ = sess.run([train_op, loss], feed_dict={xs: b_x, ys:b_y})
        
#         print("1")
        if step % 50 == 0:
#             test_batch, y_batch = next_batch(x_test, y_test, batch_size)
#             accuracy_, flat_representation = sess.run([accuracy, flat], {xs: test_batch, ys: y_batch})
#             x_test, y_test = test_batch(x_test, y_test)
            accuracy_, flat_representation = sess.run([accuracy, flat], {xs: x_test, ys: y_test})
            32
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: ', accuracy_)
            accuracy_, flat_representation = sess.run([accuracy, flat], {xs: x_train[:100], ys: y_train[:100]})
           
            print('train accuracy:',accuracy_)
#         step += 1
    print("epoch:", epoch)
    learning_curve.append(loss_)
    
    print(' *TRAIN*')
    train_con = sess.run(confusion, {xs: x_train, ys:y_train})
    print(train_con)
    training_acc.append((train_con[0,0]+train_con[1,1]+train_con[2,2])/2518)
    print('  good:', train_con[0,0]/(train_con[0,0]+train_con[0,1]+train_con[0,2]))
    print('  bad:', train_con[1,1]/(train_con[1,0]+train_con[1,1]+train_con[1,2]))
    print('  none:', train_con[2,2]/(train_con[2,0]+train_con[2,1]+train_con[2,2]))
    
    print(' *TEST*')
    test_con = sess.run(confusion, {xs: x_test, ys:y_test})
    print(test_con)
    testing_acc.append((test_con[0,0]+test_con[1,1]+test_con[2,2])/278)
    print('  good:', test_con[0,0]/(test_con[0,0]+test_con[0,1]+test_con[0,2]))
    print('  bad:', test_con[1,1]/(test_con[1,0]+test_con[1,1]+test_con[1,2]))
    print('  none:', test_con[2,2]/(test_con[2,0]+test_con[2,1]+test_con[2,2]))


# In[401]:


print(sess.run(confusion, {xs: x_test, ys: y_test}))
print(sess.run(confusion, {xs: x_train, ys: y_train}))


# In[447]:


testing_acc[-1]


# In[450]:


# plt.plot(learning_curve, label='learning_curve')
# plt.plot(training_acc, label='training_acc')
plt.plot(testing_acc, label='testing_acc')


# In[426]:


training_acc_2 = []
for i in range(len(training_acc)):
    if i % 2 != 0:
        training_acc_2.append(training_acc[i])


# In[428]:


training_acc


# In[427]:


plt.plot(training_acc_2, label='testing_acc')


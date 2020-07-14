#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix


# In[ ]:


#12000筆資料
train_data = np.load('train.npz')
#print(train_data['image'][1])
print(len(train_data['label']))

x_train = train_data['image']
y_train = train_data['label']


# In[ ]:


#5768筆資料
test_data = np.load('test.npz')
print(len(test_data['label']))
x_test = test_data['image']
y_test = test_data['label']


# In[ ]:


# onehot encoder
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y_train)
y_train = enc.transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
y_train = y_train.reshape(y_train.shape[0], -1, 1)
y_test = y_test.reshape(y_test.shape[0], -1, 1)


# In[ ]:


# normalize
x_train = x_train/255
x_test = x_test/255


# In[ ]:


# 配對 data and label
training_data = list(zip(x_train, y_train))
testing_data = list(zip(x_test, y_test))


# In[ ]:


#激活函數
def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def sigmoid_derivate(x):
    return sigmoid(x) * (1-sigmoid(x))


#loss function
def cross_entropy(output, correct):
    return np.sum( np.nan_to_num(-(correct*np.log(output) + (1-correct)*np.log(1-output))))

def cross_entropy_derivative(output, correct):
    return output - correct


# In[ ]:


# initial variable = 0
class model_DNN_zero():
    def __init__(self, layers):
        #設定有幾層神經網路
        self.layer_num = len(layers)
        self.neurons = layers
        # create weights and bias
        self.weights = [ np.zeros((j, i)) for i, j in zip(layers[:-1], layers[1:]) ]
        self.biases = [ np.zeros((i, 1)) for i in layers[1:] ]
         
        self.training_loss = []
        self.training_error_rate = []
        self.testing_error_rate = []
        self.batches_latent = []
        self.latent_count = 0
        self.latent = []
        
        
        #np.random.seed(42)
        #weights = np.random.rand(3,1)
        #bias = np.random.rand(1)
        #lr = 0.05 #learning rate
        
        
    def SGD(self, training_data, testing_data, epochs, batch_size, lr):
        #總資料量
        total_data_num = len(training_data)
        self.training_loss = []
        self.training_error_rate = []
        self.testing_error_rate = []
        
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            # 把資料切割成 mini_batch
            mini_batch = [ training_data[i : i + batch_size] for i in range(0, total_data_num, batch_size) ] 
            for single_data in mini_batch:
                self.update_mini_batch(single_data, lr)
                
            #每50個epoch顯示一次
            if (epoch % 50 == 0):
                # record info
                self.training_loss.append(self.calc_loss(training_data))
                self.training_error_rate.append(self.count_error(training_data) / len(training_data))
                self.testing_error_rate.append(self.count_error(testing_data) / len(testing_data))
                print('===================================')
                print("Epoch:" ,epoch) 
                print('    training loss: %f' % self.calc_loss(training_data))
                print('    training error rate: %d <-- %d ,rate=(%f)' % (self.count_error(training_data), len(training_data), self.count_error(training_data) / len(training_data)))
                print('    testing error rate: %d <-- %d ,rate=(%f)' % (self.count_error(testing_data), len(testing_data), self.count_error(testing_data) / len(testing_data)))
                
    def update_mini_batch(self, single_data, lr):
        sum_gradient_w = [ np.zeros(w.shape) for w in self.weights ]
        sum_gradient_b = [ np.zeros(b.shape) for b in self.biases ]
        
        # cumulate gradient of each single data
        for x, y in single_data:
            gradient_w, gradient_b = self.backward_propagation(x, y)
            sum_gradient_w = [ sw + w for sw, w in zip(sum_gradient_w, gradient_w)]
            sum_gradient_b = [ sb + b for sb, b in zip(sum_gradient_b, gradient_b)]
        
        # update weights & biases with (mean of sum of gradient * learning rate)
        self.weights = [ w - lr/len(single_data) * sw for w, sw in zip(self.weights, sum_gradient_w) ]
        self.biases = [ b - lr/len(single_data) * sb for b, sb in zip(self.biases, sum_gradient_b) ]
    
    def feed_forward(self, x):
        for w, b in zip(self.weights, self.biases):
            x = np.dot(w, x) + b
            x = sigmoid(x)
        return x
    
    def backward_propagation(self, x, y):
        # store gradient of w, b
        gradient_w = [ np.zeros(w.shape) for w in self.weights ]
        gradient_b = [ np.zeros(b.shape) for b in self.biases ]
        
        # forward
        activation = x
        preds = [] # store vectors which is input of activation function
        activations = [x] # store vectors which is output of activation function
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            preds.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # latent_feature
        self.latent_count+=1 
        self.batches_latent.append(preds[-2])
        # 每12000筆存一次(train num)
        if self.latent_count ==12000:
#             print("self.latent_len: ", len(self.latent))
#             print("self.batches_latent_len: ",len(self.batches_latent))
            self.latent.append(self.batches_latent)
            self.batches_latent = []
            self.latent_count = 0
            
                
                
        # backward 
        # we calc last layer separately, because loss function is diff with activation funcion
        delta = cross_entropy_derivative(activations[-1], y)
        gradient_b[-1] = delta * 1
        gradient_w[-1] = np.dot(delta, activations[-2].T)
        for layer in range(2, self.layer_num):
            z = preds[-layer]
            delta = np.dot(self.weights[-layer + 1].T, delta) * sigmoid_derivate(z)
            gradient_w[-layer] = np.dot(delta, activations[-layer - 1].T)
            gradient_b[-layer] = delta
        return gradient_w, gradient_b  
    
    # 計算 cross entropy的 loss
    def calc_loss(self, data):
        loss = 0
        for x, y in data:
            output = self.feed_forward(x)
            loss += cross_entropy(output, y)/ len(data)
        return loss
    
    def count_error(self, data):
        # count error number
        compare_list = [ (np.argmax(self.feed_forward(x)), np.argmax(y)) for x, y in data ]
        error_count = sum( int(y1 != y2) for y1, y2 in compare_list)
        return error_count
    
    
    def get_confusion_matrix(self, data):
        predict_ans = []
        real_ans = []
        for x, y in data:
            predict_ans.append(np.argmax(self.feed_forward(x)))
            real_ans.append(np.argmax(y))
        con_matrix = confusion_matrix(real_ans,predict_ans)
        return con_matrix


# In[ ]:


# initial variable = random
class model_DNN_random():
    def __init__(self, layers):
        #設定有幾層神經網路
        self.layer_num = len(layers)
        self.neurons = layers
        # create weights and bias
        self.weights = [ np.random.randn(j, i) for i, j in zip(layers[:-1], layers[1:]) ]
        self.biases = [ np.random.randn(i, 1) for i in layers[1:] ]
         
        self.training_loss = []
        self.training_error_rate = []
        self.testing_error_rate = []
        self.batches_latent = []
        self.latent_count = 0
        self.latent = []
        
        #np.random.seed(42)
        #weights = np.random.rand(3,1)
        #bias = np.random.rand(1)
        #lr = 0.05 #learning rate
        
        
    def SGD(self, training_data, testing_data, epochs, batch_size, lr):
        #總資料量
        total_data_num = len(training_data)
        self.training_loss = []
        self.training_error_rate = []
        self.testing_error_rate = []
        
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            # 把資料切割成 mini_batch
            mini_batch = [ training_data[i : i + batch_size] for i in range(0, total_data_num, batch_size) ] 
            for single_data in mini_batch:
                self.update_mini_batch(single_data, lr)
                
            #每50個epoch顯示一次
            if (epoch % 50 == 0):
                # record info
                self.training_loss.append(self.calc_loss(training_data))
                self.training_error_rate.append(self.count_error(training_data) / len(training_data))
                self.testing_error_rate.append(self.count_error(testing_data) / len(testing_data))
                print('===================================')
                print("Epoch:" ,epoch) 
                print('    training error rate: %d <-- %d ,rate=(%f)' % (self.count_error(training_data), len(training_data), self.count_error(training_data) / len(training_data)))
                print('    testing error rate: %d <-- %d ,rate=(%f)' % (self.count_error(testing_data), len(testing_data), self.count_error(testing_data) / len(testing_data)))
                print('    training loss: %f' % self.calc_loss(training_data))
                
    def update_mini_batch(self, single_data, lr):
        sum_gradient_w = [ np.zeros(w.shape) for w in self.weights ]
        sum_gradient_b = [ np.zeros(b.shape) for b in self.biases ]
        
        # cumulate gradient of each single data
        for x, y in single_data:
            gradient_w, gradient_b = self.backward_propagation(x, y)
            sum_gradient_w = [ sw + w for sw, w in zip(sum_gradient_w, gradient_w)]
            sum_gradient_b = [ sb + b for sb, b in zip(sum_gradient_b, gradient_b)]
        
        # update weights & biases with (mean of sum of gradient * learning rate)
        self.weights = [ w - lr/len(single_data) * sw for w, sw in zip(self.weights, sum_gradient_w) ]
        self.biases = [ b - lr/len(single_data) * sb for b, sb in zip(self.biases, sum_gradient_b) ]
    
    def feed_forward(self, x):
        for w, b in zip(self.weights, self.biases):
            x = np.dot(w, x) + b
            x = sigmoid(x)
        return x
    
    def backward_propagation(self, x, y):
        # store gradient of w, b
        gradient_w = [ np.zeros(w.shape) for w in self.weights ]
        gradient_b = [ np.zeros(b.shape) for b in self.biases ]
        
        # forward
        activation = x
        preds = [] # store vectors which is input of activation function
        activations = [x] # store vectors which is output of activation function
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            preds.append(z)
            activation = sigmoid(z)
            activations.append(activation)
                
                
        # latent_feature
        self.latent_count+=1 
        self.batches_latent.append(preds[-2])
        # 每12000筆存一次
        if self.latent_count ==12000:
#             print("self.latent_len: ", len(self.latent))
#             print("self.batches_latent_len: ", len(self.batches_latent))
            self.latent.append(self.batches_latent)
            self.latent_count = 0
            self.batches_latent = []
            
            
        # backward 
        # we calc last layer separately, because loss function is diff with activation funcion
        delta = cross_entropy_derivative(activations[-1], y)
        gradient_b[-1] = delta * 1
        gradient_w[-1] = np.dot(delta, activations[-2].T)
        for layer in range(2, self.layer_num):
            z = preds[-layer]
            delta = np.dot(self.weights[-layer + 1].T, delta) * sigmoid_derivate(z)
            gradient_w[-layer] = np.dot(delta, activations[-layer - 1].T)
            gradient_b[-layer] = delta
        return gradient_w, gradient_b  
    
    # 計算 cross entropy的 loss
    def calc_loss(self, data):
        loss = 0
        for x, y in data:
            output = self.feed_forward(x)
            loss += cross_entropy(output, y)/ len(data)
        return loss
    
    def count_error(self, data):
        # count error number
        compare_list = [ (np.argmax(self.feed_forward(x)), np.argmax(y)) for x, y in data ]
        error_count = sum( int(y1 != y2) for y1, y2 in compare_list)
        return error_count
    
    def get_confusion_matrix(self, data):
        predict_ans = []
        real_ans = []
        for x, y in data:
            predict_ans.append(np.argmax(self.feed_forward(x)))
            real_ans.append(np.argmax(y))
        con_matrix = confusion_matrix(real_ans,predict_ans)
        return con_matrix


# In[ ]:


module1 = model_DNN_zero([784, 32, 32, 2, 10])
# SGD(self, training_data, testing_data, epochs, batch_size, lr):
module1.SGD(training_data, testing_data, 1000, 100, 0.3)
con_matrix_zero = module1.get_confusion_matrix(testing_data)


# In[ ]:


import matplotlib.patches as mpatches
# draw latent feature
colors = ['red', 'green', 'lightgreen', 'gray', 'cyan','blue','yellow','purple','orange','pink']

latent_feature_20 = np.array(module1.latent[20])
latent_feature_20 = np.reshape(latent_feature_20, (2,12000))
latent_feature_80 = np.array(module1.latent[80])
latent_feature_80 = np.reshape(latent_feature_80, (2,12000))
true_label = np.array([int(y) for y in train_data['label']])

set_color = []
for i in true_label:
    set_color.append(colors[i])

#set color
c0 = mpatches.Patch(color ='red', label ='0')
c1 = mpatches.Patch(color ='green', label ='1')
c2 = mpatches.Patch(color ='lightgreen', label ='2')
c3 = mpatches.Patch(color ='gray', label ='3')
c4 = mpatches.Patch(color ='cyan', label ='4')
c5 = mpatches.Patch(color ='blue', label ='5')
c6 = mpatches.Patch(color ='yellow', label ='6')
c7 = mpatches.Patch(color ='purple', label ='7')
c8 = mpatches.Patch(color ='orange', label ='8')
c9 = mpatches.Patch(color ='pink', label ='9')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
scatter= ax.scatter(latent_feature_20[0], latent_feature_20[1], c = set_color)
cluster1 = ax.legend(handles = [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9], labels=['0','1','2','3','4','5','6','7','8','9'],  loc='best')
ax.add_artist(cluster1)
ax.set_title('20 epochs')
plt.show()


# In[ ]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
scatter= ax.scatter(latent_feature_80[0], latent_feature_80[1], c = set_color)
cluster1 = ax.legend(handles = [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9], labels=['0','1','2','3','4','5','6','7','8','9'],  loc='best')
ax.add_artist(cluster1)
ax.set_title('80 epochs')
plt.show()


# In[ ]:


latent_feature_900 = np.array(module1.latent[900])
latent_feature_900 = np.reshape(latent_feature_900, (2,12000))
true_label = np.array([int(y) for y in train_data['label']])


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
scatter= ax.scatter(latent_feature_900[0], latent_feature_900[1], c = set_color)
cluster1 = ax.legend(handles = [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9], labels=['0','1','2','3','4','5','6','7','8','9'],  loc='best')
ax.add_artist(cluster1)
ax.set_title('900 epochs')
plt.show()


# In[ ]:


new_x_axis = np.arange(0,1000, 50)
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(18, 4)
ax[0].plot(new_x_axis, module1.training_loss)
ax[0].set_title('training loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Average cross entropy')


ax[1].plot(new_x_axis, module1.training_error_rate)
ax[1].set_title('training error rate')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Error rate')

ax[2].plot(new_x_axis, module1.testing_error_rate)
ax[2].set_title('testing error rate')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Error rate')


# In[ ]:


print(con_matrix_zero)


# In[ ]:


module2 = model_DNN_random([784, 32, 32, 2, 10])
module2.SGD(training_data, testing_data, 1000, 100, 0.3)
con_matrix_random = module2.get_confusion_matrix(testing_data)


# In[ ]:


new_x_axis = np.arange(0,1000, 50)
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(18, 4)
ax[0].plot(new_x_axis, module2.training_loss)
ax[0].set_title('training loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Average cross entropy')


ax[1].plot(new_x_axis, module2.training_error_rate)
ax[1].set_title('training error rate')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Error rate')

ax[2].plot(new_x_axis, module2.testing_error_rate)
ax[2].set_title('testing error rate')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Error rate')


# In[ ]:


print(con_matrix_random)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





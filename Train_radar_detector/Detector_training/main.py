
import numpy as np
import matplotlib.pyplot as plt
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from sklearn.utils import shuffle

type = ['pon1', 'q3n1']  #Radar types
snr = ['snr10','snr20','snr30','snr40','snr50', 'snr100']  #Different SNRs
new = 0

''' This loop is for loading and concatenating the training and test data which are spectrograms that 
have LTE signal only in them (Class 0) or that have both LTE and Rdar signal in them (Class 1)'''
for radartype in type:
    for i in range(len(snr)):
        if new != 0:
            lte = np.load('numpy_data/' + radartype + '/lte_' + snr[i] + '_' + radartype + '.npy')
            lte_test = np.load('numpy_data/' + radartype + '/lte_test' + snr[i] + '_' + radartype + '.npy')
            lte_radar = np.load('numpy_data/' + radartype + '/lte_radar_' + snr[i] + '_' + radartype + '.npy')
            lte_radar_test = np.load('numpy_data/' + radartype + '/lte_radar_test' + snr[i] + '_' + radartype + '.npy')
            data = np.concatenate([lte_radar, lte])
            data_label = np.concatenate([np.ones([len(lte_radar), 1]), np.zeros([len(lte), 1])])
            data_test = np.concatenate([lte_radar_test, lte_test])
            data_label_test = np.concatenate([np.ones([len(lte_radar_test), 1]), np.zeros([len(lte_test), 1])])
            training_data = np.concatenate([data, training_data])
            training_label = np.concatenate([data_label, training_label])
            testing_data = np.concatenate([data_test, testing_data])
            testing_label = np.concatenate([data_label_test, testing_label])
        else:
            lte = np.load('numpy_data/' + radartype + '/lte_' + snr[i] + '_' + radartype + '.npy')
            lte_test = np.load('numpy_data/' + radartype + '/lte_test' + snr[i] + '_' + radartype + '.npy')
            lte_radar = np.load('numpy_data/' + radartype + '/lte_radar_' + snr[i] + '_' + radartype + '.npy')
            lte_radar_test = np.load('numpy_data/' + radartype + '/lte_radar_test' + snr[i] + '_' + radartype + '.npy')
            training_data = np.concatenate([lte_radar, lte])
            training_label = np.concatenate([np.ones([len(lte_radar), 1]), np.zeros([len(lte), 1])])
            testing_data = np.concatenate([lte_radar_test, lte_test])
            testing_label = np.concatenate([np.ones([len(lte_radar_test), 1]), np.zeros([len(lte_test), 1])])
            new += 1


'''shuffle the data vigorously'''
training_data, training_label = shuffle(training_data, training_label)
training_data, training_label = shuffle(training_data, training_label)
training_data, training_label = shuffle(training_data, training_label)
training_data, training_label = shuffle(training_data, training_label)
training_data, training_label = shuffle(training_data, training_label)
testing_data, testing_label = shuffle(testing_data, testing_label)
testing_data, testing_label = shuffle(testing_data, testing_label)
testing_data, testing_label = shuffle(testing_data, testing_label)
testing_data, testing_label = shuffle(testing_data, testing_label)
testing_data, testing_label = shuffle(testing_data, testing_label)


#Normalize training data
training_data = training_data / 255.0
training_data = tf.convert_to_tensor(training_data, dtype=tf.float32)

#Normalize testing data
testing_data = testing_data / 255.0
testing_data = tf.convert_to_tensor(testing_data, dtype=tf.float32)

''' Function to create the model architecture'''
def model_initializer(lammda, full_lammda):
    regulari = tf.keras.regularizers.L2(lammda)
    regulari2 = tf.keras.regularizers.L2(full_lammda)
    initials = tf.keras.initializers.glorot_normal
    deepmodel = tf.keras.Sequential()
    deepmodel.add(tf.keras.layers.Conv2D(16,kernel_size=5,strides=1,activation='relu',input_shape=(128,128,1),
                                         kernel_regularizer=regulari, bias_regularizer=regulari, kernel_initializer=initials))
    deepmodel.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    deepmodel.add(tf.keras.layers.Conv2D(32, kernel_size=5, strides=1, activation='relu', kernel_regularizer=regulari,
                                         kernel_initializer=initials, bias_regularizer=regulari))
    deepmodel.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
    deepmodel.add(tf.keras.layers.Conv2D(64, kernel_size=5, strides=1, activation='relu', kernel_regularizer=regulari,
                                         kernel_initializer=initials, bias_regularizer=regulari))
    deepmodel.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
    deepmodel.add(tf.keras.layers.Conv2D(128, kernel_size=5, strides=1, activation='relu', kernel_regularizer=regulari,
                                         kernel_initializer=initials, bias_regularizer=regulari))
    deepmodel.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
    deepmodel.add(tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, activation='relu', kernel_regularizer=regulari,
                                         kernel_initializer=initials, bias_regularizer=regulari))
    deepmodel.add(tf.keras.layers.Flatten())
    deepmodel.add(tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=regulari2,
                                        bias_regularizer=regulari2, kernel_initializer=initials))
    deepmodel.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regulari,
                                        bias_regularizer=regulari, kernel_initializer=initials))
    deepmodel.summary()
    return deepmodel

''' Function to train the model'''
def model_trainer(rate, epoch_no, batchsize, inputdata, inputlabel, testdata, testlabel, lammda, full_lammda):
    deepmodel = model_initializer(lammda, full_lammda)
    #Plot model Architecture
    tf.keras.utils.plot_model(deepmodel, to_file='model_plot.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)
    deepmodel.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=rate),metrics=['accuracy'])
    learningrate = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = deepmodel.fit(inputdata,inputlabel,validation_data=(testdata,testlabel),epochs=epoch_no,batch_size=batchsize, callbacks=[learningrate])
    trainloss, trainaccuracy = history.history['loss'], history.history['accuracy']
    testloss, testaccuracy = history.history['val_loss'], history.history['val_accuracy']
    testestimate = deepmodel.predict(testdata)
    deepmodel.save("Trained_Radar_Detector")
    return [trainloss,trainaccuracy,testloss,testaccuracy,testestimate]

'''Function to calculate average error'''
def classerror(labels, estimated):
    #classes = []
    for i in range(len(estimated)):
        if estimated[i] >= 0.5:
            estimated[i] = 1
        else:
            estimated[i] = 0
    #classes = np.array(classes)
    averageerror = estimated - labels
    averageerror = tf.math.count_nonzero(averageerror)/ len(labels)
    return averageerror

''' Function for Adaptive Learning Rate'''
def scheduler(epoch, lr):
    #if epoch < 30:
    if epoch < 60:
        return lr
    #elif 30 <= epoch and epoch < 60:
    elif 600 <= epoch and epoch < 800:
        lr = lr * 0.98
        return lr
    #elif 60 <= epoch and epoch < 97:
    elif 800 <= epoch and epoch < 900:
        lr = lr * 0.9
        return lr
    else:
        return lr


rate = 0.01  #Learning rate
epoch_no = 100  #No. of training epochs
batchsize = 64  #Batchsize
lammda = 0.00005  #Regularization parameter
full_lammda = 0.0001 #Regularization parameter

#Train Model
results = model_trainer(rate, epoch_no, batchsize, training_data, training_label, testing_data, testing_label,
                        lammda, full_lammda)
testaverageerror = classerror(testing_label, results[-1])
print("average accuracy is: ", 1 - testaverageerror)

#Obtain arrays for constructing Plots
averageerrortrain = 1 - np.array(results[1])
averageerrortest = 1 - np.array(results[3])
lossytrain = results[0]
lossytest = results[2]


#averagetrain and testerror Plot
learningcurve = plt.figure(21)
bx2 = plt.gca()
bx2.plot(averageerrortrain, color='green', label='Average train error')
bx2.plot(averageerrortest, color='red', label='Average test error')
plt.legend()
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
bx2.set_xlabel('epochs')
bx2.set_ylabel('Average Error')
plt.title("Plot of Training and test average errors against epochs")

#Accuracy Plot
learningcurve3 = plt.figure(22)
bx212 = plt.gca()
bx212.plot(1 - averageerrortrain, color='green', label='training average accuracy')
bx212.plot(1 - averageerrortest, color='red', label='test average accuracy')
plt.legend()
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
bx212.set_xlabel('epochs')
bx212.set_ylabel('average accuracy')
plt.title("Plot of Training and test accuracy against epochs")

#LossPlots
learningcurve1 = plt.figure(23)
bx21 = plt.gca()
bx21.plot(lossytrain, color='green', label='training NLL loss')
bx21.plot(lossytest, color='red', label='test NLL loss')
plt.legend()
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
bx21.set_xlabel('epochs')
bx21.set_ylabel('Crossentropy loss')
plt.title("Plot of Training and test Crossentropy loss against epochs")

plt.show()

print(" ")


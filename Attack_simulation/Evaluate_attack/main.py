import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import randint


def normalize_minus1_1(datar):
    # scale all pixels from [-1,1] to [0,1]
    datar = (datar + 1) / 2.0
    return datar

def normalize_0_255(data):
    # scale all pixels from [0,255] to [0,1]
    data = data / 255
    return data

type = ['pon1', 'q3n1']
snr = ['snr10', 'snr20', 'snr30', 'snr40', 'snr50', 'snr100']
imagesize = [128,128]

#Load radar detector
detectmodel = tf.keras.models.load_model('Trained_Radar_Detector')
original_estimate_array = []
perturbed_estimate_array = []
original_accuracy_array = []
perturbed_accuracy_array = []


''' Load the adversarial and clean samples'''
for radartype in type:
    for i in range(len(snr)):
        original_radar = np.load('numpy_data/' + radartype + '/Clean_sample_' + snr[i] + '_' + radartype + '.npy')
        perturbed_radar = np.load('numpy_data/' + radartype + '/Adversarial_sample_' + snr[i] + '_' + radartype + '.npy')

        #normalize the data
        original_radar = normalize_0_255(original_radar)
        perturbed_radar = normalize_0_255(perturbed_radar)

        #Get the model's confidence on the predictions on the adversarial(perturbed) sample and clean sample
        original_estimate = detectmodel.predict(original_radar)
        original_estimate_array.append(original_estimate)
        perturbed_estimate = detectmodel.predict(perturbed_radar)
        perturbed_estimate_array.append(perturbed_estimate)

        #Get the Model's loss and accuracy
        loss, acc1 = detectmodel.evaluate(original_radar, np.ones([len(original_radar), 1]), verbose=2)
        original_accuracy_array.append(acc1)
        print('Restored model, accuracy of original: {:5.2f}%'.format(100 * acc1))
        loss2, acc2 = detectmodel.evaluate(perturbed_radar, np.ones([len(perturbed_radar), 1]), verbose=2)
        perturbed_accuracy_array.append(acc2)
        print('Restored model, accuracy of perturbed: {:5.2f}%'.format(100 * acc2))

#Create plots for confidence difference metric
for radartype in range(len(type)):
    avg_confidence = []
    for i in range(len(snr)*radartype, len(snr)*radartype+len(snr)):
        confidence = ((np.array(original_estimate_array[i])-np.array(perturbed_estimate_array[i]))/np.array(original_estimate_array[i])) * 100
        avg_confidence.append(np.mean(confidence))

    xaxis = [10, 20, 30, 40, 50, 100]
    xaxis_label = ['10', '20', '30', '40', '50', '100']
    plt.plot(xaxis, avg_confidence, color='black', marker = 'D')
    plt.xticks(xaxis, xaxis_label)
    plt.xlabel("Signal-to-noise ratio (dB)")
    plt.ylabel("Confidence Difference(%)")
    plt.savefig('results/Confidence_diff' + type[radartype] + '.png', dpi=300)
    plt.show()
plt.close()

#Create plot for Accuracies
for radartype in range(len(type)):
    original_avg_accuracy = []
    perturbed_avg_accuracy = []
    for i in range(len(snr)*radartype, len(snr)*radartype+len(snr)):
        original_avg_accuracy.append(np.mean(original_accuracy_array[i]))
        perturbed_avg_accuracy.append(np.mean(perturbed_accuracy_array[i]))

    xaxis = [10, 20, 30, 40, 50, 100]
    xaxis_label = ['10', '20', '30', '40', '50', '100']
    plt.plot(xaxis, original_avg_accuracy,  color='green', label = 'Un-Perturbed Signal', marker='D')
    plt.plot(xaxis, perturbed_avg_accuracy, color='red', label = 'Perturbed Signal', marker='D')
    plt.xticks(xaxis, xaxis_label)
    plt.xlabel("Signal-to-noise ratio (dB)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('results/accuracy_' + type[radartype] + '.png', dpi=300)
    plt.show()
plt.close()

print (" ")




















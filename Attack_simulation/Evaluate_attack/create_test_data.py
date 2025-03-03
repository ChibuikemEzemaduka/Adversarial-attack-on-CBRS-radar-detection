import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def load_images_from_directory_clean(directory, target_size,radar_type):
    image_list = []
    for filename in os.listdir(directory):
        if filename.startswith('Clean') and filename.endswith(radar_type):
            img = Image.open(os.path.join(directory, filename))
            img = img.resize(target_size, Image.ANTIALIAS)
            image_array = np.array(img)
            image_list.append(image_array)
    return np.array(image_list)
def load_images_from_directory_perturbed(directory, target_size, radar_type):
    image_list = []
    for filename in os.listdir(directory):
        if filename.startswith('Adversarial') and filename.endswith(radar_type):
            img = Image.open(os.path.join(directory, filename))
            img = img.resize(target_size, Image.ANTIALIAS)
            image_array = np.array(img)
            image_list.append(image_array)
    return np.array(image_list)

''' Code to convert MATLAB generated Adversarial and Clean Spectrograms into numpy arrays '''
# Replace 'your_image_directory' with the path to the directory containing your images
size = (128,128)
type = ['pon1','q3n1']
snr_list = ['snr10','snr20','snr30','snr40','snr50','snr100']
for radar_typ in type:
    for snr in snr_list:
        image_dataset_clean = load_images_from_directory_clean('raw_data/' + snr, size, radar_typ + '.png')
        np.save('numpy_data/' + radar_typ + '/Clean_sample_' + snr + '_' + radar_typ + '.npy', image_dataset_clean)
        print(image_dataset_clean.shape)

        image_dataset_perturbed = load_images_from_directory_perturbed('raw_data/' + snr, size,
                                                                     radar_typ + '.png')
        np.save('numpy_data/' + radar_typ + '/Adversarial_sample_' + snr + '_' + radar_typ + '.npy',
                image_dataset_perturbed)
        print(image_dataset_perturbed.shape)


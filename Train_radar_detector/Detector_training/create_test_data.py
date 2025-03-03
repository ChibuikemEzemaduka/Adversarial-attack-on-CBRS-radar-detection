import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def load_images_from_directory_lte(directory, target_size,radar_type):
    image_list = []
    for filename in os.listdir(directory):
        if filename.startswith('LTE_frame') and filename.endswith(radar_type):
            img = Image.open(os.path.join(directory, filename))
            img = img.resize(target_size, Image.ANTIALIAS)
            image_array = np.array(img)
            image_list.append(image_array)
    return np.array(image_list)
def load_images_from_directory_lteradar(directory, target_size, radar_type):
    image_list = []
    for filename in os.listdir(directory):
        if filename.startswith('LTE_Radar_frame') and filename.endswith(radar_type):
            img = Image.open(os.path.join(directory, filename))
            img = img.resize(target_size, Image.ANTIALIAS)
            image_array = np.array(img)
            image_list.append(image_array)
    return np.array(image_list)



''' Code for loading and converting the test data (MATLAB generated spectrograms) to numpy arrays '''
# Replace 'your_image_directory' with the path to the directory containing your images
size = (128,128)
type = ['pon1','q3n1']
snr_list = ['snr10','snr20','snr30','snr40','snr50','snr100']
for radar_typ in type:
    for snr in snr_list:
        image_dataset_lte = load_images_from_directory_lte('rawdata_test/' + snr, size, radar_typ + '.png')
        np.save('numpy_data/' + radar_typ + '/lte_test' + snr + '_' + radar_typ + '.npy', image_dataset_lte)
        print(image_dataset_lte.shape)

        image_dataset_lteradar = load_images_from_directory_lteradar('rawdata_test/' + snr, size,
                                                                     radar_typ + '.png')
        np.save('numpy_data/' + radar_typ + '/lte_radar_test' + snr + '_' + radar_typ + '.npy',
                image_dataset_lteradar)
        print(image_dataset_lteradar.shape)

import cv2
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import torch
import h5py
import os
from PIL import Image
from tqdm import tqdm


input_path = 'resized'
count = 0
error_count = 0
dataset_count = 0
print('start loading')
for filename in tqdm(os.listdir(input_path)):
    
    

    f = os.path.join(input_path, filename)
    
    # if os.path.isfile(f):
    #     print(f)
    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).share_memory_().view(-1,128,128,3)

    if(count == 0):
        dataset = image
    else:
        dataset = torch.cat((dataset, image))


    if count == 3000:
        dataset_count += 1 
        torch.save(dataset, f'dataset_{dataset_count}.pt')
        print(dataset.shape)
        print('save one dataset')
        count = 0
        continue
    
    count += 1



# print(dataset.shape)
# print('finished loading')

# dataset = np.array(dataset)
# print(dataset.shape)

# dataset = torch.from_numpy(dataset)
# torch.save(dataset, 'dataset.pt')
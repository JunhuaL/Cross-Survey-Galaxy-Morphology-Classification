import torch
from tqdm import tqdm
import os


input_path = 'datasets'

# dataset = torch.load('dataset_final.pt').share_memory_()
# print(dataset.shape)

count = 0

for filename in tqdm(os.listdir(input_path)):
    

    f = os.path.join(input_path, filename)
    
    # if os.path.isfile(f):
    #     print(f)
    
    if 'dataset_' in filename:



        currDataset = torch.load(f).share_memory_().view(-1,128,128,3)

        if count == 0:
            dataset = currDataset
            count += 1
        else:
            
            dataset = torch.cat((dataset, currDataset), 0)
        print(dataset.shape)


torch.save(dataset, f'dataset_final.pt')

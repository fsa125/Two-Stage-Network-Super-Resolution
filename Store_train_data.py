from utils import *
from tqdm import tqdm
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StoreTrainData():
    def __init__(self, train_dataloader,directory_name):
        stacked_all = []
        for batch_idx, imgs in tqdm(enumerate(train_dataloader)):
            #print(batch_idx)
            labels = []
   
            low_res = Variable(imgs["lr"])
            high_res = Variable(imgs["hr"])
  
            low_res = low_res.to(torch.float32)
            high_res = high_res.to(torch.float32)
            stacked_dataset = TensorDataset(low_res, high_res)
            stacked_all.append(stacked_dataset)
    
            #stacked_data = torch.stack((low_res, high_res))
            #low_res_all = torch.stack(low_res)
            #low_high_all = torch.stack(high_res)
    
        torch.save(stacked_all, directory_name)
        #torch.save(high_res_all, 'high_res_all.pt')




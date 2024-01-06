from utils import *
from tqdm import tqdm
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Train():
    def __init__(self, stacked_dataset, criterion, optimizer, num_epochs, model,model2,model3):
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.model = model
        self.model2 = model2
        self.model3 = model3
        
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) #step_size = 100
        random_integer = random.randint(0, 49)
        batch_idx = 0
        # Training the model
        num_epochs = self.num_epochs
        best_validation_loss = float('inf')
        model = self.model
        model2 = self.model2
        model3 = self.model3

        for epoch in tqdm(range(num_epochs)):
    
            model.train()
            model2.train()
            model3.train()
            random_integer = random.randint(0, 49)

    
#     for batch_idx, imgs in enumerate(train_dataloader):
#         #print(high_res[0].shape)
#         #print(batch_idx)
#         low_res = Variable(imgs["lr"])
#         high_res = Variable(imgs["hr"])
#         #print(low_res.shape)
#         #print(high_res.shape)
#         #high_res = high_res[0].to(device)
#         #low_res = low_res[0].to(device)
#         #low_res  = torch.stack([downscale_transform(img) for img in high_res])
#         low_res = low_res.to(torch.float32)
#         high_res = high_res.to(torch.float32)
#         #print(low_res.max())
#         #print(high_res.max())


        # Forward pass
            low_res = stacked_dataset[random_integer].tensors[0]
            high_res = stacked_dataset[random_integer].tensors[1]
            outputs1,outputs2 = model(low_res)
            outputs_LR = model2(high_res)
            #Sobel loss
            sobel_HR = model3(high_res)
            sobel_LR = model3(outputs1)
            #print(outputs.shape)
            loss = 5*criterion(outputs1, high_res) + criterion(outputs2, outputs_LR) + criterion(sobel_HR, sobel_LR)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if loss.item() < best_validation_loss:
                best_validation_loss = loss.item()
                torch.save(model.state_dict(), 'best_model_norm_BN_sobel_small_1.pth')  # Save the model state dictionary to a file
      

            if (batch_idx+1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                #print(torch.cuda.max_memory_allocated()/(1024**3))
                # Retrieve GPU memory statistics
                memory_stats = torch.cuda.memory_stats()

                # Calculate available GPU memory
                total_memory = torch.cuda.get_device_properties(0).total_memory
                available_memory = total_memory - memory_stats["allocated_bytes.all.current"]

            # Print the result
                print(f"Available GPU memory: {available_memory / 1024**3:.2f} GB")
            batch_idx +=1
            
            scheduler.step()
        
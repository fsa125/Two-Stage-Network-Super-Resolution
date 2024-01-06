from utils import *
from tqdm import tqdm
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rgb_to_ycbcr(rgb_image):
    """
    Convert RGB image to YCbCr color space using PyTorch.
    
    Args:
        rgb_image (Tensor): RGB image tensor with shape (batch_size, 3, height, width).
        
    Returns:
        Tensor: YCbCr image tensor with shape (batch_size, 3, height, width).
    """
    # RGB to YCbCr conversion matrix
    conversion_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ]).to(rgb_image.device).float()

    # Reshape RGB image for matrix multiplication
    rgb_image_reshaped = rgb_image.permute(0, 2, 3, 1)  # (batch_size, height, width, 3)
    
    # Convert RGB to YCbCr
    ycbcr_image = torch.matmul(rgb_image_reshaped, conversion_matrix.t())
    
    # Add offset to Cb and Cr channels
    ycbcr_image[..., 1:] += 128.0
    
    # Clamp the values to the valid range [0, 255]
    ycbcr_image = torch.clamp(ycbcr_image, 0, 255)
    
    return ycbcr_image.permute(0, 3, 1, 2)  # (batch_size, 3, height, width)

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

class Test():
    def __init__(self, test_dataloader, model):
        self.test_dataloader = test_dataloader
        self.model = model

        test_dataloader = self.test_dataloader
        model = self.model

        idx = 0
        model.eval()
        with torch.no_grad():
            total_psnr = 0
            for idx, imgs in enumerate(test_dataloader):
                #low_res, high_res = low_res[0].to(device), high_res[0].to(device)
                #print(low_res.shape)
                #print(high_res.shape)
                low_res = Variable(imgs["lr"].to(device))
                high_res = Variable(imgs["hr"].to(device))
                low_res = low_res.to(torch.float32)
                high_res = high_res.to(torch.float32)
        
        
        

                # Predict high-resolution image
                predicted_high_res,dummy = model(low_res)
        
                #Only for testing PSNR
        
                low_res_T = rgb_to_ycbcr(low_res)
                high_res_T = rgb_to_ycbcr(high_res)
                predicted_high_res_T = rgb_to_ycbcr(predicted_high_res)
        
        
        
        
                #print(predicted_high_res.shape)
                mse = torch.mean((predicted_high_res_T[:,0,:,:] - high_res_T[:,0,:,:]) ** 2)
                #mse = criterion(predicted_high_res, high_res)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                total_psnr += psnr.item()
        
   
        

                # Visualize the results (first 5 images)
                if idx < 5:
                 
                 
                    #to_pil = torchvision.transforms.ToPILImage()
                    #to_pil = torchvision.transforms.ToPILImage()
                    #predicted_img = to_pil(predicted_img)
                    #print(low_res_img.shape)
            
                    low_res_img = low_res.squeeze().cpu().numpy().transpose(1, 2, 0)
                    high_res_img = high_res.squeeze().cpu().numpy().transpose(1, 2, 0)
                    predicted_img = predicted_high_res.squeeze().cpu().numpy().transpose(1, 2, 0)
           
            
            
             #low_res_img = transforms.ToPILImage()(low_res_img)
            
            

                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(low_res_img)
                    #low_res_img.show()
                    plt.title('Low-Resolution Image')
                    plt.subplot(1, 3, 2)
                    plt.imshow(predicted_img)
                    #plt.imshow((rgb2gray(predicted_img)-rgb2gray(high_res_img)), cmap='gray')
                    #plt.title('Predicted Difference Image')
                    plt.title('Predicted High-Resolution Image')
                    plt.subplot(1, 3, 3)
                    plt.imshow(high_res_img)
                    plt.title('Original High-Resolution Image')
                    plt.savefig('RCAG_8_RCAB_15_dual_loss_500 {}.png'.format(idx))
                    plt.show()
            
            
            
                #print(low_res_img)
                #print('done')
                #print(predicted_img)
            
        average_psnr = total_psnr / len(test_dataloader)
        print(f'Average PSNR on Test Data: {average_psnr:.2f} dB')
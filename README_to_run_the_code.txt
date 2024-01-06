Edge Enhanced Three-Stage Deep Network for Image Super-Resolution

1) How to run the experiment: First download the dataset from the following link: https://www.kaggle.com/datasets/joe1995/div2k-dataset


Extract it and the folder should look like this: 

archive
|
|
|-----DIV2K_train_HR-----DIV2K_train_HR
|
|
|-----DIV2K_valid_HR-----DIV2K_valid_HR

Replace the dummy "archive" folder with the one you downloaded and extracted.

Open the "Tri Stage Super-Resolution.ipynb" and Run all the cell. There are comments to explain the workflow in the jupyter notebook.

2) Code References:
	a) In the "model.py" module, the channel attention layer and the residual channel attenion blocks were directly copied from  https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py. The "image_loader.py" module was also inspired from this repository.

	b) In the "Edge_Extractor_Net.py" module, the "SobelEdgeDetection" class was inspired from 
: https://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/
        c) The basic Skeleton of of Train_loader.py and Test_loader.py was taken from the book https://www.manning.com/books/deep-learning-with-pytorch. Moreover, some elements such as counting PSNR with YCbRr is inspired by https://stackoverflow.com/questions/44944455/psnr-values-differ-in-matlab-implementation-and-python
        d) The TSN, FirstStage function from "model.py", and "SobelConvolutionNet" function from "Edge_Extractor_Net.py" were completely written by me.

        e) The dual-loop training loop is slighly modified by me. At first, I loaded all the training data on my GPU in "Store_train_data.py" module. Then in the "Train_loader.py" module, I just called random numbers to get a new batch in each epoch. Therefore, in each epoch, we do not need to load the training data. This modification allowed me to run 4000 epochs in only 30 minutes.

        f) The number of RCABs used in each RCAG is 15. But in this code, I reduced it to 5 so that it meets the memory requirement of the machine. You can change it in the "model.py" module.

	g) I have included a pre-loaded training data named "stacked_dataset.pt"

	h) You have to set Epoch = 4000 and RCAB values in each RCAG equal to 15 to get the PSNR values similar to that mentioned in the paper.


3) Dataset download link is given in (1).


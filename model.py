from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        
        y = self.avg_pool(x)
        y = self.fc(y)
        
        return x*y

class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ResidualChannelAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.attention = ChannelAttention(in_channels, reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        attention = self.attention(out)
        #out = out * attention
        attention += x
        return  attention
    

class RCAG(nn.Module):
    def __init__(self,num_blocks = 5):
        super(RCAG, self).__init__()
        self.num_blocks = num_blocks
        self.rcag =  nn.Sequential(*nn.ModuleList([ResidualChannelAttentionBlock(in_channels = 64) for _ in range(self.num_blocks)]))   

    def forward(self, x):
       
        x = self.rcag(x)
     
       
        
        return x
class TSN(nn.Module):
    def __init__(self, num_blocks = 4):
        super(TSN, self).__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride = 1, padding=4)
        self.rcag4_1 = nn.Sequential(*nn.ModuleList([RCAG() for _ in range(self.num_blocks)]))
        self.rcag4_2 = nn.Sequential(*nn.ModuleList([RCAG() for _ in range(self.num_blocks)]))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1,stride = 1, padding=0)
        self.conv2D = nn.ConvTranspose2d( in_channels = 32, out_channels = 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, stride =1, padding=2)
        #self.SobelEdgeDetection = SobelEdgeDetection()
        

    def forward(self, x):
        x = self.conv1(x)
        temp = x    
        x = self.rcag4_1(x)
        y=x
        #y= self.SobelEdgeDetection(x)
        x = self.rcag4_2(x)
        x += temp
        x = self.conv2(x)
        x = self.conv2D(x)
        x = self.conv3(x)
        return x,y
    
class FirstStage(nn.Module):
    def __init__(self, num_blocks=8):
        super(FirstStage, self).__init__()
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride = 1, padding=4)
        self.rcag8 = nn.Sequential(*nn.ModuleList([RCAG() for _ in range(self.num_blocks)]))
        #self.conv2 = nn.Conv2d(64, 32, kernel_size=1,stride = 1, padding=0)
        #self.conv2D = nn.ConvTranspose2d( in_channels = 32, out_channels = 32, kernel_size=4, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(32, 3, kernel_size=5, stride =1, padding=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.SobelEdgeDetection = SobelEdgeDetection()

    def forward(self, x): 
        x = self.conv1(x)
        x = self.rcag8(x)
        x = self.max_pool(x)
        
        #x = self.SobelEdgeDetection(x)
        return x
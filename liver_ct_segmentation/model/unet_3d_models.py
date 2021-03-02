import torch
import torch.nn as nn
import torch.nn.functional as F

def create_model(n_channels, n_class, dropout_val=0.25):
    return UNet3D(n_channels, n_class, dropout_val=dropout_val)

def create_parallel_model(n_channels, n_class, dropout_val=0.25):
    return DataParallelPassthrough(UNet3D(n_channels, n_class, dropout_val=dropout_val))

class DataParallelPassthrough(torch.nn.DataParallel):
    """
    This class solves https://github.com/pytorch/pytorch/issues/16885
    Basically, to allow the access of a model wrapped under DataParallel one needs to always
    access the underlying attributes with .module (e.g. model.module.someattr)
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

###################################

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout_val=0.001): #dropout_val=0.01
        super(ConvBlock, self).__init__()

        self.dropout_value = dropout_val
        self.dropout_1 = nn.Dropout3d(self.dropout_value)
        self.dropout_2 = nn.Dropout3d(self.dropout_value)

        self.non_linearity = nn.ReLU(inplace=False)

        self.conv_1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.batch_norm_1 = nn.BatchNorm3d(out_ch)

        self.conv_2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.batch_norm_2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):

        x = self.dropout_1(x)
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.non_linearity(x)

        x = self.dropout_2(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.non_linearity(x)

        return x

class InputBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout_val=0.001):
        super(InputBlock, self).__init__()

        self.conv_block_1 = ConvBlock(in_ch, out_ch, dropout_val=dropout_val) #, dropout_val=0.2)

    def forward(self, x):

        x = self.conv_block_1(x)

        return x

class DownSamplingBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout_val=0.001):
        super(DownSamplingBlock, self).__init__()

        self.dropout_value = 0.001 #0.25, 0.01
        
        self.down = nn.Sequential(
            nn.Dropout3d(self.dropout_value),
            
            #nn.MaxPool3d(2, stride=2),
            nn.Conv3d(in_ch, in_ch, 2, stride=2),
            
            #nn.BatchNorm3d(in_ch),
            #nn.ReLU(inplace=True),

            ConvBlock(in_ch, out_ch, dropout_val=dropout_val) #, dropout_val=0.2)
        )

    def forward(self, x):
        
        x = self.down(x)
        return x

class UpSamplingBlock(nn.Module):

    def __init__(self, in_ch, cat_ch, out_ch, dropout_val=0.001):
        super(UpSamplingBlock, self).__init__()

        self.dropout_value = 0.001 #0.25, 0.01

        self.up = nn.Sequential(
            nn.Dropout3d(self.dropout_value),

            #nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2),
            nn.Upsample(scale_factor=2, mode='nearest'),

            #nn.BatchNorm3d(in_ch),
            #nn.ReLU(inplace=True)
        )

        self.conv = ConvBlock(in_ch + cat_ch, out_ch, dropout_val=dropout_val) #, dropout_val=0.2)

    def cat_operation(self, dc, syn):
        
        return torch.cat((dc, syn), dim=1)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = self.cat_operation(x1, x2)
        x = self.conv(x)
        return x

class OutputBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(OutputBlock, self).__init__()

        self.conv_1= nn.Conv3d(in_ch, out_ch, 1)
        #self.batch_norm_1 = nn.BatchNorm3d(out_ch)
        
    def forward(self, x):
        
        x = self.conv_1(x)
        #x = self.batch_norm_1(x)
        
        #softmax out?
        #x = F.softmax(x, dim=1)

        return x

###################################

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_val=0.001):
        super(UNet3D, self).__init__()

        ################################
        #cet_unet

        self.inc = InputBlock(n_channels, 32, dropout_val=dropout_val)
        
        self.down1 = DownSamplingBlock(32, 64, dropout_val=dropout_val)
        self.down2 = DownSamplingBlock(64, 128, dropout_val=dropout_val)

        self.mid = ConvBlock(128, 128, dropout_val=dropout_val)

        self.up1 = UpSamplingBlock(128, 64, 64, dropout_val=dropout_val)
        self.up2 = UpSamplingBlock(64, 32, 32, dropout_val=dropout_val)

        self.outc = OutputBlock(32, n_classes)


    def forward(self, x):

        #####################################

        x1 = self.inc(x)
        #print('x1:' + str(x1.shape))
        
        x2 = self.down1(x1)
        #print('x2:' + str(x2.shape))

        x3 = self.down2(x2)
        #print('x3:' + str(x3.shape))

        x3 = self.mid(x3)
        #print('xmid:' + str(x3.shape))
                
        x = self.up1(x3, x2) #self.mid2(x2))
        #print('up1:' + str(x.shape))

        x = self.up2(x, x1) #self.mid1(x1))
        #print('up2:' + str(x.shape))
                
        x = self.outc(x)
        #print('out:' + str(x.shape))
        
        return x


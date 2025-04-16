import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d


class UCDNet(nn.Module):
    "Model based on architecture proposed by Basavaraju - UCDNet"
    "Implementation by Ewa Kopec, Silesian University of Technology"

    def __init__(self, input_nbr, label_nbr):
        super(UCDNet, self).__init__()

        self.input_nbr = input_nbr

        #not sure if padding is used here in general solution???
        self.conv11 = nn.Conv2d(input_nbr, 16, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.convx16 = nn.Conv2d(16, 16, kernel_size=1, padding=0)

        self.conv21 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.convx32 = nn.Conv2d(32, 32, kernel_size=1, padding=0)

        self.conv31 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.convx64 = nn.Conv2d(64, 64, kernel_size=1, padding=0)

        self.conv41 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.convx128 = nn.Conv2d(128, 128, kernel_size=1, padding=0)

        self.conv256_64 = nn.Conv2d(256, 64, kernel_size=3, padding=1)

        
        #NSPP Block
        # pooling block
        # ------------------------------------------------------------------------------------------
        # strided separable convolution, stride=s, s [2, 4, 8, 16] filters C/4? 192/4=48
        # smaller size of s because patch size is 96 (not 512) [1, 2, 4, -] and filters C/3 192/3=64
        self.stridedConv2 = nn.Conv2d(192, 64, kernel_size=3, padding=1, stride=1)
        self.stridedConv4 = nn.Conv2d(192, 64, kernel_size=3, padding=1, stride=2)
        self.stridedConv8 = nn.Conv2d(192, 64, kernel_size=3, padding=1, stride=4)
        #self.stridedConv16 = nn.Conv2d(192, 48, kernel_size=3, padding=1, stride=16)

        # average pooling, kernel size = s
        self.avgPooling2 = nn.AvgPool2d(kernel_size=1)
        self.avgPooling4 = nn.AvgPool2d(kernel_size=2)
        self.avgPooling8 = nn.AvgPool2d(kernel_size=4)
        #self.avgPooling16 = nn.AvgPool2d(kernel_size=16)
        
        self.convPoolingBlock = nn.Conv2d(192, 64, kernel_size=1, padding=0)

        # Global pooling block
        self.reduceMean = nn.AdaptiveAvgPool2d(1)
        self.convGlobalPoolingBlock = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.upConvGlobalPoolingBlock = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=(1, 1)) #another stride??? stide(H, W)

        self.conv384_192 = nn.Conv2d(384, 192, kernel_size=1, padding=0)
        #------------------------------------------------------------------------------------------------
        
        #Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upconv1 = nn.Conv2d(192, 64, kernel_size=2, padding=1)

        #320-64->64->32
        self.conv320_64 = nn.Conv2d(320, 64, kernel_size=3, padding=1)
        self.conv64_64 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv64_32 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        self.batchNorm1 = nn.BatchNorm2d(32)

        self.upconv2 = nn.Conv2d(32, 32, kernel_size=2, padding=1)

        #160->32->16
        self.conv160_32 = nn.Conv2d(160, 32, kernel_size=3, padding=1)
        self.conv32_16 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.batchNorm2 = nn.BatchNorm2d(16)
        self.upconv3 = nn.Conv2d(16, 16, kernel_size=2, padding=1)

        #80->16
        self.conv80_16 = nn.Conv2d(80, 16, kernel_size=3, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(16)

        #16->2
        self.conv16_2 = nn.Conv2d(16, label_nbr, kernel_size=1, padding=0)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x1, x2):

        #Stage 1
        x11_A = F.relu(self.conv11(x1))
        x12_1_A = F.relu(self.conv12(x11_A))

        x11_B = F.relu(self.conv11(x2))
        x12_1_B = F.relu(self.conv12(x11_B))

        x16cc = F.relu(self.convx16(torch.abs(x11_A - x11_B))) 

        # or maybe here is missing dropout or sth like this? before f.relu
        #First C Chanelwsie concatenation 
        pad1 = ReplicationPad2d((0, x16cc.size(3) - x12_1_A.size(3), 0, x16cc.size(2) - x12_1_A.size(2)))
        x1cc_A = torch.cat((pad1(x12_1_A), x16cc), 1)
        x1cc_B = torch.cat((pad1(x12_1_B), x16cc), 1)

        # Max pooling 
        x1p_A = F.max_pool2d(x1cc_A, kernel_size=2, stride=2)
        x1p_B = F.max_pool2d(x1cc_B, kernel_size=2, stride=2)

        # Stage 2
        x21_A = F.relu(self.conv21(x1p_A))
        x22_1_A = F.relu(self.conv22(x21_A))
        x21_B = F.relu(self.conv21(x1p_B))
        x22_1_B = F.relu(self.conv22(x21_B))

        x32cc = F.relu(self.convx32(torch.abs(x21_A - x21_B)))
        pad2 = ReplicationPad2d((0, x32cc.size(3) - x22_1_A.size(3), 0, x32cc.size(2) - x22_1_A.size(2)))

        # Second C channel wise concatenation
        x2cc_A = torch.cat((pad2(x22_1_A), x32cc), 1)
        x2cc_B = torch.cat((pad2(x22_1_B), x32cc), 1)

        #Max pooling
        x2p_A = F.max_pool2d(x2cc_A, kernel_size=2, stride=2)
        x2p_B = F.max_pool2d(x2cc_B, kernel_size=2, stride=2)

        #Stage 3
        x31_A = F.relu(self.conv31(x2p_A))
        x32_1_A = F.relu(self.conv32(x31_A))
        x31_B = F.relu(self.conv31(x2p_B))
        x32_1_B = F.relu(self.conv32(x31_B))

        x64cc = F.relu(self.convx64(torch.abs(x31_A - x31_B)))
        pad3 = ReplicationPad2d((0, x64cc.size(3) - x32_1_A.size(3), 0, x64cc.size(2) - x32_1_A.size(2)))

        # 3rd C channel wise concatenation
        x3cc_A = torch.cat((pad3(x32_1_A), x64cc), 1)
        x3cc_B = torch.cat((pad3(x32_1_B), x64cc), 1)

        #Max pooling
        x3p_A = F.max_pool2d(x3cc_A, kernel_size=2, stride=2)
        x3p_B = F.max_pool2d(x3cc_B, kernel_size=2, stride=2)


        #Stage 4
        x41_A = F.relu(self.conv41(x3p_A))
        x42_1_A = F.relu(self.conv42(x41_A))
        x41_B = F.relu(self.conv41(x3p_B))
        x42_1_B = F.relu(self.conv42(x41_B))

        x128cc = F.relu(self.convx128(torch.abs(x41_A - x41_B)))
        pad4 = ReplicationPad2d((0, x128cc.size(3) - x42_1_A.size(3), 0, x128cc.size(2) - x42_1_A.size(2)))

        # 4th C channel wise concatenation
        x4cc_A = torch.cat((pad4(x42_1_A), x128cc), 1)
        x4cc_B = torch.cat((pad4(x42_1_B), x128cc), 1)

        x44_A = F.relu(self.conv256_64(x4cc_A))
        x44_B = F.relu(self.conv256_64(x4cc_B))


        xx_AB = torch.abs(x44_A - x44_B)

        y1cc = torch.cat((x44_A, x44_B, xx_AB), 1) 
        
        
        # NSPP block - Bottleneck
        # Pooling block 
        # ---------------------------------------------------------------
        nspp_conv1 = F.relu(self.stridedConv2(y1cc))
        nspp_conv2 = F.relu(self.stridedConv4(y1cc))
        nspp_conv3 = F.relu(self.stridedConv8(y1cc))
        #nspp_conv4 = F.relu(self.stridedConv16(y1cc))

        avgpooling1 = self.avgPooling2(y1cc)
        avgpooling2 = self.avgPooling4(y1cc)
        avgpooling3 = self.avgPooling8(y1cc)
        #avgpooling4 = self.convPoolingBlock(self.avgPooling16(y1cc))

        convpooling1 = self.convPoolingBlock(avgpooling1)
        convpooling2 = self.convPoolingBlock(avgpooling2)
        convpooling3 = self.convPoolingBlock(avgpooling3)  

        pad_nspp1 = ReplicationPad2d((0, nspp_conv1.size(3) - convpooling1.size(3), 0, nspp_conv1.size(2) - convpooling1.size(2)))
        pad_nspp2 = ReplicationPad2d((0, nspp_conv2.size(3) - convpooling2.size(3), 0, nspp_conv2.size(2) - convpooling2.size(2)))
        pad_nspp3 = ReplicationPad2d((0, nspp_conv3.size(3) - convpooling3.size(3), 0, nspp_conv3.size(2) - convpooling3.size(2)))

        fp1 = nspp_conv1 + pad_nspp1(convpooling1)
        fp2 = nspp_conv2 + pad_nspp2(convpooling2)
        fp3 = nspp_conv3 + pad_nspp3(convpooling3)
        #fp4 = nspp_conv4 + avgpooling4


        # Global pooling block
        fg_o1 = self.convGlobalPoolingBlock(self.reduceMean(fp1))
        fg_o2 = self.convGlobalPoolingBlock(self.reduceMean(fp2))
        fg_o3 = self.convGlobalPoolingBlock(self.reduceMean(fp3))
        #fg_o4 = self.convGlobalPoolingBlock(torch.mean(fp4))

        fp_o1 = F.relu(self.upConvGlobalPoolingBlock(fg_o1))
        fp_o2 = F.relu(self.upConvGlobalPoolingBlock(fg_o2))
        fp_o3 = F.relu(self.upConvGlobalPoolingBlock(fg_o3))
        #fp_o4 = F.relu(self.upConvGlobalPoolingBlock(fg_o4))

        pad = ReplicationPad2d((0, y1cc.size(3) - fp_o1.size(3), 0, y1cc.size(2) - fp_o1.size(2)))

        f_new_spp = torch.cat((pad(fp_o1), pad(fp_o2), pad(fp_o3), y1cc), dim=1)
                    
        nspp = self.conv384_192(f_new_spp)
        #---------------------------------------------------------------
        
        ###
        # Decoder
        # up convolutional layer
        # stage 1
        yupconv1 = F.relu(self.upconv1(self.upsample(nspp))) #output 64

        pad_decoder_1 = ReplicationPad2d((0, x3cc_A.size(3) - yupconv1.size(3), 0, x3cc_A.size(2) - yupconv1.size(2)))
        y2cc = torch.cat((x3cc_A, x3cc_B, pad_decoder_1(yupconv1)), 1)

        # 320->64->64->32
        yconv1 = F.relu(self.conv64_32(self.conv64_64(self.conv320_64(y2cc))))
        batchNorm1 = self.batchNorm1(yconv1)

        #stage 2
        yupconv2 = F.relu(self.upconv2(self.upsample(batchNorm1)))

        pad_decoder_2 = ReplicationPad2d((0, x2cc_A.size(3) - yupconv2.size(3), 0, x2cc_A.size(2) - yupconv2.size(2)))
        y3cc = torch.cat((pad_decoder_2(yupconv2), x2cc_A, x2cc_B), 1)

        #160->32->16
        yconv2 = F.relu(self.conv32_16(self.conv160_32(y3cc)))
        batchNorm2 = self.batchNorm2(yconv2)

        #stage 3
        yupconv3 = self.upconv3(self.upsample(batchNorm2))

        pad_decoder_3 = ReplicationPad2d((0, x1cc_A.size(3) - yupconv3.size(3), 0, x1cc_A.size(2) - yupconv3.size(2)))
        y4cc = torch.cat((pad_decoder_3(yupconv3), x1cc_A, x1cc_B), 1)

        yconv3 = F.relu(self.conv80_16(y4cc))
        batchNorm3 = self.batchNorm3(yconv3)

        #final stage 
        #16->2 
        
        final_y = self.softmax(self.conv16_2(batchNorm3))
        return final_y









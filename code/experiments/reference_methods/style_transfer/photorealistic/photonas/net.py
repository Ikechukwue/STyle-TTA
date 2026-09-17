"""
xAILab Bamberg
University of Bamberg

@description:
PhotoNAS network architecture (Encoder and Decoder).
Based on: https://github.com/pkuanjie/StyleNAS/blob/master/models/models_photorealistic_nas/VGG_with_decoder.py

Note: The official encoder uses a pretrained VGG normalised to conv5_1 loaded from a Lua .t7 file.
We use PyTorch VGG19 weights which should produce similar results.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PhotoNASEncoder(nn.Module):
    """
    VGG19-based encoder for PhotoNAS.
    Extracts features at relu1_1, relu2_1, relu3_1, relu4_1, and relu5_1.
    
    Official implementation uses 'vgg_normalised_conv5_1.t7' which extracts up to conv5_1.
    """
    def __init__(self, pretrained=True):
        super(PhotoNASEncoder, self).__init__()
        
        # Load VGG19 weights
        if pretrained:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        else:
            vgg = models.vgg19(weights=None).features
            
        # Layer definitions matching official implementation
        # Official implementation uses ReflectionPad + Conv(valid)
        
        # 1. Preprocessing / Color Transform (3 -> 3)
        # Official code loads this from .t7 file. We init as identity.
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        with torch.no_grad():
            self.conv1.weight.copy_(torch.eye(3).reshape(3, 3, 1, 1))
            self.conv1.bias.zero_()
            
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        
        # 2. conv1_1 (3 -> 64)
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self._init_layer(self.conv2, vgg[0])
        self.relu2 = nn.ReLU(inplace=True)
        
        # 3. conv1_2 (64 -> 64)
        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self._init_layer(self.conv3, vgg[2])
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # 4. conv2_1 (64 -> 128)
        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self._init_layer(self.conv4, vgg[5])
        self.relu4 = nn.ReLU(inplace=True)
        
        # 5. conv2_2 (128 -> 128)
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self._init_layer(self.conv5, vgg[7])
        self.relu5 = nn.ReLU(inplace=True)
        
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # 6. conv3_1 (128 -> 256)
        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self._init_layer(self.conv6, vgg[10])
        self.relu6 = nn.ReLU(inplace=True)
        
        # 7. conv3_2 (256 -> 256)
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self._init_layer(self.conv7, vgg[12])
        self.relu7 = nn.ReLU(inplace=True)
        
        # 8. conv3_3 (256 -> 256)
        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self._init_layer(self.conv8, vgg[14])
        self.relu8 = nn.ReLU(inplace=True)
        
        # 9. conv3_4 (256 -> 256)
        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self._init_layer(self.conv9, vgg[16])
        self.relu9 = nn.ReLU(inplace=True)
        
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # 10. conv4_1 (256 -> 512)
        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self._init_layer(self.conv10, vgg[19])
        self.relu10 = nn.ReLU(inplace=True)
        
        # 11. conv4_2 (512 -> 512)
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 0)
        self._init_layer(self.conv11, vgg[21])
        self.relu11 = nn.ReLU(inplace=True)
        
        # 12. conv4_3 (512 -> 512)
        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 0)
        self._init_layer(self.conv12, vgg[23])
        self.relu12 = nn.ReLU(inplace=True)
        
        # 13. conv4_4 (512 -> 512)
        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 0)
        self._init_layer(self.conv13, vgg[25])
        self.relu13 = nn.ReLU(inplace=True)
        
        self.maxPool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # 14. conv5_1 (512 -> 512) - This is the final layer in official implementation
        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 0)
        self._init_layer(self.conv14, vgg[28])
        self.relu14 = nn.ReLU(inplace=True)

    def _init_layer(self, my_conv, vgg_conv):
        with torch.no_grad():
            my_conv.weight.copy_(vgg_conv.weight)
            my_conv.bias.copy_(vgg_conv.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out1 = out  # relu1_1
        
        out = self.reflecPad3(out)
        out = self.conv3(out)
        out = self.relu3(out)
        
        out, _ = self.maxPool(out)
        
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out2 = out  # relu2_1
        
        out = self.reflecPad5(out)
        out = self.conv5(out)
        out = self.relu5(out)
        
        out, _ = self.maxPool2(out)
        
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out3 = out  # relu3_1
        
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        
        out, _ = self.maxPool3(out)
        
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out4 = out  # relu4_1
        
        out = self.reflecPad11(out)
        out = self.conv11(out)
        out = self.relu11(out)
        
        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)
        
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        
        out, _ = self.maxPool4(out)
        
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        # out is now relu5_1 (bottleneck)
        
        # Return: relu5_1, relu4_1, relu3_1, relu2_1, relu1_1
        return out, out4, out3, out2, out1


class PhotoNASDecoder(nn.Module):
    """
    PhotoNAS Decoder.
    Supports both full forward pass and stage-wise forward pass (for inference with WCT).
    """
    def __init__(self):
        super(PhotoNASDecoder, self).__init__()
        
        # Pooling layers for skip connections
        self.maxPool_mid1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxPool_mid2 = nn.MaxPool2d(kernel_size=4, stride=4, return_indices=True)
        self.maxPool_mid3 = nn.MaxPool2d(kernel_size=8, stride=8, return_indices=True)
        self.maxPool_mid4 = nn.MaxPool2d(kernel_size=16, stride=16, return_indices=True)
        
        # Instance Norms for skip connections
        self.nn01 = nn.InstanceNorm2d(512)
        self.nn02 = nn.InstanceNorm2d(256)
        self.nn03 = nn.InstanceNorm2d(128)
        self.nn04 = nn.InstanceNorm2d(64)
        
        # Pyramid Convolutions (1x1)
        # Input channels depend on concatenation of bottleneck and pooled skips
        # Bottleneck: 512
        # Skips: 512, 256, 128, 64
        
        self.conv_pyramid0 = nn.Conv2d(512, 512, 1, 1, 0)
        
        self.conv_pyramid11 = nn.Conv2d(512+512, 512, 1, 1, 0)
        self.conv_pyramid12 = nn.Conv2d(512+256, 512, 1, 1, 0)
        self.conv_pyramid13 = nn.Conv2d(512+128, 512, 1, 1, 0)
        self.conv_pyramid14 = nn.Conv2d(512+64, 512, 1, 1, 0)
        
        self.conv_pyramid212 = nn.Conv2d(512+512+256, 512, 1, 1, 0)
        self.conv_pyramid213 = nn.Conv2d(512+512+128, 512, 1, 1, 0)
        self.conv_pyramid214 = nn.Conv2d(512+512+64, 512, 1, 1, 0)
        self.conv_pyramid223 = nn.Conv2d(512+256+128, 512, 1, 1, 0)
        self.conv_pyramid224 = nn.Conv2d(512+256+64, 512, 1, 1, 0)
        self.conv_pyramid234 = nn.Conv2d(512+128+64, 512, 1, 1, 0)
        
        self.conv_pyramid3234 = nn.Conv2d(512+256+128+64, 512, 1, 1, 0)
        self.conv_pyramid3134 = nn.Conv2d(512+512+128+64, 512, 1, 1, 0)
        self.conv_pyramid3124 = nn.Conv2d(512+512+256+64, 512, 1, 1, 0)
        self.conv_pyramid3123 = nn.Conv2d(512+512+256+128, 512, 1, 1, 0)
        
        self.conv_pyramid4 = nn.Conv2d(512+960, 512, 1, 1, 0)  # 512 + 512+256+128+64 = 1472
        
        # Decoder Blocks
        
        # Block 1 (Stage 1)
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu15 = nn.ReLU(inplace=True)
        
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu16 = nn.ReLU(inplace=True)
        
        # Block 2 (Stage 2)
        self.nn1 = nn.InstanceNorm2d(512)
        
        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(1024, 512, 3, 1, 0)  # Concat with skip1 (512)
        self.conv17_2 = nn.Conv2d(512, 512, 3, 1, 0)  # No concat
        self.relu17 = nn.ReLU(inplace=True)
        
        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
        self.relu18 = nn.ReLU(inplace=True)
        
        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
        self.conv19_2 = nn.Conv2d(512, 256, 1, 1, 0)  # 1x1 conv variant
        self.relu19 = nn.ReLU(inplace=True)
        
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu20 = nn.ReLU(inplace=True)
        
        # Block 3 (Stage 3)
        self.nn2 = nn.InstanceNorm2d(256)
        
        self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(512, 256, 3, 1, 0)  # Concat with skip2 (256) -> 256+256=512
        self.conv21_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu21 = nn.ReLU(inplace=True)
        
        self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu22 = nn.ReLU(inplace=True)
        
        self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
        self.conv23_2 = nn.Conv2d(256, 128, 1, 1, 0)
        self.relu23 = nn.ReLU(inplace=True)
        
        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu24 = nn.ReLU(inplace=True)
        
        # Block 4 (Stage 4)
        self.nn3 = nn.InstanceNorm2d(128)
        
        self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(256, 64, 3, 1, 0)  # Concat with skip3 (128) -> 128+128=256
        self.conv25_2 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu25 = nn.ReLU(inplace=True)
        
        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu26 = nn.ReLU(inplace=True)
        
        # Block 5 (Stage 5)
        self.nn4 = nn.InstanceNorm2d(64)
        
        self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(128, 3, 3, 1, 0)  # Concat with skip4 (64) -> 64+64=128
        self.conv27_2 = nn.Conv2d(64, 3, 3, 1, 0)

    def parse_control(self, d_control):
        d0 = [int(x) for x in d_control[:5]]
        d1 = [int(x) for x in d_control[5:8]]
        d2 = [int(x) for x in d_control[9:16]]
        d3 = [int(x) for x in d_control[16:23]]
        d4 = [int(x) for x in d_control[23:28]]
        d5 = [int(x) for x in d_control[28:32]]
        return d0, d1, d2, d3, d4, d5

    def forward(self, x, skips, d_control):
        """Full forward pass."""
        # Execute all stages sequentially
        out = self.forward_stage0(x, skips, d_control)
        out = self.forward_stage1(out, skips, d_control)
        out = self.forward_stage2(out, skips, d_control)
        out = self.forward_stage3(out, skips, d_control)
        out = self.forward_stage4(out, skips, d_control)
        out = self.forward_stage5(out, skips, d_control)
        
        return out

    def forward_stage0(self, x, skips, d_control):
        """Stage 0: Pyramid feature fusion."""
        d0, _, _, _, _, _ = self.parse_control(d_control)
        skip1, skip2, skip3, skip4 = skips
        
        skip11 = self.nn01(skip1)
        skip22 = self.nn02(skip2)
        skip33 = self.nn03(skip3)
        skip44 = self.nn04(skip4)
        
        mid1, _ = self.maxPool_mid1(skip11)
        mid2, _ = self.maxPool_mid2(skip22)
        mid3, _ = self.maxPool_mid3(skip33)
        mid4, _ = self.maxPool_mid4(skip44)
        
        mid = x
        
        if d0[1] == 1:
            mid = torch.cat((mid, mid1), 1)
        if d0[2] == 1:
            mid = torch.cat((mid, mid2), 1)
        if d0[3] == 1:
            mid = torch.cat((mid, mid3), 1)
        if d0[4] == 1:
            mid = torch.cat((mid, mid4), 1)
        
        # Select pyramid conv based on concatenation pattern
        pattern = (d0[1], d0[2], d0[3], d0[4])
        
        if pattern == (0, 0, 0, 0):
            mid = self.conv_pyramid0(mid)
        elif pattern == (1, 0, 0, 0):
            mid = self.conv_pyramid11(mid)
        elif pattern == (0, 1, 0, 0):
            mid = self.conv_pyramid12(mid)
        elif pattern == (0, 0, 1, 0):
            mid = self.conv_pyramid13(mid)
        elif pattern == (0, 0, 0, 1):
            mid = self.conv_pyramid14(mid)
        elif pattern == (1, 1, 0, 0):
            mid = self.conv_pyramid212(mid)
        elif pattern == (1, 0, 1, 0):
            mid = self.conv_pyramid213(mid)
        elif pattern == (1, 0, 0, 1):
            mid = self.conv_pyramid214(mid)
        elif pattern == (0, 1, 1, 0):
            mid = self.conv_pyramid223(mid)
        elif pattern == (0, 1, 0, 1):
            mid = self.conv_pyramid224(mid)
        elif pattern == (0, 0, 1, 1):
            mid = self.conv_pyramid234(mid)
        elif pattern == (0, 1, 1, 1):
            mid = self.conv_pyramid3234(mid)
        elif pattern == (1, 1, 0, 1):
            mid = self.conv_pyramid3124(mid)
        elif pattern == (1, 1, 1, 0):
            mid = self.conv_pyramid3123(mid)
        elif pattern == (1, 0, 1, 1):
            mid = self.conv_pyramid3134(mid)
        elif pattern == (1, 1, 1, 1):
            mid = self.conv_pyramid4(mid)
        
        return mid

    def forward_stage1(self, x, skips, d_control):
        """Stage 1: First decoder block."""
        _, d1, _, _, _, _ = self.parse_control(d_control)
        out = x
        
        if d1[1] == 1:
            out = self.reflecPad15(out)
            out = self.conv15(out)
            out = self.relu15(out)
            
        out = self.unpool(out)
        
        if d1[2] == 1:
            out = self.reflecPad16(out)
            out = self.conv16(out)
            out = self.relu16(out)
            
        return out

    def forward_stage2(self, x, skips, d_control):
        """Stage 2: Second decoder block with skip connection."""
        _, _, d2, _, _, _ = self.parse_control(d_control)
        skip1 = skips[0]
        out = x
        
        if d2[1] == 1:
            skip1 = self.nn1(skip1)
            
        if d2[2] == 1:
            out = torch.cat((out, skip1), 1)
            out = self.reflecPad17(out)
            out = self.conv17(out)
            out = self.relu17(out)
        else:
            out = self.reflecPad17(out)
            out = self.conv17_2(out)
            out = self.relu17(out)
            
        if d2[3] == 1:
            out = self.reflecPad18(out)
            out = self.conv18(out)
            out = self.relu18(out)
            
        if d2[4] == 1:
            out = self.reflecPad19(out)
            out = self.conv19(out)
            out = self.relu19(out)
        else:
            out = self.conv19_2(out)
            out = self.relu19(out)
            
        out = self.unpool2(out)
        
        if d2[5] == 1:
            out = self.reflecPad20(out)
            out = self.conv20(out)
            out = self.relu20(out)
            
        return out

    def forward_stage3(self, x, skips, d_control):
        """Stage 3: Third decoder block with skip connection."""
        _, _, _, d3, _, _ = self.parse_control(d_control)
        skip2 = skips[1]
        out = x
        
        if d3[1] == 1:
            skip2 = self.nn2(skip2)
            
        if d3[2] == 1:
            out = torch.cat((out, skip2), 1)
            out = self.reflecPad21(out)
            out = self.conv21(out)
            out = self.relu21(out)
        else:
            out = self.reflecPad21(out)
            out = self.conv21_2(out)
            out = self.relu21(out)
            
        if d3[3] == 1:
            out = self.reflecPad22(out)
            out = self.conv22(out)
            out = self.relu22(out)
            
        if d3[4] == 1:
            out = self.reflecPad23(out)
            out = self.conv23(out)
            out = self.relu23(out)
        else:
            out = self.conv23_2(out)
            out = self.relu23(out)
            
        out = self.unpool3(out)
        
        if d3[5] == 1:
            out = self.reflecPad24(out)
            out = self.conv24(out)
            out = self.relu24(out)
            
        return out

    def forward_stage4(self, x, skips, d_control):
        """Stage 4: Fourth decoder block with skip connection."""
        _, _, _, _, d4, _ = self.parse_control(d_control)
        skip3 = skips[2]
        out = x
        
        if d4[1] == 1:
            skip3 = self.nn3(skip3)
            
        if d4[2] == 1:
            out = torch.cat((out, skip3), 1)
            out = self.reflecPad25(out)
            out = self.conv25(out)
            out = self.relu25(out)
        else:
            out = self.reflecPad25(out)
            out = self.conv25_2(out)
            out = self.relu25(out)
            
        out = self.unpool4(out)
        
        if d4[3] == 1:
            out = self.reflecPad26(out)
            out = self.conv26(out)
            out = self.relu26(out)
            
        return out

    def forward_stage5(self, x, skips, d_control):
        """Stage 5: Final output layer."""
        _, _, _, _, _, d5 = self.parse_control(d_control)
        skip4 = skips[3]
        out = x
        
        if d5[1] == 1:
            skip4 = self.nn4(skip4)
            
        if d5[2] == 1:
            out = torch.cat((out, skip4), 1)
            out = self.reflecPad27(out)
            out = self.conv27(out)
        else:
            out = self.reflecPad27(out)
            out = self.conv27_2(out)
            
        return out

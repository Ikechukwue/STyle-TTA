"""
HED (Holistically-Nested Edge Detection) implementation
Based on: https://github.com/sniklaus/pytorch-hed

Original paper:
@inproceedings{Xie_ICCV_2015,
    author = {Saining Xie and Zhuowen Tu},
    title = {Holistically-Nested Edge Detection},
    booktitle = {IEEE International Conference on Computer Vision},
    year = {2015}
}

PyTorch implementation:
@misc{pytorch-hed,
    author = {Simon Niklaus},
    title = {A Reimplementation of {HED} Using {PyTorch}},
    year = {2018},
    howpublished = {https://github.com/sniklaus/pytorch-hed}
}

To download the model:
wget https://github.com/sniklaus/pytorch-hed/raw/master/network-bsds500.pytorch -O hed.pth
"""

import torch


class Network(torch.nn.Module):
    """
    HED Network for edge detection.
    
    The model expects input in range [0, 1] and returns edge maps in range [0, 1].
    Input should be in format [B, 3, H, W].
    """
    
    def __init__(self, model_type='bsds500'):
        super().__init__()
        
        self.model_type = model_type
        
        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )
        
        # Load pretrained weights
        self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Load pretrained weights from torch.hub"""
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                url=f'http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch',
                file_name=f'hed-bsds500.pth',
                map_location='cpu'
            )
            # Rename keys from 'module' to 'net'
            state_dict = {key.replace('module', 'net'): weight for key, weight in state_dict.items()}
            self.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load pretrained HED weights: {e}")
            print("Model will be initialized with random weights")

    def forward(self, tenInput):
        """
        Forward pass through HED network.
        
        Args:
            tenInput: Input tensor in range [0, 1], shape [B, 3, H, W]
            
        Returns:
            Edge map tensor in range [0, 1], shape [B, 1, H, W]
        """
        # Normalize input (scale to [0, 255] and subtract mean)
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(
            data=[104.00698793, 116.66876762, 122.67891434],
            dtype=tenInput.dtype,
            device=tenInput.device
        ).view(1, 3, 1, 1)

        # Forward through VGG layers
        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        # Get scores from each layer
        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        # Upsample scores to input size
        tenScoreOne = torch.nn.functional.interpolate(
            input=tenScoreOne,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tenScoreTwo = torch.nn.functional.interpolate(
            input=tenScoreTwo,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tenScoreThr = torch.nn.functional.interpolate(
            input=tenScoreThr,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tenScoreFou = torch.nn.functional.interpolate(
            input=tenScoreFou,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tenScoreFiv = torch.nn.functional.interpolate(
            input=tenScoreFiv,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )

        # Combine scores and apply sigmoid
        return self.netCombine(torch.cat([
            tenScoreOne,
            tenScoreTwo,
            tenScoreThr,
            tenScoreFou,
            tenScoreFiv
        ], 1))

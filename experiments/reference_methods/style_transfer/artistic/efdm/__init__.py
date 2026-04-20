"""
EFDM (Exact Feature Distribution Matching) implementation.

xAILab Bamberg, University of Bamberg
Based on: https://github.com/YBZh/EFDM/tree/main/ArbitraryStyleTransfer


Paper Information:
------------------
Title: "Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization"
Authors: Yabin Zhang, Minghan Li, Ruihuang Li, Kui Jia, Lei Zhang
Conference: CVPR 2022
Paper: https://arxiv.org/abs/2203.07740
Code: https://github.com/YBZh/EFDM


Method Overview:
----------------
EFDM outperforms AdaIN by exactly matching feature distributions through Sort-Matching.
While AdaIN only matches mean and variance (1st and 2nd moments), EFDM matches the 
entire empirical CDF, implicitly matching all moments efficiently (O(n log n)).

Key Innovation:
- AdaIN assumes Gaussian distributions → only matches 2 moments
- EFDM makes no distribution assumptions → matches all moments via eCDF
- Efficient Sort-Matching algorithm avoids expensive histogram operations


Required Model Downloads:
--------------------------

EFDM requires the following pre-trained model:

1. VGG Encoder (vgg_normalised.pth):
   Download from: https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view?usp=sharing
   
   Local:   /data/local/colorist/checkpoints/reference_methods/efdm/vgg_normalised.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/efdm/vgg_normalised.pth
   
   SHA256: (original PyTorch-AdaIN implementation)

Optional (for inference with pre-trained decoder):

2. Pre-trained Decoder (decoder_iter_160000.pth.tar):
   Download from: https://drive.google.com/file/d/18nJHi8vHVvyRnOGgbNqkWyc61NAhRCzV/view?usp=sharing
   Alternative: https://pan.baidu.com/s/1qHlMiFaIvieAWakh_PCeTQ?pwd=vayi
   
   Local:   /data/local/colorist/checkpoints/reference_methods/efdm/decoder_iter_160000.pth.tar
   Cluster: $WORK/colorist/checkpoints/reference_methods/efdm/decoder_iter_160000.pth.tar


Training from Scratch:
-----------------------

To train EFDM from scratch, only the VGG encoder is required.
The decoder is trained on your datasets.

Training parameters (from official implementation):
- Learning rate: 1e-4
- LR decay: 5e-5 per iteration (lr = base_lr / (1 + decay * iter))
- Max iterations: 160,000
- Batch size: 8
- Content weight: 1.0
- Style weight: 10.0
- Image size: 512 → RandomCrop 256


Comparison with AdaIN:
----------------------

| Method | Matches | Algorithm | Complexity |
|--------|---------|-----------|------------|
| AdaIN  | Mean, Var (2 moments) | Closed-form | O(n) |
| EFDM   | All moments (eCDF) | Sort-Matching | O(n log n) |

EFDM provides better style transfer quality by matching the complete feature
distribution rather than just first-order and second-order statistics.


Usage Example:
--------------

```python
from experiments.reference_methods.style_transfer.training_required.efdm import EFDMMethod

# For inference with pre-trained model (single self-contained file)
method = EFDMMethod(pretrained_weights='path/to/bloodmnist-bloodmnist_efdm.pth')
output = method(content_image, style_image, alpha=1.0)

# For training
method = EFDMMethod()
method.initialize_for_training(pretrained_vgg_path='path/to/vgg_normalised.pth')
method.train(
    dataset_source='coco',
    dataset_reference='wikiart',
    data_path='/data/local/colorist/data',
    checkpoint_path='/data/local/colorist/checkpoints/efdm_training',
    final_model_path='/data/local/colorist/models/efdm_coco_wikiart.pth',
    accelerator=accelerator
)
```


Citation:
---------

If you use EFDM in your research, please cite:

```bibtex
@inproceedings{zhang2022exact,
  title={Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization},
  author={Zhang, Yabin and Li, Minghan and Li, Ruihuang and Jia, Kui and Zhang, Lei},
  booktitle={CVPR},
  year={2022}
}
```
"""

from .method import Method as EFDMMethod

__all__ = ['EFDMMethod']

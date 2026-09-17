"""
AesPA-Net (Aesthetic Pattern-Aware Style Transfer Networks) implementation.

xAILab Bamberg, University of Bamberg
Based on: https://github.com/Kibeom-Hong/AesPA-Net


Paper:
------
"AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks"
Authors: Kibeom Hong, Seogkyu Jeon, Junsoo Lee, Namhyuk Ahn, Kunhee Kim, 
         Pilhyeon Lee, Daesik Kim, Youngjung Uh, Hyeran Byun
Conference: ICCV 2023
Paper (CVF): https://openaccess.thecvf.com/content/ICCV2023/papers/Hong_AesPA-Net_Aesthetic_Pattern-Aware_Style_Transfer_Networks_ICCV_2023_paper.pdf
Paper (ArXiv): https://arxiv.org/abs/2307.09724


Required Model Downloads:
------------------------

AesPA-Net requires the following pre-trained models:

1. VGG Encoder (vgg_normalised.pth):
   Copy from AdaIN or other style transfer methods.
   
   Local:  /data/local/colorist/checkpoints/reference_methods/aespanet/vgg_normalised.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/aespanet/vgg_normalised.pth

Optional (for pre-trained decoder and transformer):
2. Pre-trained Decoder (dec_model.pth):
   Download from: https://drive.google.com/file/d/1nb7dQwj7RcQpi8_cURvErSwA-BxyZTT5/view?usp=sharing
   
   Local:  /data/local/colorist/checkpoints/reference_methods/aespanet/dec_model.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/aespanet/dec_model.pth

3. Pre-trained Transformer (transformer_model.pth):
   Download from: https://drive.google.com/file/d/1YII45EfR3mVbyvqQlzvfiYFIoTCgGG_R/view?usp=sharing
   
   Local:  /data/local/colorist/checkpoints/reference_methods/aespanet/transformer_model.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/aespanet/transformer_model.pth

   
Training from Scratch:
---------------------

To train AesPA-Net from scratch, only the VGG encoder is required.
The decoder and transformer will be initialized randomly and trained end-to-end.

Note: AesPA-Net uses a complex training procedure with:
- Adaptive pattern-aware attention mechanism
- Self-supervisory reconstruction tasks
- Multiple loss components (content, style, identity, contextual, color histogram, TV)
- Adversarial discriminator (optional - not used in final version)
"""

from .method import Method as AesPANetMethod

__all__ = ['AesPANetMethod']

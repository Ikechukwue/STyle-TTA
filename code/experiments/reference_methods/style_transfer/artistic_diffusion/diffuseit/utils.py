import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Resize
import clip
import numpy as np
import kornia.augmentation as K

# ============================================================================
# ViT Extractor (from model_vit/extractor.py)
# ============================================================================

def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix

class VitExtractor:
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)
        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)
        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)
        return _get_qkv_output

    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output)
        return _get_attn_output

    def get_feature_from_input(self, input_img):  # [B, 3, H, W]
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_values_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_values_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map

# ============================================================================
# Contrastive Loss (from model_vit/contra_loss.py)
# ============================================================================

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class PatchLoss(nn.Module):
    """Patch-based contrastive loss matching official DiffuseIT implementation."""
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        feat_q = Normalize()(feat_q)
        feat_k = Normalize()(feat_k)
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        
        # Compute norms for normalization
        feat_q_norm = feat_q.norm(dim=1, keepdim=True)
        feat_k_norm = feat_k.norm(dim=1, keepdim=True)
        
        # Positive logits (diagonal elements)
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos_norm = feat_k_norm * feat_q_norm
        l_pos = l_pos.view(batchSize, 1) / (l_pos_norm + 1e-7)
        
        # Negative logits
        batch_dim_for_bmm = 1
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        feat_q_norm = feat_q_norm.view(batch_dim_for_bmm, -1, 1)
        feat_k_norm = feat_k_norm.view(batch_dim_for_bmm, -1, 1)
        
        npatches = feat_q.size(1)
        l_neg_norm = torch.bmm(feat_q_norm, feat_k_norm.transpose(2, 1))
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) / (l_neg_norm + 1e-7)
        
        # Mask diagonal (positive pairs)
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)
        
        out = torch.cat((l_pos, l_neg), dim=1) / 0.07
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss

class ConstLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        feat_q = Normalize()(feat_q)
        feat_k = Normalize()(feat_k)
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit
        batch_dim_for_bmm = 1
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.07
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss

# ============================================================================
# ViT Loss (from model_vit/loss_vit.py)
# ============================================================================

class Loss_vit(nn.Module):
    def __init__(self, device, lambda_ssim=1.0, lambda_dir_cls=1.0, lambda_contra_ssim=1.0, lambda_trg=0.0):
        super().__init__()
        # Config for DINO (matching official: dino_vits16)
        self.dino_model_name = 'dino_vits16'
        self.dino_global_patch_size = 224
        
        self.extractor = VitExtractor(model_name=self.dino_model_name, device=device)
        
        imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(self.dino_global_patch_size, max_size=480)
        
        self.global_transform = T.Compose([
            global_resize_transform,
            imagenet_norm
        ])
        
        self.normalize = Normalize()
        self.lambdas = dict(
            lambda_global_ssim=lambda_ssim,
            lambda_dir_cls=lambda_dir_cls,
            lambda_contra_ssim=lambda_contra_ssim,
            lambda_trg=lambda_trg,
        )
        self.cossim = nn.CosineSimilarity(dim=0)
        self.patch_loss = PatchLoss()
        self.const_loss = ConstLoss()
        self.device = device

    def forward(self, outputs, source, out_prev=None, use_dir=True, target=None, frac_cont=1.0):
        losses = {}
        losses_val = {}
        loss_G = 0
        
        # Inputs are [-1, 1], convert to [0, 1]
        outputs = 0.5 * outputs + 0.5
        source = 0.5 * source + 0.5
        if out_prev is not None:
            out_prev = 0.5 * out_prev + 0.5
        if target is not None:
            target = 0.5 * target + 0.5
            
        if self.lambdas['lambda_global_ssim'] > 0:
            losses['loss_global_ssim'] = self.calculate_global_ssim_loss(outputs, source)
            loss_G += losses['loss_global_ssim'] * self.lambdas['lambda_global_ssim']
            losses_val['loss_global_ssim'] = losses['loss_global_ssim'].item()
            
        if self.lambdas['lambda_contra_ssim'] > 0:
            losses['loss_contra_ssim'] = self.calculate_contra_ssim_loss(outputs, source)
            loss_G += losses['loss_contra_ssim'] * self.lambdas['lambda_contra_ssim'] * frac_cont
            losses_val['loss_contra_ssim'] = losses['loss_contra_ssim'].item()
            
        if use_dir:
            if self.lambdas['lambda_dir_cls'] > 0:
                losses['loss_dir_cls'] = self.calculate_dir_cls_loss(outputs, out_prev)
                loss_G += losses['loss_dir_cls'] * self.lambdas['lambda_dir_cls']
                losses_val['loss_dir_cls'] = losses['loss_dir_cls'].item()
                
        if target is not None:
            if self.lambdas['lambda_trg'] > 0:
                losses['loss_trg'] = self.calculate_target_loss(outputs, target)
                loss_G += losses['loss_trg'] * self.lambdas['lambda_trg']
                
        return loss_G, losses_val

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_dir_cls_loss(self, outputs, out_prev):
        loss = 0.0
        for a, b in zip(outputs, out_prev):
            a = self.global_transform(a).unsqueeze(0).to(self.device)
            b = self.global_transform(b).unsqueeze(0).to(self.device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                prev_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss -= F.mse_loss(prev_cls_token, cls_token)
        return loss

    def calculate_target_loss(self, outputs, target):
        loss = 0.0
        for a, b in zip(outputs, target):
            a = self.global_transform(a).unsqueeze(0).to(self.device)
            b = self.global_transform(b).unsqueeze(0).to(self.device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                targ_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(targ_cls_token, cls_token)
        return loss

    def calculate_contra_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys = self.extractor.get_keys_from_input(a.unsqueeze(0), 11)
                h, t, d = target_keys.shape
                concatenated_target = target_keys.transpose(0, 1).reshape(t, h * d)
            keys = self.extractor.get_keys_from_input(b.unsqueeze(0), 11)
            h, t, d = keys.shape
            concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
            
            loss += self.patch_loss(concatenated_keys, concatenated_target).mean()
        loss /= len(inputs)
        return loss

# ============================================================================
# CLIP Wrapper (from src/vqc_core.py)
# ============================================================================

# ImageNet templates for text encoding (from official src/template.py)
IMAGENET_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def compose_text_with_templates(text, templates=IMAGENET_TEMPLATES):
    """Compose text with ImageNet templates."""
    return [template.format(text) for template in templates]


class ImageAugmentations(nn.Module):
    """Image augmentations for CLIP encoding (matching official DiffuseIT)."""
    def __init__(self, output_size, augmentations_number, p=0.7):
        super().__init__()
        self.output_size = output_size
        self.augmentations_number = augmentations_number
        self.augmentations = nn.Sequential(
            K.RandomAffine(degrees=15, translate=(0.1, 0.1), p=p, padding_mode="border"),
            K.RandomPerspective(0.7, p=p),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))

    def forward(self, input):
        """Extend input batch with augmentations.
        
        If input consists of images [I1, I2], the extended augmented output
        will be [I1_resized, I2_resized, I1_aug1, I2_aug1, I1_aug2, I2_aug2 ...]
        """
        resized_images = self.avg_pool(input)
        resized_images = torch.tile(resized_images, dims=(self.augmentations_number, 1, 1, 1))
        batch_size = input.shape[0]
        # Keep at least one non-augmented image
        non_augmented_batch = resized_images[:batch_size]
        augmented_batch = self.augmentations(resized_images[batch_size:])
        updated_batch = torch.cat([non_augmented_batch, augmented_batch], dim=0)
        return updated_batch


class CLIPWrapper(nn.Module):
    """CLIP model wrapper matching official DiffuseIT implementation."""
    def __init__(self, name, device='cpu', input_size=224, erasing=False):
        super().__init__()
        self.model, self.preprocess = clip.load(name, device=device, jit=False)
        self.model.eval()
        self.name = name
        self.device = device
        self.input_size = input_size
        
        self.norm = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        # Training preprocessing matching official (with optional erasing)
        train_transforms = []
        if erasing:
            # Add 4 RandomErasing augmentations as in official
            for _ in range(4):
                train_transforms.append(T.RandomErasing(p=0.5, scale=(0.02, 0.1)))
        train_transforms.extend([
            T.RandomRotation(10),
            T.RandomResizedCrop(self.input_size, scale=(0.8, 1), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            self.norm
        ])
        self.train_prep = nn.Sequential(*train_transforms)
        
        self.test_prep = nn.Sequential(
            T.Resize(self.input_size),
            T.CenterCrop(self.input_size),
            self.norm
        )

    def encode_image(self, x, ncuts=1):
        # x is [1, 3, H, W] or [3, H, W]
        if x.dim() == 4:
            x = x.squeeze(0)
            
        if ncuts > 0:
            x = torch.stack([self.train_prep(x) for _ in range(ncuts)])
        else:
            x = self.test_prep(x)[None]
            
        x = self.model.encode_image(x.to(self.device))
        x = x / x.norm(dim=-1, keepdim=True)
        return x.mean(0)

    def encode_text(self, text):
        """Encode text using ImageNet templates (matching official)."""
        with torch.no_grad():
            output = []
            for text_item in text if isinstance(text, list) else [text]:
                # Compose text with templates
                text_templates = compose_text_with_templates(text_item, IMAGENET_TEMPLATES)
                tokens = clip.tokenize(text_templates).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features = text_features.mean(axis=0, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                output.append(text_features)
            if len(output) == 1:
                return output[0]
            return output

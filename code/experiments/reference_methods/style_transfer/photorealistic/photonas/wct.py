"""
xAILab Bamberg
University of Bamberg

@description:
Whitening and Coloring Transform (WCT) implementation for PhotoNAS.
Based on: https://github.com/pkuanjie/StyleNAS/blob/master/PhotoNAS/wct.py
"""

import torch


def whiten_and_color(cF, sF):
    """
    Whitening and Coloring Transform.
    
    Args:
        cF: Content features (C, H*W)
        sF: Style features (C, H*W)
        
    Returns:
        Transformed features (C, H*W)
    """
    cFSize = cF.size()
    c_mean = torch.mean(cF, 1)  # c
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    # Content Covariance
    c_gram = torch.mm(cF, cF.t()).div(cFSize[1] - 1)
    c_choose = torch.eye(cFSize[0]).to(cF.device).double()
    
    # Stabilize Gram matrix (matches official implementation)
    contentConv = (1 - c_choose) * c_gram + 2.0 * c_choose * c_gram + 0.5 * c_gram
    
    # SVD
    try:
        c_u, c_e, c_v = torch.linalg.svd(contentConv, full_matrices=False)
    except AttributeError:
        c_u, c_e, c_v = torch.svd(contentConv, some=False)
    except RuntimeError:
        # Fallback if SVD fails
        return cF + c_mean

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.000001:
            k_c = i
            break

    sFSize = sF.size()
    s_mean = torch.mean(sF, 1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    
    # Style Covariance
    styleConv = torch.mm(sF, sF.t()).div(sFSize[1] - 1)
    
    try:
        s_u, s_e, s_v = torch.linalg.svd(styleConv, full_matrices=False)
    except AttributeError:
        s_u, s_e, s_v = torch.svd(styleConv, some=False)
    except RuntimeError:
        return cF + c_mean

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.000001:
            k_s = i
            break
            
    # Whiten
    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cF)

    # Color
    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
    
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    return targetFeature


def transform(cF, sF, alpha):
    """
    Apply WCT to content features using style features.
    
    Args:
        cF: Content features (B, C, H, W)
        sF: Style features (B, C, H, W)
        alpha: Blending factor (0-1)
        
    Returns:
        Transformed features (B, C, H, W)
    """
    cF = cF.double()
    sF = sF.double()
    
    B, C, H, W = cF.size()
    
    output = torch.zeros_like(cF)
    
    for i in range(B):
        c_feat = cF[i].view(C, -1)
        s_feat = sF[i].view(C, -1)
        
        target_feat = whiten_and_color(c_feat, s_feat)
        
        # Blend
        cs_feat = alpha * target_feat + (1.0 - alpha) * c_feat
        
        output[i] = cs_feat.view(C, H, W)
        
    return output.float()

"""
xAILab Bamberg
University of Bamberg

@description:
Preprocessing utilities.
"""

# Imports
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors
from typing import Any, Dict, List


class ResizeWhileRetainAspectRatio(v2.Transform):
    """
    Custom transform to resize an image while retaining its aspect ratio.
    
    This transform supports arbitrary input structures (images, bounding boxes, labels, etc.)
    following the torchvision v2 Transform API. It properly handles:
    - Regular images (PIL, Tensor, tv_tensors.Image)
    - Bounding boxes (resizes and adjusts coordinates)
    - ReferenceImage types (when colorist is available)
    - Labels and other non-visual data (passes through unchanged)
    
    Args:
        size (int): The target size for both dimensions after padding.
        fill (int): The fill value for padding. Default: 0
    
    Example:
        >>> transform = ResizeWhileRetainAspectRatio(size=224)
        >>> # Works with single image
        >>> img_out = transform(img)
        >>> # Works with (img, label) tuple
        >>> img_out, label_out = transform(img, label)
        >>> # Works with arbitrary structures
        >>> output = transform({"img": img, "bboxes": bboxes, "label": label})
    """
    def __init__(self, size: int, fill: int = 0):
        """
        Initialize the transform.

        Args:
            size (int): The target size for both dimensions after padding.
            fill (int): The fill value for padding. Default: 0
        """
        super().__init__()
        self.size = size
        self.fill = fill
    
    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        """
        Generate parameters for the transform.
        
        This method is called once per transform invocation with all inputs.
        For this transform, no random parameters are needed.
        
        Args:
            flat_inputs: Flattened list of all inputs
            
        Returns:
            Empty dictionary (no parameters needed for this transform)
        """
        return {}
    
    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """
        Transform a single input.
        
        This method is called for each input in the flattened input structure.
        It handles different input types appropriately:
        - Images: resized while maintaining aspect ratio, then padded
        - Bounding boxes: resized to match new image dimensions
        - Labels/other data: passed through unchanged
        
        Args:
            inpt: Individual input (can be image, bounding box, label, etc.)
            params: Parameters from make_params() (unused for this transform)
            
        Returns:
            Transformed input of the same type as input
        """        
        # Handle bounding boxes specially
        if isinstance(inpt, tv_tensors.BoundingBoxes):
            return self._resize_bounding_boxes(inpt)
        
        # Check if this is an image-like input that should be transformed
        is_image = isinstance(inpt, (Image.Image, torch.Tensor, tv_tensors.Image))
        
        # Skip non-image inputs (labels, masks without special handling, etc.)
        if not is_image:
            return inpt
        
        # Remember original type for reconstruction
        is_pil_image = isinstance(inpt, Image.Image)
        original_type = type(inpt)

        # Convert input to tensor
        inpt_tensor = F.to_image(inpt)

        # Get the original dimensions
        h, w = inpt_tensor.shape[-2:]

        # Determine the new size and padding
        if h == w:
            # If the image is already square, just resize it
            resized_inpt = F.resize(inpt_tensor, (self.size, self.size))
        else:
            # Resize the longer edge to the target size
            if h > w:
                new_h, new_w = self.size, int(self.size * w / h)
            else:
                new_h, new_w = int(self.size * h / w), self.size
            
            resized_inpt = F.resize(inpt_tensor, (new_h, new_w))

            # Calculate padding to make it square
            pad_h = (self.size - new_h) // 2
            pad_w = (self.size - new_w) // 2
            padding = [pad_w, pad_h, self.size - new_w - pad_w, self.size - new_h - pad_h]

            # Apply padding
            resized_inpt = F.pad(resized_inpt, padding, fill=self.fill)

        # Clamp values to valid range to prevent interpolation artifacts
        # Resize interpolation can produce values slightly outside the original range
        if resized_inpt.dtype in (torch.float32, torch.float64, torch.float16):
            # Detect value range from the input to determine clamping bounds
            # If input was in [0, 1] range, clamp to [0, 1]
            # If input was in [0, 255] range, clamp to [0, 255]
            max_val = inpt_tensor.max().item()
            if max_val <= 1.0:
                # Input was in [0, 1] range
                resized_inpt = torch.clamp(resized_inpt, 0.0, 1.0)
            else:
                # Input was in [0, 255] range
                resized_inpt = torch.clamp(resized_inpt, 0.0, 255.0)
        elif resized_inpt.dtype == torch.uint8:
            # For uint8, clamp to [0, 255]
            resized_inpt = torch.clamp(resized_inpt, 0, 255)

        # Convert back to original format, preserving special types like ReferenceImage
        if is_pil_image:
            resized_inpt = F.to_pil_image(resized_inpt)
        elif original_type != torch.Tensor and issubclass(original_type, tv_tensors.TVTensor):
            # Preserve custom TVTensor types (like ReferenceImage)
            resized_inpt = tv_tensors.wrap(resized_inpt, like=inpt)

        return resized_inpt
    
    def _resize_bounding_boxes(self, bboxes: tv_tensors.BoundingBoxes) -> tv_tensors.BoundingBoxes:
        """
        Resize bounding boxes to match the resized image dimensions.
        
        Args:
            bboxes: BoundingBoxes to resize
            
        Returns:
            Resized BoundingBoxes with updated coordinates and canvas_size
        """
        # Get original canvas size
        orig_h, orig_w = bboxes.canvas_size
        
        # Calculate scaling and padding (same logic as image resize)
        if orig_h == orig_w:
            # Square image: simple resize
            scale_h = scale_w = self.size / orig_h
            pad_left = pad_top = 0
        else:
            # Non-square: resize + pad
            if orig_h > orig_w:
                new_h, new_w = self.size, int(self.size * orig_w / orig_h)
            else:
                new_h, new_w = int(self.size * orig_h / orig_w), self.size
            
            scale_h = new_h / orig_h
            scale_w = new_w / orig_w
            
            pad_top = (self.size - new_h) // 2
            pad_left = (self.size - new_w) // 2
        
        # Scale and shift bounding boxes
        scaled_bboxes = bboxes.clone()
        
        if bboxes.format == tv_tensors.BoundingBoxFormat.XYXY:
            # Format: [x1, y1, x2, y2]
            scaled_bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale_w + pad_left
            scaled_bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale_h + pad_top
        elif bboxes.format == tv_tensors.BoundingBoxFormat.XYWH:
            # Format: [x, y, w, h]
            scaled_bboxes[:, 0] = bboxes[:, 0] * scale_w + pad_left
            scaled_bboxes[:, 1] = bboxes[:, 1] * scale_h + pad_top
            scaled_bboxes[:, 2] = bboxes[:, 2] * scale_w
            scaled_bboxes[:, 3] = bboxes[:, 3] * scale_h
        elif bboxes.format == tv_tensors.BoundingBoxFormat.CXCYWH:
            # Format: [cx, cy, w, h]
            scaled_bboxes[:, 0] = bboxes[:, 0] * scale_w + pad_left
            scaled_bboxes[:, 1] = bboxes[:, 1] * scale_h + pad_top
            scaled_bboxes[:, 2] = bboxes[:, 2] * scale_w
            scaled_bboxes[:, 3] = bboxes[:, 3] * scale_h
        
        # Return with updated canvas size
        return tv_tensors.wrap(
            scaled_bboxes,
            like=bboxes,
            canvas_size=(self.size, self.size)
        )
#!/usr/bin/env python3
"""
SAM Segmentation Module for Object Extraction.
Uses Hugging Face transformers SAM for background removal.

Usage:
    from sam_segmentation import SAMSegmenter, apply_mask_to_image
    
    segmenter = SAMSegmenter()
    mask = segmenter.segment_center_object(image_rgb)
    masked_image = apply_mask_to_image(image_bgr, mask)
"""

import numpy as np
import cv2
from typing import Tuple, Optional

# Lazy import for torch and transformers
_sam_model = None
_sam_processor = None


def get_sam_model(model_name: str = "facebook/sam-vit-base", device: str = None):
    """
    Get or create SAM model (singleton pattern for efficiency).
    
    Args:
        model_name: HuggingFace model name
        device: Device to use (cuda/cpu). Auto-detected if None.
        
    Returns:
        (model, processor, device)
    """
    global _sam_model, _sam_processor
    
    import torch
    from transformers import SamModel, SamProcessor
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if _sam_model is None:
        print(f"Loading SAM model: {model_name}")
        print(f"Device: {device}")
        
        _sam_processor = SamProcessor.from_pretrained(model_name)
        _sam_model = SamModel.from_pretrained(model_name).to(device)
        _sam_model.eval()
        
        print("✓ SAM model loaded successfully")
    
    return _sam_model, _sam_processor, device


class SAMSegmenter:
    """SAM-based object segmentation using Hugging Face transformers."""
    
    def __init__(self, model_name: str = "facebook/sam-vit-base", device: str = None):
        """
        Initialize SAM segmenter.
        
        Args:
            model_name: HuggingFace model name 
                       (facebook/sam-vit-base, facebook/sam-vit-large, facebook/sam-vit-huge)
            device: Device to use (cuda/cpu). Auto-detected if None.
        """
        import torch
        self.torch = torch
        self.model, self.processor, self.device = get_sam_model(model_name, device)
    
    def segment_with_point(self, image: np.ndarray, point: Tuple[int, int], 
                           point_label: int = 1) -> np.ndarray:
        """
        Segment object using a point prompt.
        
        Args:
            image: Input image (H, W, 3) in RGB
            point: (x, y) coordinate of the point prompt
            point_label: 1 for foreground, 0 for background
            
        Returns:
            Binary mask (H, W) where 1 = object, 0 = background
        """
        # Prepare inputs
        inputs = self.processor(
            images=image,
            input_points=[[[point[0], point[1]]]],
            input_labels=[[point_label]],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate mask
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Get the best mask (highest score)
        scores = outputs.iou_scores.cpu().numpy()[0, 0]
        best_mask_idx = np.argmax(scores)
        mask = masks[0][0, best_mask_idx].numpy().astype(np.uint8)
        
        return mask
    
    def segment_with_box(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Segment object using a bounding box prompt.
        
        Args:
            image: Input image (H, W, 3) in RGB
            box: (x_min, y_min, x_max, y_max) bounding box
            
        Returns:
            Binary mask (H, W) where 1 = object, 0 = background
        """
        # Prepare inputs
        inputs = self.processor(
            images=image,
            input_boxes=[[list(box)]],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate mask
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Get the best mask (highest score)
        scores = outputs.iou_scores.cpu().numpy()[0, 0]
        best_mask_idx = np.argmax(scores)
        mask = masks[0][0, best_mask_idx].numpy().astype(np.uint8)
        
        return mask
    
    def segment_center_object(self, image: np.ndarray, 
                               center_ratio: Tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
        """
        Segment object at the center of the image.
        Useful for objects placed in the middle of the frame.
        
        Args:
            image: Input image (H, W, 3) in RGB
            center_ratio: (x_ratio, y_ratio) where 0.5, 0.5 is exact center
            
        Returns:
            Binary mask (H, W)
        """
        h, w = image.shape[:2]
        center_x = int(w * center_ratio[0])
        center_y = int(h * center_ratio[1])
        
        return self.segment_with_point(image, (center_x, center_y), point_label=1)


def refine_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Refine mask using morphological operations.
    
    Args:
        mask: Binary mask
        kernel_size: Size of morphological kernel
        
    Returns:
        Refined mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray, 
                        background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Apply binary mask to image, replacing background with solid color.
    
    Args:
        image: Input image (H, W, 3) in BGR (OpenCV format)
        mask: Binary mask (H, W) where 1 = keep, 0 = replace
        background_color: BGR color for background (default: white)
        
    Returns:
        Masked image with background replaced
    """
    # Create background
    background = np.full_like(image, background_color)
    
    # Apply mask
    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    result = np.where(mask_3ch == 1, image, background)
    
    return result


def segment_frame(frame_bgr: np.ndarray, segmenter: SAMSegmenter,
                  prompt_type: str = "center",
                  prompt_value: Optional[str] = None,
                  background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Segment a single frame and apply mask.
    
    Args:
        frame_bgr: Input frame in BGR format (OpenCV)
        segmenter: SAMSegmenter instance
        prompt_type: "center", "point", or "box"
        prompt_value: Prompt value string (e.g., "x,y" for point)
        background_color: Background color (BGR)
        
    Returns:
        Masked frame with background replaced
    """
    # Convert BGR to RGB for SAM
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Generate mask based on prompt type
    if prompt_type == "center":
        mask = segmenter.segment_center_object(frame_rgb)
    elif prompt_type == "point" and prompt_value:
        px, py = map(int, prompt_value.split(","))
        mask = segmenter.segment_with_point(frame_rgb, (px, py))
    elif prompt_type == "box" and prompt_value:
        coords = list(map(int, prompt_value.split(",")))
        mask = segmenter.segment_with_box(frame_rgb, tuple(coords))
    else:
        mask = segmenter.segment_center_object(frame_rgb)
    
    # Refine mask
    mask = refine_mask(mask)
    
    # Apply mask
    masked_frame = apply_mask_to_image(frame_bgr, mask, background_color)
    
    return masked_frame


def interactive_prompt_selection(first_frame_bgr: np.ndarray, 
                                  save_path: str = "/tmp/first_frame_preview.png") -> Tuple[str, Optional[str]]:
    """
    Interactive mode to select prompt for segmentation.
    Shows first frame info and lets user choose prompt type and value.
    
    Args:
        first_frame_bgr: First frame in BGR format
        save_path: Path to save preview image
        
    Returns:
        (prompt_type, prompt_value)
    """
    h, w = first_frame_bgr.shape[:2]
    
    print("\n" + "="*60)
    print("SAM PROMPT SELECTION")
    print("="*60)
    print(f"Frame size: {w}x{h}")
    print(f"Center point: ({w//2}, {h//2})")
    print()
    print("Options:")
    print("  1. Use center point (default) - good for centered objects")
    print("  2. Specify custom point (x,y)")
    print("  3. Specify bounding box (x1,y1,x2,y2)")
    print()
    
    # Save first frame for reference
    cv2.imwrite(save_path, first_frame_bgr)
    print(f"First frame saved to: {save_path}")
    print("(Open this file to see where to place your prompt)")
    print()
    
    choice = input("Enter choice [1/2/3] (default=1): ").strip() or "1"
    
    if choice == "1":
        return "center", None
    elif choice == "2":
        point = input(f"Enter point as x,y (e.g., {w//2},{h//2}): ").strip()
        return "point", point
    elif choice == "3":
        box = input("Enter box as x1,y1,x2,y2: ").strip()
        return "box", box
    else:
        print("Invalid choice, using center")
        return "center", None

from inspect import cleandoc
import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
import torch
from PIL import Image
import torchvision.transforms as transformsclass

class Example:
    """
    A ComfyUI node to feather a tile image into a base image at specified coordinates.

    Class methods
    -------------
    INPUT_TYPES (dict):
        Defines input parameters for the node.
    IS_CHANGED:
        Optional method to control when the node is re-executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tuple.
    RETURN_NAMES (`tuple`):
        Names of each output in the output tuple.
    FUNCTION (`str`):
        The entry-point method name.
    OUTPUT_NODE (`bool`):
        Indicates if this is an output node.
    CATEGORY (`str`):
        The category for the node in the UI.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
           "required": {
               "base_image_path": ("STRING", { "tooltip": "This is an image"}),
               "tile_path": ("STRING", { "tooltip": "This is an tile path"}),
               "output_path": ("STRING", { "tooltip": "Output path"}),
               "x": ("INT", {"min": 0, "max": 10000}),
               "y": ("INT", {"min": 0, "max": 10000}),
               "feather_width": ("INT",  {"min": 0, "max": 10000}),
           },
    }
    
    #RETURN_TYPES = ()
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "feather_tile_into_image"
    
    OUTPUT_NODE = True
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node
    
    CATEGORY = "Example"

    def create_feather_mask(self, height, width, alpha, feather_width):
        """
        Create a feathering mask based on the alpha channel of the tile.
        
        Args:
            height: Height of the tile
            width: Width of the tile
            alpha: Alpha channel of the tile (0 for transparent, 255 for opaque)
            feather_width: Width of feathering region
        Returns:
            Mask with values 0 to 1, feathering from visible pixel boundaries
        """
        # Ensure feather is at least 1
        feather = max(1, feather_width // 2)
        
        # Normalize alpha to binary (0 for transparent, 1 for visible)
        binary_alpha = (alpha > 0).astype(np.float32)
        
        # Compute distance transform from the edge of visible pixels
        # Distance is 0 at transparent pixels, positive inside visible regions
        dist = distance_transform_edt(binary_alpha)
        
        # Create mask: 0 at transparent pixels, ramping up to 1 inside visible area
        mask = np.clip(dist / feather, 0, 1)
        
        return mask
    
    def feather_tile_into_image(self, base_image_path, tile_path, output_path, x, y, feather_width):
        """
        Feather a tile with alpha channel into a 5376x5376 image at coordinates (x, y).
        
        Args:
            base_image_path: Path to the base image (5376x5376, RGB)
            tile_path: Path to the tile image (any size, RGBA)
            x: X-coordinate of the top-left corner of the tile's bounding box
            y: Y-coordinate of the top-left corner of the tile's bounding box
            feather_width: Width of feathering region for blending
            output_path: Path to save the resulting image
        """
        # Resolve paths
        base_image_path = Path(base_image_path)
        tile_path = Path(tile_path)
        output_path = Path(output_path)
    
        # Load base image (RGB)
        base_img = cv2.imread(str(base_image_path), cv2.IMREAD_COLOR)
        if base_img is None:
            raise ValueError(f"Failed to load base image: {base_image_path}")
        
        height, width, channels = base_img.shape
    
        # Load tile (RGBA)
        tile = cv2.imread(str(tile_path), cv2.IMREAD_UNCHANGED)
        if tile is None:
            raise ValueError(f"Failed to load tile image: {tile_path}")
        
        tile_height, tile_width, tile_channels = tile.shape
        if tile_channels != 4:
            raise ValueError(f"Tile must have an alpha channel (expected 4 channels, got {tile_channels})")
        
        # Check bounds
        if x < 0 or y < 0 or x + tile_width > width or y + tile_height > height:
            raise ValueError(
                f"Tile of size {tile_width}x{tile_height} at (x={x}, y={y}) "
                f"is out of bounds for {width}x{height} image"
            )
    
        # Adjust feather width to avoid exceeding tile dimensions
        feather_width = min(feather_width, tile_width // 2, tile_height // 2)
    
        # Split tile into RGB and alpha
        tile_rgb = tile[:, :, :3].astype(np.float32)
        tile_alpha = tile[:, :, 3].astype(np.float32) / 255.0  # Normalize to [0, 1]
    
        # Create feathering mask based on alpha channel
        feather_mask = self.create_feather_mask(tile_height, tile_width, tile_alpha, feather_width)
        feather_mask = np.stack([feather_mask] * 3, axis=2)  # Match RGB channels
    
        # Convert base image to float
        base_img = base_img.astype(np.float32)
    
        # Extract the region of the base image
        base_region = base_img[y:y+tile_height, x:x+tile_width]
    
        # Blend using alpha and feather mask
        # Effective mask combines alpha (for transparency) and feather mask (for blending)
        effective_mask = tile_alpha[:, :, np.newaxis] * feather_mask
        blended_region = base_region * (1 - effective_mask) + tile_rgb * effective_mask
    
        # Update base image
        base_img[y:y+tile_height, x:x+tile_width] = blended_region
        
        base_img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
    
        base_img_tensor = base_img_rgb / 255.0
    
        # Save result
        base_img = np.clip(base_img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_path), base_img)

        result_tensor = torch.from_numpy(base_img_tensor).float().unsqueeze(0)  # Shape: (1, height, width, 3)
        print(f"Feathered tile of size {tile_width}x{tile_height} into image and saved to {output_path}")
        return (result_tensor,)
#        return {}

NODE_CLASS_MAPPINGS = {
    "Example": Example
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Example Node"
}

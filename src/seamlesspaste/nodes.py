from inspect import cleandoc
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

class Example:
    """
    A ComfyUI node to feather a tile image into a base image at specified coordinates.
    This version properly feathers only from visible areas and supports bi-directional (in/out) feathering.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE", {"tooltip": "Base image (RGB, tensor)"}),
                "tile_image": ("IMAGE", {"tooltip": "Tile image (RGBA or RGB, tensor)"}),
                "x": ("INT", {"min": 0, "max": 10000, "tooltip": "X-coordinate for tile placement"}),
                "y": ("INT", {"min": 0, "max": 10000, "tooltip": "Y-coordinate for tile placement"}),
                "feather_width": ("INT", {"min": 0, "max": 10000, "tooltip": "Width of feathering region"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_output",)
    FUNCTION = "feather_tile_into_image"
    CATEGORY = "Example"
    OUTPUT_NODE = True
    DESCRIPTION = cleandoc(__doc__)

    # -------------------------------------------------------------------------
    # Feather Mask Creation
    # -------------------------------------------------------------------------
    def create_feather_mask(self, alpha, feather_width):
        """
        Create a bi-directional feathering mask that starts from visible alpha edges.
        Feathers both inward (toward opaque) and outward (toward transparent) smoothly.

        Args:
            alpha (np.ndarray): Alpha channel (0 to 1)
            feather_width (int): Width of feathering in pixels

        Returns:
            np.ndarray: Mask (0..1) where 1 = fully visible, 0 = transparent
        """
        if feather_width <= 0:
            return alpha.astype(np.float32)

        # Binary mask for where pixels are visible
        binary_alpha = (alpha > 0).astype(np.uint8)

        # Distances from transparency and from opacity
        dist_inside = distance_transform_edt(binary_alpha)          # inside opaque region
        dist_outside = distance_transform_edt(1 - binary_alpha)     # outside opaque region

        # Normalized distances
        inside_mask = np.clip(dist_inside / feather_width, 0, 1)
        outside_mask = np.clip(1 - (dist_outside / feather_width), 0, 1)

        # Combine for a smooth bi-directional falloff
        combined_mask = np.where(binary_alpha > 0, inside_mask, outside_mask)

        # Multiply by alpha to respect existing transparency
        return combined_mask * alpha

    # -------------------------------------------------------------------------
    # Main blending logic
    # -------------------------------------------------------------------------
    def feather_tile_into_image(self, base_image, tile_image, x, y, feather_width):
        """
        Feather a tile image into a base image at coordinates (x, y).

        Args:
            base_image (torch.Tensor): Base image tensor (1, H, W, 3), RGB, [0,1]
            tile_image (torch.Tensor): Tile image tensor (1, h, w, 3 or 4), RGB(A), [0,1]
            x (int): X-coordinate of the top-left corner of the tile
            y (int): Y-coordinate of the top-left corner of the tile
            feather_width (int): Width of feathering region

        Returns:
            (torch.Tensor,): Resulting blended image tensor (1, H, W, 3)
        """
        # Convert from torch to numpy
        base_image = base_image[0].cpu().numpy()  # (H, W, 3)
        tile_image = tile_image[0].cpu().numpy()  # (h, w, 3 or 4)

        height, width, channels = base_image.shape
        tile_height, tile_width, tile_channels = tile_image.shape

        # Validate
        if channels != 3:
            raise ValueError(f"Base image must have 3 channels (RGB), got {channels}")
        if tile_channels not in (3, 4):
            raise ValueError(f"Tile image must have 3 or 4 channels, got {tile_channels}")

        # Extract RGB + Alpha
        if tile_channels == 4:
            tile_rgb = tile_image[:, :, :3]
            tile_alpha = tile_image[:, :, 3]
        else:
            tile_rgb = tile_image
            tile_alpha = np.ones((tile_height, tile_width), dtype=np.float32)

        # Check placement bounds
        if x < 0 or y < 0 or x + tile_width > width or y + tile_height > height:
            raise ValueError(
                f"Tile of size {tile_width}x{tile_height} at (x={x}, y={y}) "
                f"is out of bounds for {width}x{height} image"
            )

        # Limit feather width
        feather_width = min(feather_width, tile_width // 2, tile_height // 2)

        # --- Create feathering mask ---
        feather_mask = self.create_feather_mask(tile_alpha, feather_width)
        feather_mask = np.stack([feather_mask] * 3, axis=2)  # (h, w, 3)

        # --- Extract region from base image ---
        base_img = base_image.astype(np.float32)
        base_region = base_img[y:y + tile_height, x:x + tile_width]

        # --- Blend ---
        blended_region = base_region * (1 - feather_mask) + tile_rgb * feather_mask

        # --- Update base image ---
        base_img[y:y + tile_height, x:x + tile_width] = blended_region

        # Convert back to tensor
        result_tensor = torch.from_numpy(np.clip(base_img, 0, 1)).float().unsqueeze(0)
        cv2.imwrite("base_img_100.png", (base_img * 255).astype(np.uint8))
        return (result_tensor,)


# -------------------------------------------------------------------------
# Node Mappings for ComfyUI
# -------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "Example": Example
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Example Feather Node"
}

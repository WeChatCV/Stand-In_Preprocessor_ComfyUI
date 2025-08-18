import torch

class VideoFramePreprocessor:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",), # Input is a batch of video frames
            }
        }

    # CHANGED: Added three INT outputs for width, height, and frame_count
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("processed_images", "width", "height", "frame_count")
    FUNCTION = "process_frames"
    CATEGORY = "Stand-In"

    def process_frames(self, images: torch.Tensor):
        if images.dim() != 4:
            raise ValueError("Input must be a batch of images (video frames).")

        total_frames, original_h, original_w, _ = images.shape
        print(f"Original video specs: {total_frames} frames, {original_w}x{original_h}")

        # 1. Trim frame count to be 4n+1
        new_total_frames = total_frames - ((total_frames - 1) % 4)
        
        if new_total_frames != total_frames:
            print(f"Trimming frames to be 4n+1: {total_frames} -> {new_total_frames}")
            images = images[:new_total_frames, :, :, :]
        else:
            print("Frame count already meets 4n+1 requirement. No trimming needed.")

        # 2. Crop dimensions to the nearest multiple of 16 (rounding down)
        new_h = (original_h // 16) * 16
        new_w = (original_w // 16) * 16

        if new_h != original_h or new_w != original_w:
            print(f"Cropping dimensions to a multiple of 16: {original_w}x{original_h} -> {new_w}x{new_h}")
            
            h_to_remove = original_h - new_h
            w_to_remove = original_w - new_w
            
            h_start = h_to_remove // 2
            w_start = w_to_remove // 2
            
            processed_images = images[:, h_start : h_start + new_h, w_start : w_start + new_w, :]
        else:
            print("Dimensions are already multiples of 16. No cropping needed.")
            processed_images = images
            
        # Get final dimensions from the processed tensor
        final_frames, final_h, final_w, _ = processed_images.shape
        
        print(f"Final video specs: {final_frames} frames, {final_w}x{final_h}")

        # CHANGED: Return the processed images along with the final dimensions
        return (processed_images, final_w, final_h, final_frames)


NODE_CLASS_MAPPINGS = {
    "VideoFramePreprocessor": VideoFramePreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFramePreprocessor": "Stand-In Trimmer & Cropper",
}
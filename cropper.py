import torch

class VideoFramePreprocessor:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",), # Input is a batch of video frames
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_images",)
    FUNCTION = "process_frames"
    CATEGORY = "Stand-In" # Placed in a sub-category for organization

    def process_frames(self, images: torch.Tensor):
        # Input tensor shape: (batch_size/frames, height, width, channels)
        if images.dim() != 4:
            raise ValueError("Input must be a batch of images (video frames).")

        total_frames, original_h, original_w, _ = images.shape
        print(f"Original video specs: {total_frames} frames, {original_w}x{original_h}")

        # 1. Trim frame count to be 4n+1
        # We find the largest number <= total_frames that satisfies the condition.
        new_total_frames = total_frames - ((total_frames - 1) % 4)
        
        if new_total_frames != total_frames:
            print(f"Trimming frames to be 4n+1: {total_frames} -> {new_total_frames}")
            images = images[:new_total_frames, :, :, :]
        else:
            print("Frame count already meets 4n+1 requirement. No trimming needed.")

        # 2. Crop dimensions to the nearest multiple of 8 (rounding down)
        new_h = (original_h // 8) * 8
        new_w = (original_w // 8) * 8

        if new_h != original_h or new_w != original_w:
            print(f"Cropping dimensions to a multiple of 8: {original_w}x{original_h} -> {new_w}x{new_h}")
            
            # Calculate the amount to remove from each side for a center crop
            h_to_remove = original_h - new_h
            w_to_remove = original_w - new_w
            
            h_start = h_to_remove // 2
            w_start = w_to_remove // 2
            
            # Perform the centered crop using tensor slicing
            processed_images = images[:, h_start : h_start + new_h, w_start : w_start + new_w, :]
        else:
            print("Dimensions are already multiples of 8. No cropping needed.")
            processed_images = images
            
        print(f"Final video specs: {processed_images.shape[0]} frames, {processed_images.shape[2]}x{processed_images.shape[1]}")

        return (processed_images,)


NODE_CLASS_MAPPINGS = {
    "VideoFramePreprocessor": VideoFramePreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFramePreprocessor": "Stand-In Trimmer & Cropper",
}

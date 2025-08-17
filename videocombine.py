import torch
import cv2
import numpy as np
from torchvision.transforms.functional import normalize
from tqdm import tqdm
from PIL import Image

# --- Helper Functions (can be shared if in the same file as the original node) ---

def tensor_to_cv2_img(tensor_frame: torch.Tensor) -> np.ndarray:
    """Converts a single PyTorch image tensor (H, W, C) to a CV2 image (H, W, C) in BGR format."""
    img_np = (tensor_frame.cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def tensor_to_pil(tensor_frame: torch.Tensor, mode='RGB') -> Image.Image:
    """Converts a single PyTorch image tensor (H, W, C) to a PIL Image."""
    return Image.fromarray((tensor_frame.cpu().numpy() * 255).astype(np.uint8), mode)

# --- Updated Node: VideoBackgroundRestorer ---

class VideoBackgroundRestorer:
    """
    Analyzes a synthesized video to create a face mask and then uses this mask
    to composite the synthesized face onto the background of an original video.
    Includes edge dilation and feathering for seamless blending.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_processor": ("FACE_PROCESSOR",),
                "synth_images": ("IMAGE",),
                "orig_images": ("IMAGE",),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "face_crop_scale": ("FLOAT", {"default": 1.8, "min": 1.0, "max": 3.0, "step": 0.1}),
                "dilation_kernel_size": ("INT", {"default": 10, "min": 0, "max": 50, "step": 1}),
                "feather_amount": ("INT", {"default": 50, "min": 0, "max": 151, "step": 2, "display": "slider"}),
                "with_neck": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_video",)
    FUNCTION = "restore_background"
    CATEGORY = "Stand-In"

    def restore_background(self, face_processor, synth_images: torch.Tensor, orig_images: torch.Tensor, confidence_threshold: float, face_crop_scale: float, dilation_kernel_size: int, feather_amount: int, with_neck: bool):
        detection_model, parsing_model, device = face_processor
        
        if synth_images.shape != orig_images.shape:
            raise ValueError("Synthesized and original videos must have the same dimensions and frame count.")

        total_frames, h, w = synth_images.shape[0], synth_images.shape[1], synth_images.shape[2]
        
        print(f"Processing {total_frames} frames ({w}x{h}) to restore background with edge feathering.")

        processed_frames_tensors = []

        for i in tqdm(range(total_frames), desc="Restoring video background"):
            synth_frame_tensor = synth_images[i]
            orig_frame_tensor = orig_images[i]
            
            synth_frame_bgr = tensor_to_cv2_img(synth_frame_tensor)
            
            results = detection_model(synth_frame_bgr, verbose=False)
            confident_boxes = results[0].boxes.xyxy[results[0].boxes.conf > confidence_threshold]

            full_mask_np = np.zeros((h, w), dtype=np.uint8)

            if confident_boxes.shape[0] > 0:
                areas = (confident_boxes[:, 2] - confident_boxes[:, 0]) * (confident_boxes[:, 3] - confident_boxes[:, 1])
                x1, y1, x2, y2 = map(int, confident_boxes[torch.argmax(areas)])

                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                side_len = int(max(x2 - x1, y2 - y1) * face_crop_scale)
                half_side = side_len // 2
                
                crop_y1, crop_x1 = max(center_y - half_side, 0), max(center_x - half_side, 0)
                crop_y2, crop_x2 = min(center_y + half_side, h), min(center_x + half_side, w)
                
                face_crop_bgr = synth_frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

                if face_crop_bgr.size > 0:
                    face_resized = cv2.resize(face_crop_bgr, (512, 512), interpolation=cv2.INTER_AREA)
                    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                    face_tensor_in = torch.from_numpy(face_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

                    with torch.no_grad():
                        normalized_face = normalize(face_tensor_in, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        parsing_map = parsing_model(normalized_face)[0].argmax(dim=1, keepdim=True)
                    
                    parsing_map_np = parsing_map.squeeze().cpu().numpy().astype(np.uint8)
                    if with_neck:
                        final_mask_512 = (parsing_map_np != 0).astype(np.uint8) * 255
                    else:
                        parts_to_exclude = [0, 14, 15, 16, 18] 
                        final_mask_512 = np.isin(parsing_map_np, parts_to_exclude, invert=True).astype(np.uint8) * 255

                    # --- EDGE PROCESSING LOGIC ---
                    # 1. DILATION (Expansion)
                    if dilation_kernel_size > 0:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
                        final_mask_512 = cv2.dilate(final_mask_512, kernel, iterations=1)
                    
                    # 2. FEATHERING (Softening)
                    if feather_amount > 0:
                        # Ensure the kernel size is odd
                        if feather_amount % 2 == 0:
                            feather_amount += 1
                        # Apply Gaussian Blur to soften the edges of the mask
                        final_mask_512 = cv2.GaussianBlur(final_mask_512, (feather_amount, feather_amount), 0)

                    mask_resized_to_crop = cv2.resize(final_mask_512, (face_crop_bgr.shape[1], face_crop_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                    full_mask_np[crop_y1:crop_y2, crop_x1:crop_x2] = mask_resized_to_crop
            
            mask_tensor = torch.from_numpy(full_mask_np.astype(np.float32) / 255.0).unsqueeze(-1).to(device)
            
            combined_frame = synth_frame_tensor.to(device) * mask_tensor + orig_frame_tensor.to(device) * (1 - mask_tensor)
            
            processed_frames_tensors.append(combined_frame)
        
        output_image_batch = torch.stack(processed_frames_tensors).cpu()
        
        return (output_image_batch,)

NODE_CLASS_MAPPINGS = {
    "VideoBackgroundRestorer": VideoBackgroundRestorer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
   "VideoBackgroundRestorer": "Stand-In Background Restorer",
}
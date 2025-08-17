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
    Includes edge dilation, feathering, color matching, and a face-only mode.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_processor": ("FACE_PROCESSOR",),
                "synth_images": ("IMAGE",),
                "orig_images": ("IMAGE",),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "face_crop_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "dilation_kernel_size": ("INT", {"default": 25, "min": 0, "max": 50, "step": 1}),
                "feather_amount": ("INT", {"default": 50, "min": 0, "max": 151, "step": 2, "display": "slider"}),
                "with_neck": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "color_match_enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "color_match_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                # NEW: Face Only Pasting Mode
                "face_only_mode": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_video",)
    FUNCTION = "restore_background"
    CATEGORY = "Stand-In"

    def restore_background(self, face_processor, synth_images: torch.Tensor, orig_images: torch.Tensor, confidence_threshold: float, face_crop_scale: float, dilation_kernel_size: int, feather_amount: int, with_neck: bool, color_match_enabled: bool, color_match_strength: float, face_only_mode: bool):
        detection_model, parsing_model, device = face_processor
        
        if synth_images.shape != orig_images.shape:
            raise ValueError("Synthesized and original videos must have the same dimensions and frame count.")

        total_frames, h, w = synth_images.shape[0], synth_images.shape[1], synth_images.shape[2]
        
        print(f"Processing {total_frames} frames ({w}x{h}) to restore background.")

        processed_frames_tensors = []

        # Define parts to exclude for face-only skin analysis and pasting
        parts_to_exclude_for_face_only = [0, 14, 15, 16, 17, 18] # bg, neck, cloth, hair etc.

        for i in tqdm(range(total_frames), desc="Restoring video background"):
            synth_frame_tensor = synth_images[i]
            orig_frame_tensor = orig_images[i]
            
            synth_frame_bgr = tensor_to_cv2_img(synth_frame_tensor)
            orig_frame_bgr = tensor_to_cv2_img(orig_frame_tensor)
            
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
                    
                    if color_match_enabled and color_match_strength > 0:
                        face_skin_mask_512 = np.isin(parsing_map_np, parts_to_exclude_for_face_only, invert=True).astype(np.uint8)
                        
                        if np.sum(face_skin_mask_512) > 0:
                            face_skin_mask_crop = cv2.resize(face_skin_mask_512, (face_crop_bgr.shape[1], face_crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

                            orig_face_crop_bgr = orig_frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                            source_lab = cv2.cvtColor(orig_face_crop_bgr, cv2.COLOR_BGR2LAB)
                            target_lab = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2LAB)
                            
                            source_mean, source_std = cv2.meanStdDev(source_lab, mask=face_skin_mask_crop)
                            target_mean, target_std = cv2.meanStdDev(target_lab, mask=face_skin_mask_crop)

                            l, a, b = cv2.split(target_lab)
                            
                            eps = 1e-6
                            l = (l - target_mean[0][0]) * (source_std[0][0] / (target_std[0][0] + eps)) + source_mean[0][0]
                            a = (a - target_mean[1][0]) * (source_std[1][0] / (target_std[1][0] + eps)) + source_mean[1][0]
                            b = (b - target_mean[2][0]) * (source_std[2][0] / (target_std[2][0] + eps)) + source_mean[2][0]

                            corrected_lab = cv2.merge([l, a, b])
                            corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
                            
                            corrected_face_crop_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
                            
                            face_crop_bgr = cv2.addWeighted(corrected_face_crop_bgr, color_match_strength, face_crop_bgr, 1 - color_match_strength, 0)

                            synth_frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2] = face_crop_bgr
                            
                            corrected_synth_rgb = cv2.cvtColor(synth_frame_bgr, cv2.COLOR_BGR2RGB)
                            synth_frame_tensor = torch.from_numpy(corrected_synth_rgb.astype(np.float32) / 255.0)

                    # --- NEW: MASK SELECTION LOGIC ---
                    if face_only_mode:
                        # If face_only_mode is ON, use the precise skin mask and ignore with_neck setting
                        final_mask_512 = np.isin(parsing_map_np, parts_to_exclude_for_face_only, invert=True).astype(np.uint8) * 255
                    elif with_neck:
                        # Standard mode: include neck and hair
                        final_mask_512 = (parsing_map_np != 0).astype(np.uint8) * 255
                    else:
                        # Standard mode: exclude neck
                        parts_to_exclude_neck = [0, 14, 15, 16, 18]
                        final_mask_512 = np.isin(parsing_map_np, parts_to_exclude_neck, invert=True).astype(np.uint8) * 255
                    # --- END OF MASK SELECTION ---

                    if dilation_kernel_size > 0:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
                        final_mask_512 = cv2.dilate(final_mask_512, kernel, iterations=1)
                    
                    if feather_amount > 0:
                        if feather_amount % 2 == 0:
                            feather_amount += 1
                        final_mask_512 = cv2.GaussianBlur(final_mask_512, (feather_amount, feather_amount), 0)

                    mask_resized_to_crop = cv2.resize(final_mask_512, (face_crop_bgr.shape[1], face_crop_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                    full_mask_np[crop_y1:crop_y2, crop_x1:crop_x2] = mask_resized_to_crop
            
            mask_tensor = torch.from_numpy(full_mask_np.astype(np.float32) / 255.0).unsqueeze(-1).to(device)
            
            combined_frame = synth_frame_tensor.to(device) * mask_tensor + orig_frame_tensor.to(device) * (1 - mask_tensor)
            
            processed_frames_tensors.append(combined_frame)
        
        output_image_batch = torch.stack(processed_frames_tensors).cpu()
        
        return (output_image_batch,)
import torch
import cv2
import numpy as np
from torchvision.transforms.functional import normalize
from tqdm import tqdm
from PIL import Image
import random

# Helper function to convert a ComfyUI IMAGE tensor to an OpenCV BGR image.
def tensor_to_cv2_img(tensor_frame: torch.Tensor) -> np.ndarray:
    """Converts a single PyTorch image tensor (H, W, C) to a CV2 image (H, W, C) in BGR format."""
    img_np = (tensor_frame.cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# Helper function to convert a ComfyUI IMAGE tensor to a PIL Image.
def tensor_to_pil(tensor_frame: torch.Tensor, mode='RGB') -> Image.Image:
    """Converts a single PyTorch image tensor (H, W, C) to a PIL Image."""
    return Image.fromarray((tensor_frame.cpu().numpy() * 255).astype(np.uint8), mode)

class VideoInputPreprocessor:
    """
    Processes a batch of image frames to generate a face mask. It can either paste a
    provided RGBA image into the face's bounding box (default) or use a generated,
    feathered mask for a precise, non-rectangular composite ("face only" mode).
    It also supports random horizontal flipping of the input face on a per-frame basis.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_processor": ("FACE_PROCESSOR",),
                "images": ("IMAGE",),
                "face_rgba": ("IMAGE",), 
                "denoise_strength": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "face_crop_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                "dilation_kernel_size": ("INT", {"default": 10, "min": 0, "max": 50, "step": 1}),
                "with_neck": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "face_only_mode": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "feather_amount": ("INT", {"default": 21, "min": 0, "max": 151, "step": 2, "display": "slider"}),
                "random_horizontal_flip_chance": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("processed_images", "denoise_strength")
    FUNCTION = "generate_mask_and_paste"
    CATEGORY = "Stand-In"

    def generate_mask_and_paste(self, face_processor, images: torch.Tensor, face_rgba: torch.Tensor, denoise_strength: float, confidence_threshold: float, face_crop_scale: float, dilation_kernel_size: int, with_neck: bool, face_only_mode: bool, feather_amount: int, random_horizontal_flip_chance: float):
        detection_model, parsing_model, device = face_processor
        total_frames, h, w = images.shape[0], images.shape[1], images.shape[2]
        
        print(f"Processing {total_frames} frames ({w}x{h}) to paste new face.")

        if face_rgba.shape[3] != 4:
            raise ValueError("The 'face_to_paste' image must be an RGBA image.")
        face_to_paste_pil = tensor_to_pil(face_rgba[0], mode='RGBA')

        processed_frames_tensors = []

        for i in tqdm(range(total_frames), desc="Pasting face onto frames"):
            frame_tensor = images[i]
            frame_bgr = tensor_to_cv2_img(frame_tensor)
            
            results = detection_model(frame_bgr, verbose=False)
            confident_boxes = results[0].boxes.xyxy[results[0].boxes.conf > confidence_threshold]

            target_frame_pil = tensor_to_pil(frame_tensor).copy()

            if confident_boxes.shape[0] > 0:
                areas = (confident_boxes[:, 2] - confident_boxes[:, 0]) * (confident_boxes[:, 3] - confident_boxes[:, 1])
                x1, y1, x2, y2 = map(int, confident_boxes[torch.argmax(areas)])

                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                side_len = int(max(x2 - x1, y2 - y1) * face_crop_scale)
                half_side = side_len // 2
                
                crop_y1, crop_x1 = max(center_y - half_side, 0), max(center_x - half_side, 0)
                crop_y2, crop_x2 = min(center_y + half_side, h), min(center_x + half_side, w)
                
                x, y = crop_x1, crop_y1
                box_w, box_h = crop_x2 - crop_x1, crop_y2 - crop_y1

                if box_w > 0 and box_h > 0:
                    # Resize the source face image first
                    face_to_paste_resized = face_to_paste_pil.resize((box_w, box_h), Image.Resampling.LANCZOS)

                    # --- MODIFIED: Per-frame random flip logic is now INSIDE the loop ---
                    if random.random() < random_horizontal_flip_chance:
                        face_to_paste_resized = face_to_paste_resized.transpose(Image.FLIP_LEFT_RIGHT)
                    # --- END OF MODIFICATION ---

                    if not face_only_mode:
                        target_frame_pil.paste(face_to_paste_resized, (x, y), face_to_paste_resized)
                    else:
                        face_crop_bgr = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        if face_crop_bgr.size > 0:
                            face_resized = cv2.resize(face_crop_bgr, (512, 512), interpolation=cv2.INTER_AREA)
                            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                            face_tensor_in = torch.from_numpy(face_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

                            with torch.no_grad():
                                normalized_face = normalize(face_tensor_in, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                parsing_map = parsing_model(normalized_face)[0].argmax(dim=1, keepdim=True)
                            
                            parsing_map_np = parsing_map.squeeze().cpu().numpy().astype(np.uint8)
                            
                            parts_to_exclude = [0, 14, 15, 16, 17, 18]
                            final_mask_512 = np.isin(parsing_map_np, parts_to_exclude, invert=True).astype(np.uint8) * 255

                            if dilation_kernel_size > 0:
                                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
                                final_mask_512 = cv2.dilate(final_mask_512, kernel, iterations=1)
                            
                            if feather_amount > 0:
                                if feather_amount % 2 == 0:
                                    feather_amount += 1
                                final_mask_512 = cv2.GaussianBlur(final_mask_512, (feather_amount, feather_amount), 0)
                            
                            mask_resized_to_crop = cv2.resize(final_mask_512, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
                            generated_mask_pil = Image.fromarray(mask_resized_to_crop, mode='L')
                            
                            target_frame_pil.paste(face_to_paste_resized, (x, y), mask=generated_mask_pil)

            processed_np = np.array(target_frame_pil.convert("RGB")).astype(np.float32) / 255.0
            processed_tensor = torch.from_numpy(processed_np)
            processed_frames_tensors.append(processed_tensor)
        
        output_image_batch = torch.stack(processed_frames_tensors)
        
        return (output_image_batch, denoise_strength)
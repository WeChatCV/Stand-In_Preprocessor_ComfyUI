import torch
import cv2
import numpy as np
from torchvision.transforms.functional import normalize
from tqdm import tqdm
from PIL import Image
import random

# (Helper functions remain the same)
def tensor_to_cv2_img(tensor_frame: torch.Tensor) -> np.ndarray:
    img_np = (tensor_frame.cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def tensor_to_cv2_bgra_img(tensor_frame: torch.Tensor) -> np.ndarray:
    if tensor_frame.shape[2] != 4:
        raise ValueError("Input tensor must be an RGBA image with 4 channels.")
    img_np = (tensor_frame.cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)

def tensor_to_pil(tensor_frame: torch.Tensor, mode='RGB') -> Image.Image:
    if tensor_frame.shape[2] == 4 and mode == 'RGB':
      mode = 'RGBA'
    elif tensor_frame.shape[2] == 3 and mode == 'RGBA':
      mode = 'RGB'
    return Image.fromarray((tensor_frame.cpu().numpy() * 255).astype(np.uint8), mode)


class VideoInputPreprocessor:
    def __init__(self):
        self.landmark_model = None

    @classmethod
    def INPUT_TYPES(cls):
        # (INPUT_TYPES definition is unchanged)
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
                "random_horizontal_flip_chance": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05}),
                "match_angle_and_size": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("processed_images", "denoise_strength")
    FUNCTION = "generate_mask_and_paste"
    CATEGORY = "Stand-In"

    def _get_mediapipe_landmarks(self, image_bgr: np.ndarray):
        # (This helper method is unchanged)
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.landmark_model.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0]
        key_indices = [33, 263, 1, 61, 291]
        landmarks = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in key_indices])
        return landmarks

    def generate_mask_and_paste(self, face_processor, images: torch.Tensor, face_rgba: torch.Tensor, denoise_strength: float, confidence_threshold: float, face_crop_scale: float, dilation_kernel_size: int, with_neck: bool, face_only_mode: bool, feather_amount: int, random_horizontal_flip_chance: float, match_angle_and_size: bool):
        detection_model, parsing_model, device = face_processor
        total_frames, h, w = images.shape[0], images.shape[1], images.shape[2]
        
        print(f"Processing {total_frames} frames ({w}x{h}) to paste new face.")

        if face_rgba.shape[-1] != 4:
            raise ValueError("The 'face_to_paste' image must be an RGBA image with 4 channels.")
        
        source_kpts = None
        face_to_paste_cv2 = None
        
        if match_angle_and_size:
            if self.landmark_model is None:
                print("Angle matching enabled. Initializing MediaPipe Face Mesh model...")
                try: import mediapipe as mp
                except ImportError: raise ImportError("MediaPipe is required for 'match_angle_and_size'. Please run: pip install mediapipe")
                mp_face_mesh = mp.solutions.face_mesh
                self.landmark_model = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
                print("MediaPipe model loaded and cached.")

            face_to_paste_cv2 = tensor_to_cv2_bgra_img(face_rgba[0])
            source_kpts = self._get_mediapipe_landmarks(face_to_paste_cv2)
            if source_kpts is None:
                print("[Warning] No landmarks found in source 'face_rgba' image. Angle matching disabled.")
                match_angle_and_size = False

        face_to_paste_pil = tensor_to_pil(face_rgba[0], mode='RGBA')

        processed_frames_tensors = []

        for i in tqdm(range(total_frames), desc="Pasting face onto frames"):
            frame_tensor = images[i]
            frame_bgr = tensor_to_cv2_img(frame_tensor)
            
            results = detection_model(frame_bgr, verbose=False)
            confident_boxes = results[0].boxes.xyxy[results[0].boxes.conf > confidence_threshold]
            
            pasted = False
            if confident_boxes.shape[0] > 0:
                if match_angle_and_size:
                    target_kpts = self._get_mediapipe_landmarks(frame_bgr)
                    if target_kpts is not None:
                        # --- THIS IS THE UPGRADED ALGORITHM ---
                        # Use all 5 keypoints to estimate a more robust similarity transform (rotation, uniform scale, translation)
                        M, _ = cv2.estimateAffinePartial2D(source_kpts, target_kpts, method=cv2.LMEDS)

                        if M is not None:
                            b, g, r, a = cv2.split(face_to_paste_cv2)
                            source_rgb = cv2.merge([b, g, r])
                            
                            warped_face = cv2.warpAffine(source_rgb, M, (w, h))
                            warped_alpha = cv2.warpAffine(a, M, (w, h))
                            
                            alpha_float = warped_alpha.astype(np.float32) / 255.0
                            alpha_expanded = np.expand_dims(alpha_float, axis=2)
                            
                            frame_bgr = (1.0 - alpha_expanded) * frame_bgr + alpha_expanded * warped_face
                            frame_bgr = frame_bgr.astype(np.uint8)
                            pasted = True
                        else:
                            print(f"[Warning] Frame {i}: Could not compute a stable affine transform. Falling back to box pasting.")

                if not pasted: # Fallback logic for box-pasting
                    # (The entire box-pasting logic from the previous version goes here, unchanged)
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
                        source_img = face_to_paste_pil.copy()
                        if random.random() < random_horizontal_flip_chance:
                            source_img = source_img.transpose(Image.FLIP_LEFT_RIGHT)
                        source_w, source_h = source_img.size
                        dest_aspect = box_w / box_h
                        source_aspect = source_w / source_h
                        if source_aspect > dest_aspect:
                            new_w = box_w
                            new_h = int(new_w / source_aspect)
                        else:
                            new_h = box_h
                            new_w = int(new_h * source_aspect)
                        face_resized_correct_aspect = source_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        canvas = Image.new('RGBA', (box_w, box_h), (0, 0, 0, 0))
                        paste_x = (box_w - new_w) // 2
                        paste_y = (box_h - new_h) // 2
                        canvas.paste(face_resized_correct_aspect, (paste_x, paste_y))
                        face_to_paste_resized = canvas
                        
                        target_frame_pil = tensor_to_pil(frame_tensor).copy()
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
                                    if feather_amount % 2 == 0: feather_amount += 1
                                    final_mask_512 = cv2.GaussianBlur(final_mask_512, (feather_amount, feather_amount), 0)
                                mask_resized_to_crop = cv2.resize(final_mask_512, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
                                generated_mask_pil = Image.fromarray(mask_resized_to_crop, mode='L')
                                target_frame_pil.paste(face_to_paste_resized, (x, y), mask=generated_mask_pil)
                        frame_bgr = tensor_to_cv2_img(torch.from_numpy(np.array(target_frame_pil).astype(np.float32) / 255.0))
            
            processed_np = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            processed_tensor = torch.from_numpy(processed_np.astype(np.float32) / 255.0)
            processed_frames_tensors.append(processed_tensor)
        
        output_image_batch = torch.stack(processed_frames_tensors)
        
        return (output_image_batch, denoise_strength)
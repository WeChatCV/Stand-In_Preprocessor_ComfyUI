import os
import cv2
import requests
import torch
import numpy as np
import PIL.Image
import PIL.ImageOps
from ultralytics import YOLO
from facexlib.parsing import init_parsing_model
from torchvision.transforms.functional import normalize
from typing import Union, Optional
import folder_paths
from packaging.version import parse as parse_version
# --- GLOBALS ---
LOADED_PROCESSORS = {}

# --- DIRECTORY SETUP ---
def set_extra_config_model_path(extra_config_models_dir_key, models_dir_name: str):
    """Helper function to set up model directories within ComfyUI."""
    models_dir_default = os.path.join(folder_paths.models_dir, models_dir_name)
    if not os.path.exists(models_dir_default):
        os.makedirs(models_dir_default, exist_ok=True)

    if extra_config_models_dir_key not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[extra_config_models_dir_key] = (
            [models_dir_default],
            folder_paths.supported_pt_extensions,
        )
    else:
        folder_paths.add_model_folder_path(extra_config_models_dir_key, models_dir_default, is_default=True)

# Set up paths for YOLO and face parsing models
set_extra_config_model_path("yolo", "yolo")
set_extra_config_model_path("face_parsing", "face_parsing")

# --- MODEL DOWNLOADING ---
def download_yolo_model(model_name="model.pt"):
    """
    Checks if a YOLO model exists and downloads it from a public source if not.
    """
    model_url = f"https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/{model_name}"
    yolo_dir = os.path.join(folder_paths.get_folder_paths("yolo")[0])
    model_path = os.path.join(yolo_dir, model_name)

    os.makedirs(yolo_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Model '{model_name}' not found locally. Starting download from public source...")
        try:
            with requests.get(model_url, stream=True, timeout=60) as r:
                if r.status_code == 401:
                    print(f"\nDownload failed: Received 401 Unauthorized. The link '{model_url}' may require authentication.")
                    raise requests.exceptions.HTTPError(f"401 Client Error: Unauthorized for url: {model_url}")
                r.raise_for_status()
                
                total_size = int(r.headers.get('content-length', 0))
                with open(model_path, 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        done = int(50 * downloaded / total_size) if total_size > 0 else 0
                        progress_mb = downloaded / (1024*1024)
                        total_mb = total_size / (1024*1024)
                        print(f"\r[{'=' * done}{' ' * (50-done)}] {progress_mb:.2f}MB / {total_mb:.2f}MB", end='')
            print(f"\nModel successfully downloaded to: {model_path}")
        except Exception as e:
            print(f"\nFailed to download model: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
            raise e
    return model_path

# --- IMAGE UTILS ---
def _img2tensor(img: np.ndarray, bgr2rgb: bool = True) -> torch.Tensor:
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)

def _pad_to_square(img: np.ndarray, pad_color: int = 255) -> np.ndarray:
    h, w, _ = img.shape
    if h == w:
        return img
    if h > w:
        pad_size = (h - w) // 2
        padded_img = cv2.copyMakeBorder(img, 0, 0, pad_size, h - w - pad_size, cv2.BORDER_CONSTANT, value=[pad_color] * 3)
    else:
        pad_size = (w - h) // 2
        padded_img = cv2.copyMakeBorder(img, pad_size, w - h - pad_size, 0, 0, cv2.BORDER_CONSTANT, value=[pad_color] * 3)
    return padded_img

def tensor_to_cv2_img(tensor: torch.Tensor) -> np.ndarray:
    """Converts a ComfyUI IMAGE tensor (B, H, W, C) to an OpenCV BGR image (H, W, C)."""
    img_np = tensor.squeeze(0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr

def cv2_img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Converts an OpenCV BGR image (H, W, C) to a ComfyUI IMAGE tensor (1, H, W, C)."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)
    return img_tensor

# --- NODES ---
class FaceProcessorLoader:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "yolo_model_name": ("STRING", {"default": "model.pt"}) } }

    RETURN_TYPES = ("FACE_PROCESSOR",)
    RETURN_NAMES = ("face_processor",)
    FUNCTION = "load_processor"
    CATEGORY = "Stand-In"

    def load_processor(self, yolo_model_name="model.pt"):
        if os.path.isdir(yolo_model_name):
            print(f"Warning: You provided a directory ('{yolo_model_name}') instead of a filename. Using default model 'model.pt'.")
            yolo_model_name = "model.pt"

        processor_key = f"face_processor_{yolo_model_name}"
        if processor_key in LOADED_PROCESSORS:
            print("Reusing cached face processor model.")
            return (LOADED_PROCESSORS[processor_key],)

        print("Initializing face processor models...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            if not yolo_model_name.endswith(('.pt', '.onnx')):
                raise ValueError(f"Invalid model name: '{yolo_model_name}'. Model filename must end with '.pt' or '.onnx'.")
            
            model_path = download_yolo_model(yolo_model_name)
            
            print(f"Loading YOLO model from local path: {model_path}")
            detection_model = None
            TORCH_VERSION = parse_version(torch.__version__)
            original_torch_load = torch.load
            if TORCH_VERSION >= parse_version("2.6.0"):
                print(f"PyTorch version {TORCH_VERSION} >= 2.6.0 detected. Applying compatibility patch for model loading.")
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_torch_load(*args, **kwargs)
                torch.load = patched_load
            
            try:
                detection_model = YOLO(model_path)
            finally:
                torch.load = original_torch_load
                if TORCH_VERSION >= parse_version("2.6.0"):
                    print("Restored original torch.load.")
            
            if detection_model is None:
                raise Exception("YOLO model could not be loaded.")

            detection_model.to(device)
            print("YOLO model loaded successfully.")

        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise e

        parsing_model_path = os.path.join(folder_paths.get_folder_paths("face_parsing")[0])
        parsing_model = init_parsing_model(model_name="bisenet", half=False, model_rootpath=parsing_model_path, device=device)
        parsing_model.eval()
        print("Face parsing model loaded successfully.")

        processor_tuple = (detection_model, parsing_model, device)
        LOADED_PROCESSORS[processor_key] = processor_tuple
        
        print("Face processor (YOLO detection + BiSeNet parsing) initialized and cached successfully.")
        
        return (processor_tuple,)


class ApplyFaceProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_processor": ("FACE_PROCESSOR",),
                "image": ("IMAGE",),
                "resize_to": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "border_thresh": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "face_crop_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "with_neck": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                # NEW: Face Only Mode
                "face_only_mode": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("processed_image", "face_rgba")
    FUNCTION = "apply_processing"
    CATEGORY = "Stand-In"

    def apply_processing(self, face_processor, image, resize_to, border_thresh, face_crop_scale, confidence_threshold, with_neck, face_only_mode):
        detection_model, parsing_model, device = face_processor
        
        frame = tensor_to_cv2_img(image)
        h, w, _ = frame.shape
        image_to_process = None

        results = detection_model(frame, verbose=False)
        
        boxes = results[0].boxes.xyxy
        conf = results[0].boxes.conf
        confident_boxes = boxes[conf > confidence_threshold]

        if confident_boxes.shape[0] == 0:
            print("[Warning] No confident face detected. Using the whole image padded to a square.")
            image_to_process = _pad_to_square(frame, pad_color=255)
        else:
            areas = (confident_boxes[:, 2] - confident_boxes[:, 0]) * (confident_boxes[:, 3] - confident_boxes[:, 1])
            largest_face_idx = torch.argmax(areas)
            x1, y1, x2, y2 = map(int, confident_boxes[largest_face_idx])

            is_close_to_border = (x1 <= border_thresh or y1 <= border_thresh or 
                                  x2 >= w - border_thresh or y2 >= h - border_thresh)

            if is_close_to_border:
                print("[Info] Face is close to the border. Padding the original image to a square.")
                image_to_process = _pad_to_square(frame, pad_color=255)
            else:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                side = int(max(x2 - x1, y2 - y1) * face_crop_scale)
                half = side // 2

                left, top = max(cx - half, 0), max(cy - half, 0)
                right, bottom = min(cx + half, w), min(cy + half, h)

                cropped_face = frame[top:bottom, left:right]
                image_to_process = _pad_to_square(cropped_face, pad_color=255)

        image_resized = cv2.resize(image_to_process, (resize_to, resize_to), interpolation=cv2.INTER_AREA)

        face_tensor = _img2tensor(image_resized, bgr2rgb=True).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # The BiSeNet model in facexlib was trained on a different normalization
            normalized_face = normalize(face_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            parsing_out = parsing_model(normalized_face)[0]
            parsing_map_tensor = parsing_out.argmax(dim=1, keepdim=True)

        parsing_map_np = parsing_map_tensor.squeeze().cpu().numpy().astype(np.uint8)
        
        # --- UPDATED: MASK SELECTION LOGIC ---
        if face_only_mode:
            # If face_only_mode is ON, create a precise mask for core facial features
            # This will ignore the with_neck setting
            parts_to_exclude = [0, 14, 15, 16, 17, 18] # bg, neck, cloth, hair etc.
            final_mask_np = np.isin(parsing_map_np, parts_to_exclude, invert=True).astype(np.uint8)
        elif with_neck:
            # Standard mode: include neck and hair
            final_mask_np = (parsing_map_np != 0).astype(np.uint8)
        else:
            # Standard mode: exclude neck (original behavior)
            parts_to_exclude = [0, 14, 15, 16, 18]
            final_mask_np = np.isin(parsing_map_np, parts_to_exclude, invert=True).astype(np.uint8)
        # --- END OF MASK SELECTION ---
        
        # --- Create the primary output (masked face on white background) ---
        white_background = np.ones_like(image_resized, dtype=np.uint8) * 255
        mask_3channel = cv2.cvtColor(final_mask_np * 255, cv2.COLOR_GRAY2BGR)
        result_img_bgr = np.where(mask_3channel != 0, image_resized, white_background)
        result_tensor = cv2_img_to_tensor(result_img_bgr)

        # --- Create the secondary RGBA face output ---
        image_resized_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        alpha_channel = final_mask_np * 255
        face_rgba_np = np.dstack((image_resized_rgb, alpha_channel))
        face_rgba_tensor = torch.from_numpy(face_rgba_np.astype(np.float32) / 255.0).unsqueeze(0)

        return (result_tensor, face_rgba_tensor,)

# --- MAPPINGS ---
NODE_CLASS_MAPPINGS = {
    "FaceProcessorLoader": FaceProcessorLoader,
    "ApplyFaceProcessor": ApplyFaceProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceProcessorLoader": "Stand-In Processor Loader",
    "ApplyFaceProcessor": "Apply Stand-In Processor"
}



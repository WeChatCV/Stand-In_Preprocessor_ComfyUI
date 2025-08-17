class FaceOnlyModeSwitch:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "FACE ONLY", 
                    "label_off": "FULL MASK"  
                }),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("face_only_mode",)
    FUNCTION = "toggle"
    CATEGORY = "Stand-In"

    def toggle(self, enabled: bool):
        return (enabled,)


NODE_CLASS_MAPPINGS = {
    "FaceOnlyModeSwitch": FaceOnlyModeSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceOnlyModeSwitch": "Face Only Mode Switch (Stand-In)",
}
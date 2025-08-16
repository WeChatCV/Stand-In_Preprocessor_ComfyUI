import logging
import os

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO), 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

NODE_PACK_NAME = "Stand-In"
logger.info("=" * 40 + f" {NODE_PACK_NAME} " + "=" * 40)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes import FaceProcessorLoader, ApplyFaceProcessor
    from .videoinput import VideoInputPreprocessor
    
    # Map the class names to the classes themselves
    NODE_CLASS_MAPPINGS["FaceProcessorLoader"] = FaceProcessorLoader
    NODE_CLASS_MAPPINGS["ApplyFaceProcessor"] = ApplyFaceProcessor
    NODE_CLASS_MAPPINGS["VideoInputPreprocessor"] = VideoInputPreprocessor

    NODE_DISPLAY_NAME_MAPPINGS["FaceProcessorLoader"] = "Stand-In Processor Loader"
    NODE_DISPLAY_NAME_MAPPINGS["ApplyFaceProcessor"] = "Apply Stand-In Processor"
    NODE_DISPLAY_NAME_MAPPINGS["VideoInputPreprocessor"] = "Stand-In VideoInputPreprocessor"

    logger.info("Successfully loaded all Stand-In nodes.")

except ImportError:
    logger.exception(f"Failed to import nodes for {NODE_PACK_NAME}. Please ensure all dependencies are installed.")
except Exception as e:
    logger.exception(f"An unexpected error occurred while loading nodes for {NODE_PACK_NAME}: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# --- Footer Banner ---
logger.info("=" * (82 + len(NODE_PACK_NAME)))
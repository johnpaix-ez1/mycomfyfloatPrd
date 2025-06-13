import folder_paths
import os
import comfy.model_management as mm
import time
import torchaudio
import torchvision.utils as vutils
import requests # For requests.exceptions.RequestException
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError # For specific Hugging Face errors
import torch # For torch.cuda.empty_cache()
import gc # For gc.collect()
import sys # For the sys.modules check (though we'll use try-except for torch.cuda)

from .generate import InferenceAgent
from .options.base_options import BaseOptionsJson

class LoadFloatModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (['float.pth'],)
            },
        }

    RETURN_TYPES = ("FLOAT_PIPE",)
    RETURN_NAMES = ("float_pipe",)
    FUNCTION = "loadmodel"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Models are auto-downloaded to /ComfyUI/models/float"

    def loadmodel(self, model):
        # download models if not exist
        float_models_dir = os.path.join(folder_paths.models_dir, "float")
        os.makedirs(float_models_dir, exist_ok=True)

        wav2vec2_base_960h_models_dir = os.path.join(float_models_dir,"wav2vec2-base-960h") 
        wav2vec_english_speech_emotion_recognition_models_dir = os.path.join(float_models_dir,"wav2vec-english-speech-emotion-recognition") 
        float_model_path = os.path.join(float_models_dir,"float.pth")

        if not os.path.exists(float_model_path) or not os.path.isdir(wav2vec2_base_960h_models_dir) or not os.path.isdir(wav2vec_english_speech_emotion_recognition_models_dir):
            from huggingface_hub import snapshot_download
            try:
                print(f"ComfyUI-FLOAT: Downloading models from Hugging Face repository 'yuvraj108c/float' to {float_models_dir}...")
                snapshot_download(repo_id="yuvraj108c/float", local_dir=float_models_dir, local_dir_use_symlinks=False)
                print("ComfyUI-FLOAT: Model download complete.")
            except HfHubHTTPError as e:
                print(f"ComfyUI-FLOAT: Error downloading models (HfHubHTTPError): {e}. Please check your internet connection and the repository URL.")
                raise
            except LocalEntryNotFoundError as e:
                print(f"ComfyUI-FLOAT: Error downloading models (LocalEntryNotFoundError): {e}. A file might be missing in the repository or there could be a local caching issue.")
                raise
            except requests.exceptions.RequestException as e:
                print(f"ComfyUI-FLOAT: Error downloading models (RequestException): {e}. This is likely a network issue.")
                raise
            except Exception as e:
                print(f"ComfyUI-FLOAT: An unexpected error occurred during model download: {e}.")
                raise

        # use custom dictionary instead of original parser for arguments
        opt = BaseOptionsJson
        opt.rank, opt.ngpus  = 0,1
        opt.ckpt_path = float_model_path
        opt.pretrained_dir = float_models_dir
        opt.wav2vec_model_path = wav2vec2_base_960h_models_dir
        opt.audio2emotion_path = wav2vec_english_speech_emotion_recognition_models_dir

        try:
            agent = InferenceAgent(opt)
        except Exception as e:
            print(f"ComfyUI-FLOAT: Error initializing the InferenceAgent: {e}. Check model files and configurations.")
            raise

        return (agent,)

class FloatProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "ref_audio": ("AUDIO",),
                "float_pipe": ("FLOAT_PIPE",),
                "a_cfg_scale": ("FLOAT", {"default": 2.0,"min": 1.0, "step": 0.1}),
                "r_cfg_scale": ("FLOAT", {"default": 1.0,"min": 1.0, "step": 0.1}),
                "e_cfg_scale": ("FLOAT", {"default": 1.0,"min": 1.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 25, "step": 1}),
                "emotion": (['none', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], {"default": "none"}),
                "crop": ("BOOLEAN",{"default":False},),
                "seed": ("INT", {"default": 62064758300528, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "floatprocess"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Float Processing"

    def floatprocess(self, ref_image, ref_audio, float_pipe, a_cfg_scale, r_cfg_scale, e_cfg_scale, fps, emotion, crop, seed):
        # save audio
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        
        # Define paths early for the finally block
        audio_save_path = os.path.join(temp_dir, f"{int(time.time())}.wav")
        # Generate a slightly more unique name for the image in case time.time() is the same
        image_save_path = os.path.join(temp_dir, f"{int(time.time())}_img.png")

        try:
            try:
                torchaudio.save(audio_save_path, ref_audio['waveform'].squeeze(0), ref_audio["sample_rate"])
            except Exception as e:
                print(f"ComfyUI-FLOAT: Error saving temporary audio file {audio_save_path}: {e}")
                raise

            # save image
            if ref_image.shape[0] != 1:
                # This check should ideally be at the very beginning of the INPUT_TYPES validation,
                # but keeping it here as per original structure before refactoring for file saving.
                raise Exception("Only a single image is supported.")

            ref_image_bchw = ref_image.permute(0, 3, 1, 2)
            try:
                vutils.save_image(ref_image_bchw[0], image_save_path)
            except Exception as e:
                print(f"ComfyUI-FLOAT: Error saving temporary image file {image_save_path}: {e}")
                raise

            float_pipe.G.to(float_pipe.rank)
            float_pipe.opt.fps = fps

            try:
                images_bhwc = float_pipe.run_inference(
                    None,
                    image_save_path,
                    audio_save_path,
                    a_cfg_scale = a_cfg_scale,
                    r_cfg_scale = r_cfg_scale,
                    e_cfg_scale = e_cfg_scale,
                    emo 		= None if emotion == "none" else emotion,
                    no_crop 	= not crop,
                    seed 		= seed
                )
            except Exception as e:
                print(f"ComfyUI-FLOAT: Error during inference: {e}. Check console for more details from the model.")
                raise

            float_pipe.G.to(mm.unet_offload_device())
            return (images_bhwc,)
        finally:
            if os.path.exists(audio_save_path):
                try:
                    os.remove(audio_save_path)
                except Exception as e: # Keep original specific logging for cleanup
                    print(f"[FLOAT Node] Error deleting temporary audio file {audio_save_path}: {e}")
            if os.path.exists(image_save_path):
                try:
                    os.remove(image_save_path)
                except Exception as e: # Keep original specific logging for cleanup
                    print(f"[FLOAT Node] Error deleting temporary image file {image_save_path}: {e}")

            # Clear CUDA cache and collect garbage
            try:
                if torch.cuda.is_available(): # Check if CUDA is available before trying to empty cache
                    torch.cuda.empty_cache()
                    print("ComfyUI-FLOAT: Cleared CUDA cache.")
            except Exception as e:
                print(f"ComfyUI-FLOAT: Could not clear CUDA cache (this is normal if CUDA is not available/used or if torch is not fully initialized): {e}")

            gc.collect()
            print("ComfyUI-FLOAT: Collected garbage.")
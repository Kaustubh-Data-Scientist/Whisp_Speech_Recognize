import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import os

def save_model_and_processor(model_id, save_directory):
    """Saves the specified model and processor to the given directory."""
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
            low_cpu_mem_usage=True, use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained(model_id)
        
        os.makedirs(save_directory, exist_ok=True)
        model.save_pretrained(save_directory)
        processor.save_pretrained(save_directory)
        print(f"Model and processor saved to: {save_directory}")
        return save_directory
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
        return None

# Example usage
if __name__ == "__main__":
    model_id = "openai/whisper-large-v3-turbo"
    save_directory = "models/whisper-large-v3-turbo"  # Desired save location
    save_model_and_processor(model_id, save_directory)

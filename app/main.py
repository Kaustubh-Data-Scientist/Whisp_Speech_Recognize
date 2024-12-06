import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def load_model_and_processor(model_directory):
    """Loads the model and processor from the specified directory."""
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_directory, torch_dtype=torch_dtype)
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_directory)
        print("Model and processor loaded successfully!")
        return model, processor
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None, None

# Example usage
if __name__ == "__main__":
    model_directory = "models/whisper-large-v3-turbo"  # Path to the saved model
    model, processor = load_model_and_processor(model_directory)

    if model and processor:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if torch.cuda.is_available() else -1,
        )

        # Path to the audio file
        audio_file = r"C:\Users\Ajay\Desktop\Data Science\Project\Whisp_Speech_Recognize\app\resources_sample-calls.mp3"
        result = pipe(audio_file, chunk_length_s=30, stride_length_s=5)
        print("Transcription Result:")
        print(result["text"])

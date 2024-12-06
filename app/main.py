# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# def load_model_and_processor(model_directory):
#     """Loads the model and processor from the specified directory."""
#     try:
#         device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#         model = AutoModelForSpeechSeq2Seq.from_pretrained(model_directory, torch_dtype=torch_dtype)
#         model.to(device)
#         processor = AutoProcessor.from_pretrained(model_directory)
#         print("Model and processor loaded successfully!")
#         return model, processor
#     except Exception as e:
#         print(f"An error occurred while loading the model: {e}")
#         return None, None

# # Example usage
# if __name__ == "__main__":
#     model_directory = "models/whisper-large-v3-turbo"  # Path to the saved model
#     model, processor = load_model_and_processor(model_directory)

#     if model and processor:
#         pipe = pipeline(
#             "automatic-speech-recognition",
#             model=model,
#             tokenizer=processor.tokenizer,
#             feature_extractor=processor.feature_extractor,
#             device=0 if torch.cuda.is_available() else -1,
#         )

#         # Path to the audio file
#         audio_file = r"C:\Users\Ajay\Desktop\Data Science\Project\Whisp_Speech_Recognize\app\resources_sample-calls.mp3"
#         result = pipe(audio_file, chunk_length_s=30, stride_length_s=5)
#         print("Transcription Result:")
#         print(result["text"])

from flask import Flask, request, render_template, redirect, url_for
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = Flask(__name__)

MODEL_DIR = "models/whisper-large-v3-turbo"

# Load ASR model and processor
def load_whisper_model():
    if not os.path.exists(MODEL_DIR):
        print("Downloading Whisper model...")
        from pipeline import save_model_and_processor
        save_model_and_processor("openai/whisper-large-v3-turbo", MODEL_DIR)

    print("Loading Whisper model and processor...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_DIR, torch_dtype=torch_dtype)
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    return model, processor

# Load Whisper model
whisper_model, whisper_processor = load_whisper_model()

# ASR pipeline with Whisper
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    device=0 if torch.cuda.is_available() else -1,
    batch_size=8,
)

# Load NER pipeline
def load_ner_pipeline():
    print("Loading NER model...")
    general_ner_model = "dslim/bert-base-NER"  # General-purpose NER
    medical_ner_model = "blaze999/Medical-NER"  # Medical NER
    return pipeline("ner", model=general_ner_model), pipeline("ner", model=medical_ner_model)

general_ner_pipeline, medical_ner_pipeline = load_ner_pipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return redirect(url_for('index'))

    audio_file = request.files['audio']
    if not audio_file.filename:
        return redirect(url_for('index'))

    filepath = os.path.join('uploads', audio_file.filename)
    os.makedirs('uploads', exist_ok=True)
    audio_file.save(filepath)

    try:
        # Perform transcription
        transcription_result = asr_pipeline(filepath, chunk_length_s=30, stride_length_s=5, generate_kwargs={"language": "english"})
        transcription = transcription_result["text"]

        # General entity extraction
        general_entities = general_ner_pipeline(transcription)
        general_entities = [{"text": ent["word"], "label": ent["entity"]} for ent in general_entities]

        # Medical entity extraction
        medical_entities = medical_ner_pipeline(transcription)
        medical_entities = [{"text": ent["word"], "label": ent["entity"]} for ent in medical_entities]

        # Combine entities
        entities = general_entities + medical_entities

    except Exception as e:
        transcription = f"Error during transcription: {e}"
        entities = []

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return render_template('result.html', transcription=transcription, entities=entities)

if __name__ == "__main__":
    app.run(debug=True)

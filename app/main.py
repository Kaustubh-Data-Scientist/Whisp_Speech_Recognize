from flask import Flask, request, render_template, redirect, url_for
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import os
import torch
import librosa

# Flask app setup
app = Flask(__name__)

# Paths to model and tokenizer
MODEL_PATH = "models/model/"
PROCESSOR_PATH = "models/tokenizer/"  # Use processor instead of tokenizer

# Load the model and processor once during app initialization
processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_PATH)

# Upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Ensure file is uploaded
    if 'audio' not in request.files:
        return redirect(url_for('index'))
    
    audio_file = request.files['audio']
    
    # Save file
    if audio_file and audio_file.filename != '':
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        audio_file.save(filepath)
    else:
        return redirect(url_for('index'))

    try:
        # Load and preprocess audio
        audio, sr = librosa.load(filepath, sr=16000)

        # Convert audio to tensor format expected by the processor
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            logits = model(**inputs).logits

        # Decode the output into text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    except Exception as e:
        # Log the error for debugging purposes
        app.logger.error(f"An error occurred during transcription: {e}")
        transcription = f"An error occurred: {str(e)}"  # Return the error message for debugging

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

    return render_template('result.html', transcription=transcription)

if __name__ == '__main__':
    app.run(debug=True)

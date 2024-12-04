# import torch
# from transformers import AutoModelForCasualLM, AutoTokenizer

# model_name = "mediaProcessing/Transcriber-Medium"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(f"models/tokenizer/{model_name}")

# model = AutoModelForCasualLM.from_pretrained(model_name)
# model.save_pretrained(f"models/model/{model_name}")

# tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizer/{model_name}")
# model = AutoModelForCasualLM.from_pretrained(f"models/model/{model_name}")

from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq
import os

# Define model and tokenizer paths
# model_name = "mediaProcessing/Transcriber-small"
# save_tokenizer = "models/tokenizer/"
# save_model = "models/model/"

# # Create directories if they do not exist
# os.makedirs(save_tokenizer, exist_ok=True)
# os.makedirs(save_model, exist_ok=True)

# # Download and save tokenizer
# print(f"Downloading tokenizer for {model_name}...")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(save_tokenizer)
# print(f"Tokenizer saved at {save_tokenizer}")

# # Download and save model
# print(f"Downloading model for {model_name}...")
# model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
# model.save_pretrained(save_model)
# print(f"Model saved at {save_model}")

from transformers import AutoProcessor

model_name = "mediaProcessing/Transcriber-small"  # Replace with your model name
save_processor = "models/tokenizer/"

# Create directory if not exists
import os
os.makedirs(save_processor, exist_ok=True)

# Download and save the processor
print(f"Downloading processor for {model_name}...")
processor = AutoProcessor.from_pretrained(model_name)
processor.save_pretrained(save_processor)
print(f"Processor saved at {save_processor}")

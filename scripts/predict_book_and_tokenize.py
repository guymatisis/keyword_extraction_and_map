import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
import re
from pathlib import Path
import numpy as np

def init_models():
    model_path = "bloomberg/KeyBART"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

def embed_phrase(phrase, model, tokenizer):
    inputs = tokenizer(phrase, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_encoder()(**inputs)
    token_embeds = outputs.last_hidden_state
    phrase_embed = token_embeds.mean(dim=1)  # average pooling
    return phrase_embed.squeeze().numpy()

def extract_chapter_page(filename):
    match = re.match(r'ch(\d+)_page(\d+)\.txt', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def process_file(filepath, model, tokenizer):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Extract keyphrases using the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_length=64, num_beams=4)
    keyphrases = tokenizer.decode(outputs[0], skip_special_tokens=True)
    keyphrases = keyphrases.split(';')
    
    # Get embeddings for each keyphrase
    embeddings = [embed_phrase(kp.strip(), model, tokenizer) for kp in keyphrases if kp.strip()]
    
    return keyphrases, embeddings

def main():
    # Initialize models
    model, tokenizer = init_models()
    
    # Path to the text files
    base_path = Path("/home/guymat/projects/keyphrase_extraction/data/hands-on_machine_learning_with_scikit-learn_keras_and_tensorflow/page_text")
    
    results = []
    
    # Process all chapter files
    for filename in os.listdir(base_path):
        if filename.startswith('ch') and filename.endswith('.txt'):
            chapter, page = extract_chapter_page(filename)
            print(page)
            if chapter is not None:
                filepath = base_path / filename
                keyphrases, embeddings = process_file(filepath, model, tokenizer)
                
                # Store results
                results.append({
                    'chapter': chapter,
                    'page': page,
                    'keyphrases': keyphrases,
                    'embeddings': [emb.tolist() for emb in embeddings]  # Convert numpy arrays to lists for JSON serialization
                })
    
    # Save results to JSON
    output_path = Path("/home/guymat/projects/keyphrase_extraction/outputs/keyphrases_embeddings.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()


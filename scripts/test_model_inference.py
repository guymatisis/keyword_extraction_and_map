import os
import random
import glob
from transformers import BartForConditionalGeneration, AutoTokenizer

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, "model", 'keybart_finetuned_e0_20250420_1223') 
TEXT_FOLDER = os.path.join(BASE_PATH, "data", "hands-on_machine_learning_with_scikit-learn_keras_and_tensorflow", "page_text")
PAGE_TO_KEYPHRASES = os.path.join(BASE_PATH, "data", "hands-on_machine_learning_with_scikit-learn_keras_and_tensorflow", "processed_inputs", "page_to_keyphrases.json")
NUMBER_OF_PAGES = 1

# random page selection

pages_in_folder = glob.glob(os.path.join(TEXT_FOLDER, "*.txt"))
pages = []
for i in range(NUMBER_OF_PAGES):
    page = random.choice(pages_in_folder)
    page_number = int(page[-5])
    pages_in_folder.remove(page)
    pages.append([page, page_number])

# load model

model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# get page to keyphrases mapping
import json
with open(PAGE_TO_KEYPHRASES, "r") as f:
    page_to_keyphrases = json.load(f)

# infer keywords from text
for page, page_number in pages:
    with open(page, "r") as f:
        input_text = f.read()
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    output_ids = model.generate(**inputs, max_length=64, num_beams=4)
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predicted_phrases = decoded_output.split(';')
    gt_keyphrases = page_to_keyphrases[str(page_number)]

    from evaluate_model import get_bertscore, sbert_soft_f1, exact_f1_at_k

    print('bertscore:', get_bertscore(predicted_phrases, gt_keyphrases))
    print('f1 at 1:', exact_f1_at_k(predicted_phrases, gt_keyphrases, 1))
    print('f1 at 2:', exact_f1_at_k(predicted_phrases, gt_keyphrases, 2))
    print('f1 at 3:', exact_f1_at_k(predicted_phrases, gt_keyphrases, 3))
    print('sbert soft f1:', sbert_soft_f1(predicted_phrases, gt_keyphrases))
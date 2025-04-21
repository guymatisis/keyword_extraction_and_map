import os
import random
import glob
import argparse
from transformers import BartForConditionalGeneration, AutoTokenizer
import pandas as pd
import json
from evaluate_model import get_bertscore, sbert_soft_f1, exact_f1_at_k
import torch

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATION_CSV = os.path.join(BASE_PATH, "data", "hands-on_machine_learning_with_scikit-learn_keras_and_tensorflow", "processed_inputs", "valid.csv")


def main(model_path):
# load model
    cuda_avialable = torch.cuda.is_available
    model = BartForConditionalGeneration.from_pretrained(model_path).eval()
    model = model.to("cuda") if cuda_avialable else model
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # get page to keyphrases mapping
    validation_df = pd.read_csv(VALIDATION_CSV, sep="|")
    validation_df = validation_df[["text", "keyphrases"]]
    validation_df["keyphrases"] = validation_df["keyphrases"].apply(lambda x: x.split(";"))

    # Initialize metrics dictionary
    metrics = {
        'bertscore': [],
        'f1_at_1': [],
        'f1_at_2': [],
        'f1_at_3': [],
        'sbert_soft_f1': []
    }

    # infer keywords from text
    for row in validation_df.itertuples():
        text = row.text
        gt_keyphrases = row.keyphrases
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        inputs = inputs.to("cuda") if cuda_avialable else inputs
        output_ids = model.generate(**inputs, max_length=64, num_beams=4)
        decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicted_phrases = decoded_output.split(';')

        # Calculate and store metrics
        bertscore = get_bertscore(predicted_phrases, gt_keyphrases)
        f1_1 = exact_f1_at_k(predicted_phrases, gt_keyphrases, 1)
        f1_2 = exact_f1_at_k(predicted_phrases, gt_keyphrases, 2)
        f1_3 = exact_f1_at_k(predicted_phrases, gt_keyphrases, 3)
        sbert_f1 = sbert_soft_f1(predicted_phrases, gt_keyphrases)

        metrics['bertscore'].append(bertscore)
        metrics['f1_at_1'].append(f1_1)
        metrics['f1_at_2'].append(f1_2)
        metrics['f1_at_3'].append(f1_3)
        metrics['sbert_soft_f1'].append(sbert_f1)

        # Print ground truth and predicted keywords
        print('Ground Truth Keyphrases:', gt_keyphrases)
        print('Predicted Keyphrases:', predicted_phrases)
        
        # Print individual results
        print('bertscore:', bertscore)
        print('f1 at 1:', f1_1)
        print('f1 at 2:', f1_2)
        print('f1 at 3:', f1_3)
        print('sbert soft f1:', sbert_f1)
        print('-' * 50)

    # Calculate averages
    avg_metrics = {metric: sum(values)/len(values) for metric, values in metrics.items()}

    # Print average results
    print('\nAverage Metrics:')
    print('-' * 50)
    for metric, value in avg_metrics.items():
        print(f'{metric}: {value:.4f}')

    # Save results to file
    results_file = os.path.join(model_path, 'validation_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'average_metrics': avg_metrics,
            'all_metrics': metrics
        }, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test KeyBART model inference on validation data')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model directory containing the model files')
    args = parser.parse_args()
    main(args.model_path)
import os
import random
import glob
import argparse
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import json
from evaluate_model import get_bertscore, accuracy_score, calc_f1_score, compute_rouge_scores
import torch

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV = os.path.join(BASE_PATH, "data", "hands-on_machine_learning_with_scikit-learn_keras_and_tensorflow", "processed_inputs", "test.csv")


def main(model_path):
    # load model
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    is_longformer = "led" in model.config.model_type.lower() or "longformer" in model_path.lower()

    # get page to keyphrases mapping
    validation_df = pd.read_csv(TEST_CSV, sep="|")
    validation_df = validation_df[["text", "keyphrases"]]
    validation_df["keyphrases"] = validation_df["keyphrases"].apply(lambda x: x.split(";"))

    # Initialize metrics dictionary
    metrics = {
        'bertscore': [],
        'accuracy': [],
        'f1': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }

    # infer keywords from text
    for row in validation_df.itertuples():
        text = row.text
        gt_keyphrases = row.keyphrases

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=4096 if is_longformer else 1024,
        )
        if is_longformer:
            inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
            inputs["global_attention_mask"][:, 0] = 1

        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_ids = model.generate(**inputs, max_length=64, num_beams=4)
        decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicted_phrases = decoded_output.split(';')

        # Calculate and store metrics
        bertscore = get_bertscore(predicted_phrases, gt_keyphrases)
        accuracy = accuracy_score(predicted_phrases, gt_keyphrases)
        f1 = calc_f1_score(predicted_phrases, gt_keyphrases)
        rouge1, rouge2, rougeL = compute_rouge_scores(predicted_phrases, gt_keyphrases)

        metrics['bertscore'].append(bertscore)
        metrics['accuracy'].append(accuracy)
        metrics['f1'].append(f1)
        metrics['rouge1'].append(rouge1)
        metrics['rouge2'].append(rouge2)
        metrics['rougeL'].append(rougeL)

        # Print ground truth and predicted keywords
        print('Ground Truth Keyphrases:', gt_keyphrases)
        print('Predicted Keyphrases:', predicted_phrases)
        
        # Print individual results
        print('BERTScore:', bertscore)
        print('Accuracy:', accuracy)
        print('F1 Score:', f1)
        print('ROUGE-1:', rouge1)
        print('ROUGE-2:', rouge2)
        print('ROUGE-L:', rougeL)
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
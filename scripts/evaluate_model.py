from typing import List, Tuple
import re
import nltk
import logging
import warnings
from sentence_transformers import SentenceTransformer, util
import evaluate
import transformers
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import PreTrainedTokenizerBase


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mute model loading messages
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)




# ------------------------------------------------------------------------------
# 1. Rouge Score 1,2,L
# ------------------------------------------------------------------------------

_rouge_metric = evaluate.load("rouge")

def compute_rouge_scores(predicted_phrases, ground_truth_phrases):
    """
    predicted_phrases: List[str] → list of predicted keyphrases (joined by ';' or just space)
    ground_truth_phrases: List[str] → list of reference keyphrases (same format)

    Returns:
        dict with rouge1, rouge2, rougeL (F1 scores)
    """

    # Convert keyphrases to text blocks if they're lists of words
    pred_text = "; ".join([p.strip() for p in predicted_phrases if p.strip()])
    ref_text = "; ".join([r.strip() for r in ground_truth_phrases if r.strip()])

    if not pred_text or not ref_text:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    result = _rouge_metric.compute(predictions=[pred_text], references=[ref_text])
    
    return result["rouge1"], result["rouge2"], result["rougeL"]


# ------------------------------------------------------------------------------
# 2. Precision / Recall / F1 for top-k predicted phrases
# ------------------------------------------------------------------------------

def calc_f1_score(predicted, ground_truth):
    # Normalize (lowercase and strip)
    predicted_set = set([k.strip().lower() for k in predicted])
    ground_truth_set = set([k.strip().lower() for k in ground_truth])

    # True positives: exact matches
    tp = len(predicted_set & ground_truth_set)
    fp = len(predicted_set - ground_truth_set)
    fn = len(ground_truth_set - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return f1


# ------------------------------------------------------------------------------
# 3. Semantic similarity using BERTScore
# ------------------------------------------------------------------------------
_bertscore_metric = evaluate.load("bertscore")

def get_bertscore(predicted_phrases: List[str], correct_phrases: List[str]) -> float:
    logger.info("Computing BERTScore semantic similarity")
    max_len = max(len(predicted_phrases), len(correct_phrases))
    padded_preds = predicted_phrases + [""] * (max_len - len(predicted_phrases))
    padded_refs  = correct_phrases + [""] * (max_len - len(correct_phrases))

    result = _bertscore_metric.compute(predictions=padded_preds, references=padded_refs, lang="en")
    score = sum(result["f1"]) / len(result["f1"])
    logger.info(f"BERTScore F1: {score:.3f}")
    return score


# ------------------------------------------------------------------------------
# 4. accuracy
# ------------------------------------------------------------------------------
def accuracy_score(predictions, ground_truths):
    """
    predictions: List of lists of predicted keyphrases
    ground_truths: List of lists of ground truth keyphrases

    Returns: float accuracy (0.0 to 1.0)
    """

    pred_set = set([k.strip().lower() for k in predictions])
    gt_set = set([k.strip().lower() for k in ground_truths])
    correct_num = len(pred_set & gt_set)
    total_num = len(pred_set)


    return correct_num / total_num if total_num > 0 else 0.0



# ------------------------------------------------------------------------------
# 5. call back fuction for model evaluation
# ------------------------------------------------------------------------------

def sanitize_preds(preds, pad_token_id):
    return [
        [token if token != -100 else pad_token_id for token in pred]
        for pred in preds
    ]

def make_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        try:
            preds, labels = eval_preds

            preds = sanitize_preds(preds, tokenizer.pad_token_id)
            decoded_output = tokenizer.decode(preds[0], skip_special_tokens=True)
            predicted_phrases = decoded_output.split(';')

            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            gt_labels = decoded_labels[0].split(';')

            bertscore = get_bertscore(predicted_phrases, gt_labels)
            accuracy_score_value = accuracy_score(predicted_phrases, gt_labels)
            f1_score = calc_f1_score(predicted_phrases, gt_labels)
            rouge_1, rouge_2, rouge_L = compute_rouge_scores(predicted_phrases, gt_labels)
            

            return {
                "f1": f1_score,
                "bertscore": bertscore,
                "rouge1": rouge_1,
                "rouge2": rouge_2,
                "rougeL": rouge_L,
                "accuracy": accuracy_score_value
            }
        except Exception as e:
            print(f"Error in compute_metrics: {e}")
            return {
                "f1": float("nan"),
                "bertscore": float("nan"),
                "rouge1": float("nan"),
                "rouge2": float("nan"),
                "rougeL": float("nan"),
                "accuracy": float("nan")
            }
    
    return compute_metrics

if __name__ == "__main__":
    gt_labels = ["neural network", "gradient descent", "back-propagation"]
    predicted_phrases = ["Neural network", "backpropagation", "deep learning"]

    bertscore = get_bertscore(predicted_phrases, gt_labels)
    accuracy_score_value = accuracy_score(predicted_phrases, gt_labels)
    f1_score = calc_f1_score(predicted_phrases, gt_labels)
    rouge_1, rouge_2, rouge_L = compute_rouge_scores(predicted_phrases, gt_labels)

    print(f"BERTScore: {bertscore}")
    print(f"Accuracy: {accuracy_score_value}")
    print(f"F1 Score: {f1_score}")
    print(f"Rouge-1: {rouge_1}")
    print(f"Rouge-2: {rouge_2}")
    print(f"Rouge-L: {rouge_L}")

# easy_metrics.py
#
# A simple set of keyword evaluation metrics:
# - Precision, Recall, F1@k
# - BERTScore for semantic similarity
# - SBERT-based fuzzy matching (soft F1)

from typing import List, Tuple
import re
import nltk
import logging
import warnings
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, util
import evaluate
import transformers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mute model loading messages
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)

# Download the NLTK stemmer data if not already installed
nltk.download("punkt", quiet=True)
stemmer = PorterStemmer()

# Used to find word-like pieces in phrases (ignores punctuation)
WORD_PATTERN = re.compile(r"\b\w+\b")


# ------------------------------------------------------------------------------
# 1. Clean up a phrase for easier matching (lowercase, basic stemming)
# ------------------------------------------------------------------------------
def normalize(phrase: str) -> str:
    words = WORD_PATTERN.findall(phrase.lower())
    stems = [stemmer.stem(w) for w in words]
    return " ".join(stems)


# ------------------------------------------------------------------------------
# 2. Precision / Recall / F1 for top-k predicted phrases
# ------------------------------------------------------------------------------
def exact_f1_at_k(predicted_phrases: List[str], correct_phrases: List[str], k: int) -> Tuple[float, float, float]:
    logger.info(f"Computing exact F1@{k} score")
    predicted = predicted_phrases[:k]
    gold_set = {normalize(p) for p in correct_phrases}
    matched_gold = set()

    true_positives = 0
    for phrase in predicted:
        cleaned = normalize(phrase)
        if cleaned in gold_set and cleaned not in matched_gold:
            true_positives += 1
            matched_gold.add(cleaned)

    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(correct_phrases) if correct_phrases else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    logger.info(f"F1@{k} results - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    return precision, recall, f1


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
# 4. Fuzzy matching using sentence embeddings (SBERT)
# ------------------------------------------------------------------------------
_sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def sbert_soft_f1(predicted_phrases: List[str], correct_phrases: List[str], similarity_threshold: float = 0.8) -> float:
    logger.info(f"Computing SBERT soft F1 score (threshold: {similarity_threshold})")
    if not predicted_phrases or not correct_phrases:
        logger.info("Empty input - returning 0.0")
        return 0.0

    pred_embeddings = _sbert_model.encode(predicted_phrases, convert_to_tensor=True)
    ref_embeddings = _sbert_model.encode(correct_phrases, convert_to_tensor=True)
    similarity_scores = util.cos_sim(pred_embeddings, ref_embeddings)

    matched_refs = set()
    true_positives = 0
    for i in range(len(predicted_phrases)):
        best_match_index = int(similarity_scores[i].argmax())
        if similarity_scores[i][best_match_index] >= similarity_threshold and best_match_index not in matched_refs:
            true_positives += 1
            matched_refs.add(best_match_index)

    precision = true_positives / len(predicted_phrases)
    recall = true_positives / len(correct_phrases)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    logger.info(f"SBERT Soft F1 score: {f1:.3f}")
    return f1

def make_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        decoded_output = tokenizer.decode(preds[0], skip_special_tokens=True)
        predicted_phrases = decoded_output.split(';')

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        gt_labels = decoded_labels[0].split(';')

        # aggregate metrics
        f1_at_5 = []
        f1_at_o = []
        sbert_f1s = []
        bertscores = []

        f1_at_5 = exact_f1_at_k(predicted_phrases, gt_labels, 5)
        f1_at_1 = exact_f1_at_k(predicted_phrases, gt_labels, 1)
        sbert_f1 = sbert_soft_f1(predicted_phrases, gt_labels)
        bertscore = get_bertscore(predicted_phrases, gt_labels)

        return {
            "f1@5": f1_at_5,
            "f1@1": f1_at_1,
            "sbert_f1": sbert_f1,
            "bertscore": bertscore,
        }
    return compute_metrics

# ------------------------------------------------------------------------------
# 5. Run this file directly to test
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    ground_truth = ["neural network", "gradient descent", "back-propagation"]
    predicted = ["Neural networks", "backpropagation", "deep learning"]

    for k in (1, 2, 3):
        p, r, f = exact_f1_at_k(predicted, ground_truth, k)
        print(f"F1@{k}: {f:.3f}")

    print("BERTScore F1:", get_bertscore(predicted, ground_truth))
    print("SBERT Soft F1:", sbert_soft_f1(predicted, ground_truth))

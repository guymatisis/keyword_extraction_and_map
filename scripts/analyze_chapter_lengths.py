import os
import glob
import numpy as np

import tiktoken

# Load tokenizer (e.g., for GPT-3.5 or GPT-4)
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Count tokens
    num_tokens = len(enc.encode(text))
    return num_tokens

def analyze_lengths(directory):
    """Analyzes the lengths of text files in a directory."""
    file_pattern = os.path.join(directory, 'ch*_page*.txt')
    file_paths = glob.glob(file_pattern)

    word_counts = [count_tokens(file_path) for file_path in file_paths]

    if not word_counts:
        print("No files found in the specified directory.")
        return

    word_counts = np.array(word_counts)

    print("Statistics for file lengths (in words):")
    print(f"Total files analyzed: {len(word_counts)}")
    print(f"Max length: {np.max(word_counts)} words")
    print(f"Min length: {np.min(word_counts)} words")
    print(f"Mean length: {np.mean(word_counts):.2f} words")
    print(f"Median length: {np.median(word_counts)} words")
    print(f"25th percentile: {np.percentile(word_counts, 25)} words")
    print(f"75th percentile: {np.percentile(word_counts, 75)} words")

def main():
    directory = os.path.join(os.path.dirname(__file__), '../data/self_extracted_chapters')
    analyze_lengths(directory)

if __name__ == "__main__":
    main()

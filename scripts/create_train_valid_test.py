import os
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split

# CSV layout:  text  ,  keyphrases
# keyphrases column should be a single string:  "deep learning; model compression; pruning"

# Define input and output paths
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, 'data', 'natural-language-processing-with-transformers-revised-edition','page_text','processed_inputs')
output_dir = data_dir

page_to_keyphrases_file = os.path.join(data_dir, 'page_to_keyphrases.json')
self_extracted_chapters_dir = os.path.dirname(data_dir)

# Load page-to-keyphrase mapping
with open(page_to_keyphrases_file, 'r', encoding='utf-8') as f:
    page_to_keyphrases = json.load(f)

# Helper: map page number to (chapter, page) using filenames
page_to_chapter_page = {}
for fname in os.listdir(self_extracted_chapters_dir):
    m = re.match(r'page_(\d+)\.txt', fname)
    if m:
        page = int(m.group(1))
        page_to_chapter_page[str(page)] = fname


train_pages, valid_pages = {}, {}
for page, keyphrases in page_to_keyphrases.items():
    info = page_to_chapter_page.get(page)
    fname = info
    train_pages[page] = (keyphrases, fname)

# Prepare dataframes for train, valid, and test
def prepare_dataframe(pages):
    data = []
    for page, (keyphrases, fname) in pages.items():
        if not fname: continue
        page_file = os.path.join(self_extracted_chapters_dir, fname)
        if os.path.exists(page_file):
            with open(page_file, 'r', encoding='utf-8') as f:
                text = f.read()
            keyphrases_str = '; '.join(keyphrases)
            data.append({'text': text, 'keyphrases': keyphrases_str})
    return pd.DataFrame(data)

train_df = prepare_dataframe(train_pages)

# Split train_df into train and valid
train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Save dataframes to CSV
os.makedirs(output_dir, exist_ok=True)
train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, sep='|')
valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False, sep='|')




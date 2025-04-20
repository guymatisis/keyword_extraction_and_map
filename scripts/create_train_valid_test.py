import os
import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split

# CSV layout:  text  ,  keyphrases
# keyphrases column should be a single string:  "deep learning; model compression; pruning"

# Define input and output paths
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, 'data')
output_dir = os.path.join(base_dir, 'data', 'input_csvs')

page_to_keyphrases_file = os.path.join(data_dir, 'page_to_keyphrases.json')
self_extracted_chapters_dir = os.path.join(data_dir, 'self_extracted_chapters')

# Load page-to-keyphrase mapping
with open(page_to_keyphrases_file, 'r', encoding='utf-8') as f:
    page_to_keyphrases = json.load(f)

# Helper: map page number to (chapter, page) using filenames
page_to_chapter_page = {}
for fname in os.listdir(self_extracted_chapters_dir):
    m = re.match(r'ch(\d+)_page(\d+)\.txt', fname)
    if m:
        chapter, page = int(m.group(1)), int(m.group(2))
        page_to_chapter_page[str(page)] = (chapter, page, fname)

# Split pages by chapter
train_chapters = set([1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19])
test_chapters = set([6, 10, 11, 12])

train_pages, test_pages, valid_pages = {}, {}, {}
for page, keyphrases in page_to_keyphrases.items():
    info = page_to_chapter_page.get(page)
    if not info:
        continue
    chapter, _, fname = info
    if chapter in train_chapters:
        train_pages[page] = (keyphrases, fname)
    elif chapter in test_chapters:
        test_pages[page] = (keyphrases, fname)
    # Optionally, add a valid split here if needed

# Prepare dataframes for train, valid, and test
def prepare_dataframe(pages):
    data = []
    for page, (keyphrases, fname) in pages.items():
        page_file = os.path.join(self_extracted_chapters_dir, fname)
        if os.path.exists(page_file):
            with open(page_file, 'r', encoding='utf-8') as f:
                text = f.read()
            keyphrases_str = '; '.join(keyphrases)
            data.append({'text': text, 'keyphrases': keyphrases_str})
    return pd.DataFrame(data)

train_df = prepare_dataframe(train_pages)
test_df = prepare_dataframe(test_pages)

# Split train_df into train and valid
train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Save dataframes to CSV
os.makedirs(output_dir, exist_ok=True)
train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, sep='|')
valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False, sep='|')
test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False, sep='|')




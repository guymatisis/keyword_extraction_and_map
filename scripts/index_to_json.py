import re
import json
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
input_file = os.path.join(base_dir, 'data','index.txt')
output_file = os.path.join(base_dir, 'data', 'page_to_keyphrases.json')


page_dict = {}
parent_phrase = ""

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()
        if not line:
            continue

        is_indented = line.startswith('\t') or line.startswith(' ')

        # Extract page numbers (including ranges)
        numbers = re.findall(r'\d+', line)

        # Strip off numbers from the end, but keep the phrase
        match = re.search(r'^(.*?)(?:,\s*(\d[\d,\s-]*))?$', line)
        if match:
            phrase_part = match.group(1).strip().rstrip(',')
            number_part = match.group(2) or ''
            numbers = re.findall(r'\d+', number_part)
        else:
            continue  # skip malformed lines

        if is_indented:
            if not parent_phrase:
                continue  # skip if there's no parent yet
            phrase = f"{parent_phrase}; {phrase_part}"
        else:
            parent_phrase = phrase_part
            phrase = parent_phrase

        for number in numbers:
            page = int(number)
            page_dict.setdefault(page, []).append(phrase)

page_dict = {k: list(set(v)) for k, v in page_dict.items()} 
page_dict = dict(sorted(page_dict.items()))
# Save to JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(page_dict, f, indent=2)

print(f"Saved to {output_file}")


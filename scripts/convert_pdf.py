import fitz
import os 
import pdfplumber
import re

base_dir = os.path.dirname(os.path.dirname(__file__))
pdf_path = os.path.join(base_dir, 'data', 'book.pdf')
output_path = os.path.join(base_dir, 'data', 'self_extracted_chapters')

chapter_pages = [30, 64, 114, 140, 182, 204, 218, 242, (264, 305), 308, 360, 404, 442, 474, 526, 554, 596, 638, (696, 747)]
offset = 29
pdf = pdfplumber.open("data/book.pdf")

# for chapter, pages in enumerate(chapter_pages, start=1):
#     if type(pages) == tuple:
#         start_page, end_page = pages
#     else:
#         start_page = pages
#         end_page = chapter_pages[chapter]
#         if type(end_page) == tuple:
#             end_page = end_page[0]

        
#     for page_num in range(start_page, end_page):
#         with open(os.path.join(output_path,f'ch{chapter}_page{page_num- offset}.txt' ), 'w', encoding='utf-8') as f:

#             page = pdf.pages[page_num]
#             cropped = page.crop((0, 0, page.width, page.height - 50))
#             text = cropped.extract_text(layout=True)
#             f.write(text)

# INDEX

# def normalize_indentation(text):
#     return re.sub(r'^\s{6,}', '', text, flags=re.MULTILINE)  # Remove lines starting with 6+ spaces


# with open(os.path.join(output_path,f'index.txt' ), 'w', encoding='utf-8') as f:
    
#     for page_num in range(830,849):
#         page = pdf.pages[page_num]
#         cropped = page.crop((0, 0, page.width, page.height - 50))

#         mid_x = page.width / 2

#         # Left column
#         left = cropped.crop((0, 0, mid_x, cropped.height))
#         left_text = left.extract_text(x_tolerance=1, layout=True)

#         # Right column
#         right = cropped.crop((mid_x, 0, cropped.width, cropped.height))
#         right_text = right.extract_text(x_tolerance=1, layout=True)

#         # Combine as needed
#         text = left_text + '\n' + right_text
#         cleaned_text = normalize_indentation(text)

#         f.write(cleaned_text)


def merge_wrapped_page_numbers(lines):
    merged = []
    for line in lines:
        if merged and all(c.isdigit() or c in {'-', ',', ' '} for c in line.strip()):
            # Append to previous line (after a space)
            merged[-1] += ' ' + line.strip()
        else:
            merged.append(line)
    return merged


def extract_column_text(words):
    lines = {}
    min_x = min(w['x0'] for w in words)  # left edge of column

    for word in words:
        y = round(word['top'])
        lines.setdefault(y, []).append(word)

    result_lines = []
    for y in sorted(lines.keys()):
        words_in_line = sorted(lines[y], key=lambda w: w['x0'])
        first_x = words_in_line[0]['x0']
        
        # Tab level relative to leftmost x in column
        tab_level = int((first_x - min_x)) > 0
        prefix = '\t' if tab_level else ''

        line = ' '.join(w['text'] for w in words_in_line)
        result_lines.append(f"{prefix}{line}")
    result_lines = merge_wrapped_page_numbers(result_lines)
    return '\n'.join(result_lines)

def process_page(page):
    mid_x = page.width / 2

    # Crop left and right columns
    cropped = page.crop((0, 0, page.width, page.height - 50))
    left = cropped.crop((0, 0, mid_x, cropped.height))
    right = cropped.crop((mid_x, 0, page.width, cropped.height))

    # Extract words from each column
    left_words = left.extract_words()
    right_words = right.extract_words()

    # Convert to text
    left_text = extract_column_text(left_words)
    right_text = extract_column_text(right_words)

    return left_text + '\n' + right_text

output= []
for page_num in range(830,849):
    page = pdf.pages[page_num]
    page_text = process_page(page)
    output.append(page_text)

final_text = '\n'.join(output)

# Save to file
with open(os.path.join(output_path,'index.txt'), 'w', encoding='utf-8') as f:
    f.write(final_text)


pdf.close()


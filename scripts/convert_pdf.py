import os 
import pdfplumber
import re
from collections import defaultdict
import statistics
import argparse


from collections import defaultdict

def group_chars_to_words_with_metadata(chars, x_tolerance=2.0, y_tolerance=2.0):
    """
    Groups low-level PDF characters into word-level dictionaries with metadata.
    """
    lines = defaultdict(list)

    # Group characters by approximate vertical line position
    for char in chars:
        line_key = round(char["top"] / y_tolerance)
        lines[line_key].append(char)

    grouped_words = []

    for line_chars in lines.values():
        line_chars = sorted(line_chars, key=lambda c: c["x0"])

        current_word = ""
        word_chars = []
        prev_x1 = None

        for char in line_chars:
            if prev_x1 is not None and (char["x0"] - prev_x1) > x_tolerance:
                # End of word
                if word_chars:
                    word_dict = {
                        "text": current_word.strip(),
                        "x0": word_chars[0]["x0"],
                        "x1": word_chars[-1]["x1"],
                        "top": min(c["top"] for c in word_chars),
                        "bottom": max(c["bottom"] for c in word_chars),
                        "fontname": word_chars[0]["fontname"],  # could be mixed!
                        "size": word_chars[0]["size"],
                    }
                    grouped_words.append(word_dict)

                # Start new word
                current_word = char["text"]
                word_chars = [char]
            else:
                current_word += char["text"]
                word_chars.append(char)

            prev_x1 = char["x1"]

        # Final word in line
        if word_chars:
            word_dict = {
                "text": current_word.strip(),
                "x0": word_chars[0]["x0"],
                "x1": word_chars[-1]["x1"],
                "top": min(c["top"] for c in word_chars),
                "bottom": max(c["bottom"] for c in word_chars),
                "fontname": word_chars[0]["fontname"],
                "size": word_chars[0]["size"],
            }
            grouped_words.append(word_dict)

    return grouped_words

def add_special_characters(lines, words, median_font_size=None):
    """Add special characters for formatting with optional median_font_size parameter."""
    formatted_text = ""
    if len(words) == 0:
        return '\n'

    for line in lines:
        line_fonts = [w["fontname"].lower() for w in line]
        line_sizes = [w["size"] for w in line]
        is_bold = any("bold" in f for f in line_fonts)
        is_big = max(line_sizes) >= median_font_size * 1.3

        line_text = ""
        for word in line:
            text = word["text"]
            font = word["fontname"].lower()

            if "bold" in font:
                text = f"<b>{text}</b>"
            if "italic" in font or "oblique" in font or '-it' in font:
                text = f"<i>{text}</i>"
            if word["size"] >= median_font_size * 1.3:
                text = f"<largefont>{text}</largefont>"

            line_text += text + " "


        line_text = line_text.strip()

        if is_bold and is_big:
            line_text = f"<header>{line_text}</header>"

        formatted_text += line_text + "\n"

    return formatted_text.strip()

def group_words_into_lines(words, line_tolerance=3):
    # Sort words top-to-bottom, left-to-right
    words = sorted(words, key=lambda w: (round(w['top'] / line_tolerance), w['x0']))
    lines = defaultdict(list)

    for word in words:
        line_key = round(word['top'] / line_tolerance)
        lines[line_key].append(word)

    # Return lines as lists of word dicts
    return [lines[k] for k in sorted(lines)]

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

def get_all_words_from_pdf(pdf, chapter_pages):
    """Gather all words from all pages to calculate global statistics."""
    all_words = []
    for pages in chapter_pages:
        if isinstance(pages, tuple):
            start_page, end_page = pages
        else:
            start_page = pages
            next_idx = chapter_pages.index(pages) + 1
            if next_idx < len(chapter_pages):
                end_page = chapter_pages[next_idx]
                if isinstance(end_page, tuple):
                    end_page = end_page[0]
            else:
                continue  # Skip the last chapter as we don't know its end

        for page_num in range(start_page, end_page):
            page = pdf.pages[page_num]
            cropped = page.crop((0, 0, page.width, page.height - 50))
            chars = cropped.extract_words(use_text_flow=True, extra_attrs=[
                "fontname", "size", "x0", "x1", "top", "bottom", "doctop"
            ])
            words = group_chars_to_words_with_metadata(chars)
            all_words.extend(words)
    
    return all_words

def process_chapters(pdf, chapter_pages, output_path, median_font_size, page_offset=29):
    """Process chapters from the PDF and save them as individual text files."""
    for chapter, pages in enumerate(chapter_pages, start=1):
        if isinstance(pages, tuple):
            start_page, end_page = pages
        else:
            start_page = pages
            end_page = chapter_pages[chapter]
            if isinstance(end_page, tuple):
                end_page = end_page[0]
        
        for page_num in range(start_page, end_page):
            output_file = os.path.join(output_path, f'ch{chapter}_page{page_num - page_offset}.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                page = pdf.pages[page_num]
                cropped = page.crop((0, 0, page.width, page.height - 50))
                chars = cropped.extract_words(use_text_flow=True,extra_attrs=[
                    "fontname", "size", "x0", "x1", "top", "bottom", "doctop"
                ])
            
                words = group_chars_to_words_with_metadata(chars)
                lines = group_words_into_lines(words)
                formatted_text = add_special_characters(lines, words, median_font_size)  
                f.write(formatted_text)

def process_index(pdf, start_page, end_page, output_path):
    """Process the index pages from the PDF and save them to a single file."""
    output = []
    for page_num in range(start_page, end_page):
        page = pdf.pages[page_num]
        mid_x = page.width / 2

        # Crop left and right columns
        cropped = page.crop((0, 0, page.width, page.height - 50))
        left = cropped.crop((0, 0, mid_x, cropped.height))
        right = cropped.crop((mid_x, 0, page.width, cropped.height))

        # Extract words from each column using text flow
        left_chars = left.extract_words(use_text_flow=True, extra_attrs=[
            "fontname", "size", "x0", "x1", "top", "bottom", "doctop"
        ])
        right_chars = right.extract_words(use_text_flow=True, extra_attrs=[
            "fontname", "size", "x0", "x1", "top", "bottom", "doctop"
        ])

        # Convert to words with metadata
        left_words = group_chars_to_words_with_metadata(left_chars)
        right_words = group_chars_to_words_with_metadata(right_chars)

        # Convert to text
        left_text = extract_column_text(left_words)
        right_text = extract_column_text(right_words)

        # Process the text
        column_text = left_text + '\n' + right_text
        
        # Filter lines:
        # 1. Not indented (doesn't start with tab)
        # 2. Has numbers at the end
        filtered_lines = []
        for line in column_text.split('\n'):
            if not line.startswith('\t') and re.search(r'\d+\s*$', line.strip()):
                filtered_lines.append(line)

        if filtered_lines:
            output.append('\n'.join(filtered_lines))

    final_text = '\n'.join(output)
    
    # Save to file
    with open(os.path.join(output_path, 'index.txt'), 'w', encoding='utf-8') as f:
        f.write(final_text)

def process_pdf_without_chapters(pdf,start_page, end_page, output_path, median_font_size, page_offset=0):
    """Process the entire PDF without chapter divisions."""
    for page_num in range(start_page-1, end_page):
        output_file = os.path.join(output_path, f'page_{page_num-page_offset:04d}.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            page = pdf.pages[page_num]
            cropped = page.crop((0, 0, page.width, page.height - 50))
            chars = cropped.extract_words(use_text_flow=True, extra_attrs=[
                "fontname", "size", "x0", "x1", "top", "bottom", "doctop"
            ])
            words = group_chars_to_words_with_metadata(chars)
            lines = group_words_into_lines(words)
            formatted_text = add_special_characters(lines, words, median_font_size)
            f.write(formatted_text)

def get_all_words_sequential(pdf, start_page=0, end_page=None):
    """Gather all words from a range of pages to calculate global statistics."""
    if end_page is None:
        end_page = len(pdf.pages)
    
    all_words = []
    for page_num in range(start_page, end_page):
        page = pdf.pages[page_num]
        cropped = page.crop((0, 0, page.width, page.height - 50))
        chars = cropped.extract_words(use_text_flow=True, extra_attrs=[
            "fontname", "size", "x0", "x1", "top", "bottom", "doctop"
        ])
        words = group_chars_to_words_with_metadata(chars)
        all_words.extend(words)
    
    return all_words

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Convert PDF to formatted text files with special character markup.')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file to process')
    parser.add_argument('--page-offset', type=int, default=0, 
                    help='Page offset for chapter numbering (default: 0)')
    parser.add_argument('--no-chapters', action='store_true',
                    help='Process PDF sequentially without chapter divisions')
    parser.add_argument('--start-page', type=int, default=0,
                    help='First page to process (0-based, default: 0)')
    parser.add_argument('--end-page', type=int,
                    help='Last page to process (0-based, default: last page)')
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        raise FileNotFoundError(f"PDF file not found: {args.pdf_path}")

    # Setup paths
    base_dir = os.path.dirname(args.pdf_path)
    pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
    output_path = os.path.join(base_dir, 'page_text')
    os.makedirs(output_path, exist_ok=True)

    # Define chapter pages for structured mode
    handson = [30, 64, 114, 140, 182, 204, 218, 242, (264, 305), 308, 
               360, 404, 442, 474, 526, 554, 596, 638, (696, 747)]
    chapter_pages = handson

    # Open PDF
    pdf = pdfplumber.open(args.pdf_path)
    
    try:
        # Get all words to calculate global median font size
        # print("Calculating global median font size...")
        # if args.no_chapters:
        #     all_words = get_all_words_sequential(pdf, args.start_page, args.end_page)
        # else:
        #     all_words = get_all_words_from_pdf(pdf, chapter_pages)
        
        # median_font_size = statistics.median([w["size"] for w in all_words])
        # print(f"Global median font size: {median_font_size}")
        median_font_size = 10.5
        # Process PDF based on mode
        # if args.no_chapters:
        #     process_pdf_without_chapters(pdf,args.start_page, args.end_page, output_path, median_font_size, args.page_offset)
        # else:
        #     process_chapters(pdf, chapter_pages, output_path, median_font_size, args.page_offset)
        process_index(pdf, 399  , 406, output_path)
        
        print(f"Processing complete. Output files saved to: {output_path}")
    
    finally:
        # Always close the PDF
        pdf.close()

if __name__ == "__main__":
    main()


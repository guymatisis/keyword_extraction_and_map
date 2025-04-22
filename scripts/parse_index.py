#!/usr/bin/env python3
"""
parse_index.py

Parses an indented book index (plain text) into a JSON mapping of
keyphrase -> list of pages, using heuristics to decide which entries
are standalone keyphrases and which should merge into their parent.
"""

import argparse
import json
import re
import sys
import os
import pathlib

# Generic single-word descriptors to merge into parent, not standalone
BLACKLIST = {"about", "definition", "examples", "see", 'overview','variants', 
             "also", "section", "entry", 'of', 'in', 'defined', 'for', 'example'}

def is_standalone(phrase: str) -> bool:
    """
    Decide if a phrase is a standalone keyphrase:
      1. Multi-word (>=2 words)
      2. Internal uppercase (acronyms or CamelCase)
      3. Contains hyphen, slash, parenthesis, or digits
    And not in the generic blacklist.
    """
    lower = phrase.lower()
    if any([w in BLACKLIST for w in phrase.split()]):
        return False
    # 1. Multi-word
    if len(phrase.split()) > 2:
        return True
    # 2. Internal uppercase (beyond first char)
    if any(c.isupper() for c in phrase[1:]):
        return True
    # 3. Special chars or digits
    if re.search(r'[-/()0-9]', phrase):
        return True
    return False

def parse_index(lines):
    """
    Parse lines of an indented index into a dict {phrase: [pages]}.
    """
    entries = {}
    parent = None

    for raw in lines:
        # Count leading spaces/tabs to get indent level
        indent = len(raw) - len(raw.lstrip(' \t'))
        line = raw.strip()
        if not line:
            continue

        # Split into phrase and pages at first comma
        if ',' in line:
            phrase, pages_str = line.split(',', 1)
            pages = [p.strip() for p in pages_str.split(',') if p.strip()]
        else:
            phrase = line
            pages = []

        phrase = phrase.strip()

        if indent == 0:
            # Top-level entry
            if pages:
                entries[phrase] = pages
                parent = None
            else:
                # Parent stub without pages
                entries.setdefault(phrase, [])
                parent = phrase
        else:
            # Indented entry
            if pages:
                if is_standalone(phrase):
                    # Standalone child becomes its own keyphrase
                    entries[phrase] = pages
                else:
                    # Generic descriptor merges into parent
                    if parent:
                        entries[parent].extend(pages)
            # if no pages, ignore

    return entries

def invert_mapping(phrase_to_pages):
    """
    Inverts a mapping of {phrase: [pages]} to {page: [phrases]}.
    Handles both single page numbers and page ranges (e.g., "130-134").
    """
    page_to_phrases = {}
    
    def add_page(page_num, phrase):
        if page_num not in page_to_phrases:
            page_to_phrases[page_num] = []
        if phrase not in page_to_phrases[page_num]:
            page_to_phrases[page_num].append(phrase)
    
    for phrase, pages in phrase_to_pages.items():
        for page in pages:
            # Handle page ranges (e.g., "130-134")
            if '-' in page:
                try:
                    start, end = map(int, page.split('-'))
                    for page_num in range(start, end + 1):
                        add_page(page_num, phrase)
                except ValueError:
                    continue  # Skip if range is invalid
            else:
                # Handle single pages
                try:
                    page_num = int(page)
                    add_page(page_num, phrase)
                except ValueError:
                    continue  # Skip if page isn't a valid number
    
    return page_to_phrases

# Preprocessing snippet to merge lines ending with '-' (hyphenated splits)

import re

def merge_hyphenated_lines(raw_lines):
    """
    Combines lines where a line ends with a hyphen '-' (or en/em dash),
    indicating the word or phrase is split across lines.
    """
    merged_lines = []
    buffer = None

    for line in raw_lines:
        # Remove only newline, keep other whitespace for detecting indent
        line_stripped_nl = line.rstrip('\n')
        # Strip trailing whitespace to detect hyphens cleanly
        stripped = line_stripped_nl.rstrip()
        
        # Check for hyphen, en‑dash, or em‑dash at end
        if re.search(r'[-\u2010-\u2015]\s*$', stripped):
            # Remove the dash and any trailing spaces
            prefix = re.sub(r'[-\u2010-\u2015]\s*$', '', stripped)
            # Start buffering; next line will be appended
            buffer = prefix
        else:
            if buffer is not None:
                # Append current line (with leading whitespace trimmed)
                merged = buffer + line.lstrip(' \t')
                merged_lines.append(merged)
                buffer = None
            else:
                merged_lines.append(line_stripped_nl)
    
    # If file ends with a hyphenated line, flush it without dash
    if buffer is not None:
        merged_lines.append(buffer)
    
    return merged_lines


def main():
    parser = argparse.ArgumentParser(
        description="Parse an indented index into keyphrase→pages JSON"
    )
    parser.add_argument("--input", help="Plain-text file with the index")
    args = parser.parse_args()

    try:
        with open(args.input, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    input_dir = os.path.dirname(os.path.abspath(args.input))
    output_dir = os.path.join(input_dir, "processed_inputs")
    os.makedirs(output_dir, exist_ok=True)

    # Set output file path
    output_file = os.path.join(output_dir, "page_to_keyphrases.json")

    # Merge hyphenated lines
    merged_lines = merge_hyphenated_lines(lines)

    # Get the phrase to pages mapping and invert it
    phrase_to_pages = parse_index(merged_lines)
    page_to_phrases = invert_mapping(phrase_to_pages)
    
    # Sort the dictionary by page number for better readability
    sorted_mapping = {k: page_to_phrases[k] for k in sorted(page_to_phrases.keys())}
    out_json = json.dumps(sorted_mapping, indent=2, ensure_ascii=False)
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"Successfully wrote output to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()


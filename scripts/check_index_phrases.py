import os
import re

def load_index_by_chapter(index_file):
    """Load key phrases grouped by their respective chapters."""
    index_by_chapter = {}
    current_chapter = None

    with open(index_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith('chapter '):
                current_chapter = line.lower().replace('chapter ', '').strip()
                index_by_chapter[current_chapter] = []
            elif line and current_chapter:
                # Remove trailing commas and page numbers
                cleaned_phrase = re.sub(r',?\s*\d+$', '', line)
                index_by_chapter[current_chapter].append(cleaned_phrase)

    return index_by_chapter

def find_missing_phrases_by_chapter(index_by_chapter, chapters_dir):
    """Find missing phrases for each chapter."""
    missing_phrases = {}

    for chapter, phrases in index_by_chapter.items():
        chapter_file = os.path.join(chapters_dir, f'ch{chapter}.txt')
        if not os.path.exists(chapter_file):
            print(f"Warning: Chapter file {chapter_file} not found.")
            continue

        with open(chapter_file, 'r') as f:
            chapter_text = f.read()
        missing_phrases[chapter] = [phrase for phrase in phrases if phrase not in chapter_text]

    return missing_phrases

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))

    index_file = os.path.join(base_dir, 'data', 'index_by_chapter.txt')
    chapters_dir = os.path.join(base_dir, 'data', 'self_extracted_chapters')

    # Load phrases grouped by chapter
    index_by_chapter = load_index_by_chapter(index_file)

    # Find missing phrases by chapter
    missing_phrases_by_chapter = find_missing_phrases_by_chapter(index_by_chapter, chapters_dir)

    # Output missing phrases
    for chapter, missing_phrases in missing_phrases_by_chapter.items():
        if missing_phrases:
            print(f"Chapter {chapter}: Phrases not found:")
            for phrase in missing_phrases:
                print(f"  {phrase}")
            # Calculate and print the percentage of missing phrases
            missing_percentage = (len(missing_phrases) / len(index_by_chapter[chapter])) * 100
            print(f"  Percentage of missing phrases: {missing_percentage:.2f}%")
        else:
            print(f"Chapter {chapter}: All phrases are present.")

    # Output total percentage of missing phrases
    total_phrases = sum(len(phrases) for phrases in index_by_chapter.values())
    total_missing = sum(len(missing) for missing in missing_phrases_by_chapter.values())
    total_missing_percentage = (total_missing / total_phrases) * 100 if total_phrases > 0 else 0
    print(f"\nTotal percentage of missing phrases: {total_missing_percentage:.2f}%")

if __name__ == "__main__":
    main()

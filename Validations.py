import nltk_init
import re
import numpy as np
import pdfplumber
from nltk import sent_tokenize

def detect_columns_per_page(page, gap_threshold_ratio=0.04):
    """
    Detect number of columns on a single PDF page using horizontal gaps.
    """
    words = page.extract_words()
    if not words:
        return 1

    # midpoints of each word
    x_positions = sorted([
        (float(w['x0']) + float(w['x1'])) / 2
        for w in words
    ])

    # remove duplicates & noise
    x_positions = np.unique(x_positions)

    if len(x_positions) < 3:
        return 1

    gaps = np.diff(x_positions)
    page_width = page.width
    gap_threshold = page_width * gap_threshold_ratio

    # gaps larger than threshold create column boundaries
    num_columns = int(np.sum(gaps > gap_threshold) + 1)
    return num_columns

#DETECTS IF IT HAS TABLE
def has_table(pdf_path, sample_pages=3):
    """
    Detects if the PDF has a table in the first sample_pages pages.
    Returns True if at least one table is found.
    """
    with pdfplumber.open(pdf_path) as pdf:
        pages_to_check = pdf.pages[:sample_pages]
        for page in pages_to_check:
            tables = page.extract_tables()
            if tables and len(tables) > 0:
                # Optional: check if table has more than 1 row and column
                for table in tables:
                    if len(table) > 1 and len(table[0]) > 1:
                        return True
    return False

#DETECT IF THE PAGE HAS MULTI COLUMN
def pdf_is_multicolumn(pdf_path, sample_pages=3, gap_threshold_ratio=0.04):
    """
    True if the PDF has more than 1 column in ANY sampled page.
    False if consistently single-column.
    """

    with pdfplumber.open(pdf_path) as pdf:
        pages_to_check = pdf.pages[:sample_pages]
        column_counts = []

        for page in pages_to_check:
            cols = detect_columns_per_page(page, gap_threshold_ratio)
            column_counts.append(cols)

        # If any page has more than 1 column → multicolumn PDF
        return any(c > 1 for c in column_counts), column_counts[0]

#VALIDATE IF IT HAS TABLE AND MULTIPLE COLUMN
def validate_structure(pdf_path, sample_pages=3):
    column_counts = pdf_is_multicolumn(pdf_path)[1]
    if has_table(pdf_path):
        return True
    elif has_table(pdf_path) and column_counts > 1:
        #RETURN TRUE IF THERE IS A TABLE AND COLUMN IS MORE THAN 2
        return True

    return False

def validate_text(text, min_sentences=10):
    clean_text = re.sub(r'--- Page \d+ ---', '', text)
    sentences = sent_tokenize(clean_text)
    return len(sentences) >= min_sentences, len(sentences)

def extract_text_excluding_tables(pdf_path, overlap_ratio=0.05):
    """
    Extract text from a PDF while ignoring all words inside table regions.
    Handles multi-column layouts with optional overlap.
    """
    is_complex, num_columns = pdf_is_multicolumn(pdf_path)
    #VALIDATE
    if validate_structure(pdf_path):
        return False

    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_width = page.width
            column_width = page_width / num_columns
            overlap = column_width * overlap_ratio
            page_text = ""

            # Detect tables using page.find_tables() to get bounding boxes
            table_bboxes = []
            for table in page.find_tables():
                # table.bbox = (x0, top, x1, bottom)
                table_bboxes.append(table.bbox)

            # Extract all words
            words = page.extract_words()

            for col in range(num_columns):
                left = max(col * column_width - overlap, 0)
                right = min((col + 1) * column_width + overlap, page_width)

                # Filter words in this column
                col_words = [
                    w for w in words
                    if ((float(w['x0']) + float(w['x1'])) / 2 >= left) and
                       ((float(w['x0']) + float(w['x1'])) / 2 <= right)
                ]

                # Exclude words inside table bounding boxes
                def is_in_table_bbox(word):
                    x0, top, x1, bottom = float(word['x0']), float(word['top']), float(word['x1']), float(
                        word['bottom'])
                    for tb in table_bboxes:
                        if x0 >= tb[0] and x1 <= tb[2] and top >= tb[1] and bottom <= tb[3]:
                            return True
                    return False

                col_words = [w for w in col_words if not is_in_table_bbox(w)]

                # Sort words top to bottom, left to right
                col_words.sort(key=lambda w: (float(w['top']), float(w['x0'])))

                col_text = " ".join(w['text'] for w in col_words)
                col_text = re.sub(r'\s+', ' ', col_text).strip()
                page_text += col_text + " "

            full_text += f"\n--- Page {page_num + 1} ---\n"
            full_text += page_text.strip()

    return full_text.strip()

#DETECTS IF EXTRACTED TEXT REALLY HAS SENTENCES
def contains_actual_sentences(text, min_words_per_sentence=3):
    """
    Check if text contains at least one sentence with >= min_words_per_sentence words.
    This filters out text that is just a list of words.
    """
    clean_text = re.sub(r'--- Page \d+ ---', '', text)
    sentences = sent_tokenize(clean_text)

    for sent in sentences:
        if len(sent.split()) >= min_words_per_sentence:
            return True
    return False

if __name__ == "__main__":
    pdf_file = "C:/Users/Dan/Downloads/3sampletext.pdf"
    extracted_text = extract_text_excluding_tables(pdf_file)
    if not extracted_text:
        print("File has complex structure")
    else:
        is_valid_length, sentence_count = validate_text(extracted_text)
        has_actual_sentences = contains_actual_sentences(extracted_text)

        if not has_actual_sentences:
            print("Warning: Extracted text does not contain proper sentences!")
        elif not is_valid_length:
            print(f"Text too short! Only {sentence_count} sentences detected.")
        else:
            print(f"Valid text with {sentence_count} sentences:\n")
            print(extracted_text)
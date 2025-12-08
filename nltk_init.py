# nltk_init.py
import nltk
import os


def initialize_nltk():
    """Initialize NLTK with proper paths for Railway"""
    nltk_data_path = '/tmp/nltk_data'

    # Ensure directory exists
    os.makedirs(nltk_data_path, exist_ok=True)

    # Add to NLTK path
    nltk.data.path.insert(0, nltk_data_path)  # Insert at beginning to check first
    os.environ['NLTK_DATA'] = nltk_data_path

    # Download if needed
    punkt_tab_path = os.path.join(nltk_data_path, 'tokenizers/punkt_tab')
    if not os.path.exists(punkt_tab_path):
        print("Downloading NLTK data...")
        nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)

    print(f"NLTK initialized at: {nltk_data_path}")


# Run initialization
initialize_nltk()
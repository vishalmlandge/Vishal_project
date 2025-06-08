import torch
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
import numpy as np
import re
import gdown
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths
MODEL_PATH = "./multilingual_toxic_detector_model"
BEST_MODEL_PATH = "./best_model.pt"
SAFETENSORS_PATH = "./distilbert_local/model.safetensors"

# Google Drive URLs for large files (replace with actual URLs)
BEST_MODEL_URL = "https://drive.google.com/file/d/1EDScf3YTZXTvKP5HmjpqSsA_i5X80Rbo/view?usp=sharing"
SAFETENSORS_URL = "https://drive.google.com/file/d/1x78fUu2yvLhxH_nBIc1pn5vj6iBDTrL-/view?usp=sharing"

# Global variables for lazy loading
device = torch.device("cpu")  # Force CPU for Render free tier
tokenizer = None
model = None

def load_model():
    """Load tokenizer and model lazily, downloading files if needed."""
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return

    logger.info("Initializing tokenizer and model...")

    # Download large files if not present
    if not os.path.exists(BEST_MODEL_PATH):
        logger.info(f"Downloading best_model.pt from {BEST_MODEL_URL}")
        gdown.download(BEST_MODEL_URL, BEST_MODEL_PATH, quiet=False)
    if not os.path.exists(SAFETENSORS_PATH):
        logger.info(f"Downloading model.safetensors from {SAFETENSORS_URL}")
        gdown.download(SAFETENSORS_URL, SAFETENSORS_PATH, quiet=False)

    try:
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        logger.info("Tokenizer loaded successfully")

        # Load model configuration
        config = DistilBertConfig.from_pretrained(
            MODEL_PATH,
            num_labels=6,
            problem_type="multi_label_classification",
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3
        )

        # Load model
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_PATH,
            config=config,
            ignore_mismatched_sizes=True
        )

        # Load best model weights
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(f"Best model weights not found at {BEST_MODEL_PATH}")
        state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        logger.info(f"Successfully loaded multilingual model with weights from {BEST_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise e

def is_marathi_hindi(text):
    """
    Determine if the text is in Marathi or Hindi based on Devanagari script.
    
    Args:
        text (str): The text to analyze.
    
    Returns:
        bool: True if the text is Marathi or Hindi, False otherwise.
    """
    devanagari_regex = r'[\u0900-\u097F]'
    return bool(re.search(devanagari_regex, text))

def score_comment(text):
    """
    Analyze the input text for toxicity and identify specific toxic words using the multilingual model.
    
    Args:
        text (str): The text to analyze.
    
    Returns:
        dict: A dictionary containing:
            - is_toxic (bool): Whether the text is toxic.
            - toxic_words (list): List of toxic words for toxic chats (empty for non-toxic).
            - scores (dict): Toxicity scores for each category.
    """
    logger.info(f"Processing comment: {text!r}")
    logger.info(f"Input ASCII: {[ord(c) for c in text]}")
    
    # Normalize input: strip whitespace and normalize spaces
    text = ' '.join(text.strip().split())
    logger.info(f"Normalized text: {text!r}")
    
    # Check language
    language = "Marathi-Hindi" if is_marathi_hindi(text) else "English"
    logger.info(f"Detected language: {language}")
    
    # Check for toxic patterns: asterisk, backslash, forward slash, or ellipsis
    has_toxic_pattern = bool(re.search(r'\b[\w*\\\/]*[\*\\\/\.]{1,}[\w*\\\/]*\b', text))
    logger.info(f"Contains toxic pattern (*, \, /, ...): {has_toxic_pattern}")
    
    # Load model if not already loaded
    load_model()
    
    # Tokenize the input text for overall scoring
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
    logger.info(f"Tokenized input IDs: {inputs['input_ids'].tolist()}")
    logger.info(f"Decoded tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    
    # Run inference for overall message
    with torch.no_grad():
        outputs = model(**inputs).logits
    probs = torch.sigmoid(outputs).cpu().numpy()[0]
    logger.info(f"Raw logits: {outputs.cpu().numpy()[0].tolist()}")
    
    # Define toxicity categories (aligned with sample.ipynb)
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Create scores dictionary
    scores = {cat: float(prob) for cat, prob in zip(categories, probs)}
    
    # Extract words for toxicity check
    words = re.findall(r'\b[\w\'*\\\/-]+\b', text.lower())
    logger.info(f"Extracted words: {words}")
    
    # Determine if the text is toxic (model threshold 0.3 or toxic pattern)
    is_toxic = has_toxic_pattern or any(prob > 0.3 for prob in probs)
    logger.info(f"Is toxic: {is_toxic}, scores: {scores}")
    
    # Extract toxic words for toxic messages
    toxic_words = []
    if is_toxic:
        for word in words:
            # Skip common non-toxic words to avoid false positives
            if word in ['tu', 'ahe', 'kya', 'kar', 'raha', 'hai', 'ka', 'se']:
                logger.info(f"Skipped non-toxic word: {word}")
                continue
            # Add words with toxic patterns
            if any(p in word for p in ['*', '\\', '/', '...']):
                toxic_words.append(word)
                logger.info(f"Added toxic word (pattern): {word}")
                continue
            # Score other words
            word_inputs = tokenizer(word, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
            with torch.no_grad():
                word_outputs = model(**word_inputs).logits
            word_probs = torch.sigmoid(word_outputs).cpu().numpy()[0]
            if any(word_prob > 0.3 for word_prob in word_probs):
                toxic_words.append(word)
                logger.info(f"Added model-detected toxic word: {word}, scores: {list(word_probs)}")
            else:
                logger.info(f"Word not toxic: {word}, scores: {list(word_probs)}")
    
    # Remove duplicates while preserving order
    toxic_words = list(dict.fromkeys(toxic_words))
    logger.info(f"Final toxic words: {toxic_words}")
    
    return {
        "is_toxic": is_toxic,
        "toxic_words": toxic_words,
        "scores": scores
    }

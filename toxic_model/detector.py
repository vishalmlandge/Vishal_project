import torch
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
import numpy as np
import re

# Load the multilingual model and tokenizer
model_path = "./multilingual_toxic_detector_model"
best_model_path = "./best_model.pt"
try:
    # Define device before using it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    config = DistilBertConfig.from_pretrained(
        model_path,
        num_labels=6,
        problem_type="multi_label_classification",
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        ignore_mismatched_sizes=True
    )
    # Load the best model weights
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model weights not found at {best_model_path}")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Successfully loaded multilingual model with weights from {best_model_path}")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
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
    print(f"Processing comment: {text!r}")
    print(f"Input ASCII: {[ord(c) for c in text]}")
    
    # Normalize input: strip whitespace and normalize spaces
    text = ' '.join(text.strip().split())
    print(f"Normalized text: {text!r}")
    
    # Check language
    language = "Marathi-Hindi" if is_marathi_hindi(text) else "English"
    print(f"Detected language: {language}")
    
    # Check for toxic patterns: asterisk, backslash, forward slash, or ellipsis
    has_toxic_pattern = bool(re.search(r'\b[\w*\\\/]*[\*\\\/\.]{1,}[\w*\\\/]*\b', text))
    print(f"Contains toxic pattern (*, \, /, ...): {has_toxic_pattern}")
    
    # Tokenize the input text for overall scoring
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
    print(f"Tokenized input IDs: {inputs['input_ids'].tolist()}")
    print(f"Decoded tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    
    # Run inference for overall message
    with torch.no_grad():
        outputs = model(**inputs).logits
    probs = torch.sigmoid(outputs).cpu().numpy()[0]
    print(f"Raw logits: {outputs.cpu().numpy()[0].tolist()}")
    
    # Define toxicity categories (aligned with sample.ipynb)
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Create scores dictionary
    scores = {cat: float(prob) for cat, prob in zip(categories, probs)}
    
    # Extract words for toxicity check
    words = re.findall(r'\b[\w\'*\\\/-]+\b', text.lower())
    print(f"Extracted words: {words}")
    
    # Determine if the text is toxic (model threshold 0.3 or toxic pattern)
    is_toxic = has_toxic_pattern or any(prob > 0.3 for prob in probs)
    print(f"Is toxic: {is_toxic}, scores: {scores}")
    
    # Extract toxic words for toxic messages
    toxic_words = []
    if is_toxic:
        for word in words:
            # Skip common non-toxic words to avoid false positives
            if word in ['tu', 'ahe', 'kya', 'kar', 'raha', 'hai', 'ka', 'se']:
                print(f"Skipped non-toxic word: {word}")
                continue
            # Add words with toxic patterns
            if any(p in word for p in ['*', '\\', '/', '...']):
                toxic_words.append(word)
                print(f"Added toxic word (pattern): {word}")
                continue
            # Score other words
            word_inputs = tokenizer(word, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
            with torch.no_grad():
                word_outputs = model(**word_inputs).logits
            word_probs = torch.sigmoid(word_outputs).cpu().numpy()[0]
            if any(word_prob > 0.3 for word_prob in word_probs):
                toxic_words.append(word)
                print(f"Added model-detected toxic word: {word}, scores: {list(word_probs)}")
            else:
                print(f"Word not toxic: {word}, scores: {list(word_probs)}")
    
    # Remove duplicates while preserving order
    toxic_words = list(dict.fromkeys(toxic_words))
    print(f"Final toxic words: {toxic_words}")
    
    return {
        "is_toxic": is_toxic,
        "toxic_words": toxic_words,
        "scores": scores
    }
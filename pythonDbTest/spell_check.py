import re
from spellchecker import SpellChecker
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from tqdm import tqdm
import pandas as pd

LEMMATIZE = False

def convert_to_antonym(words):
    new_words = []
    temp_word = ''

    for word in words:
        if word == 'not':
            temp_word = 'not_'
        elif temp_word == 'not_':
            antonyms = set()
            for syn in wordnet.synsets(word):
                antonyms.update(a.name() for s in syn.lemmas() for a in s.antonyms())
            if antonyms:
                word = next(iter(antonyms))  # Take one antonym
            else:
                word = temp_word + word  # When antonym is not found
            temp_word = ''

        if word != 'not':
            new_words.append(word)

    return new_words


lmtzr = WordNetLemmatizer()
spell = SpellChecker()

# Cache to store corrections
correction_cache = {}

def spell_check(words):
    corrected_words = []
    for word in words:
        if word in spell:
            corrected_words.append(word)  # No change
        else:
            corrected_word = spell.candidates(word)
            if corrected_word:
                corrected_words.append(next(iter(corrected_word)))  # Take the first suggestion
            else:
                corrected_words.append(word)  # Keep original if no suggestion
    return corrected_words

def replace_url(text):
    if pd.isna(text):  # Check for NaN values
        return text  # Return as is if NaN
    if isinstance(text, str):  # Ensure the input is a string
        # Define the pattern for URLs
        url_pattern = re.compile(r'https?://\S+|www\.\S+|http://\S+')

        # Clean the text
        clean_text = url_pattern.sub('', text)
        clean_text = re.sub(r'@[^\s]+', '', clean_text)
        clean_text = clean_text.lower()
        clean_text = re.sub(r'[^\w\s#]', ' ', clean_text)

        clean_text = re.sub(r'\d+', ' ', clean_text)

        # Tokenize once
        words = nltk.word_tokenize(clean_text)

        # Apply antonym conversion
        words = convert_to_antonym(words)

        # Apply spell check
        if LEMMATIZE:
            words = spell_check(words)

        # Lemmatization
        words = [lmtzr.lemmatize(word) for word in words]

        return " ".join(words)

    return text  # Return original if not a string

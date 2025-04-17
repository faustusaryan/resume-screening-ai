import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """Extract important words from text using spaCy"""
    doc = nlp(text.lower())
    # Get nouns, proper nouns, and verbs
    keywords = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop]
    return " ".join(keywords)

def calculate_semantic_match_score(resume_text, job_desc_text):
    """Calculate semantic similarity score using spaCy word vectors"""
    # Clean the texts
    cleaned_resume = preprocess(resume_text)
    cleaned_jd = preprocess(job_desc_text)
    
    # Process texts with spaCy
    doc1 = nlp(cleaned_resume)
    doc2 = nlp(cleaned_jd)
    
    # Calculate similarity (spaCy's built-in method)
    similarity = doc1.similarity(doc2)
    
    # Convert to percentage
    score_percent = round(similarity * 100, 2)
    return score_percent
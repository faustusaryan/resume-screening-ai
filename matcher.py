from sentence_transformers import SentenceTransformer, util
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import defaultdict

# Load models globally
transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text, top_n=20):
    """Extract top keywords using TF-IDF"""
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform([text])
    features = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]
    return [features[i] for i in np.argsort(scores)[-top_n:]]

def get_keyword_matches(resume_text, jd_text):
    """Get matched keywords with highlighting"""
    jd_keywords = extract_keywords(jd_text)
    resume_keywords = extract_keywords(resume_text)
    return list(set(jd_keywords) & set(resume_keywords))

def get_spacy_score(resume_text, jd_text):
    doc1 = nlp(resume_text.lower())
    doc2 = nlp(jd_text.lower())
    return round(doc1.similarity(doc2) * 100, 2)

def get_transformer_score(resume_text, jd_text):
    jd_embedding = transformer_model.encode(jd_text, convert_to_tensor=True)
    resume_embedding = transformer_model.encode(resume_text, convert_to_tensor=True)
    similarity = util.cos_sim(jd_embedding, resume_embedding)
    return round(float(similarity[0][0]) * 100, 2)

def get_score(resume_text, jd_text, weights=(0.5, 0.3, 0.2)):
    """Hybrid scoring with configurable weights"""
    # Calculate all scores
    transformer_score = get_transformer_score(resume_text, jd_text)
    spacy_score = get_spacy_score(resume_text, jd_text)
    keyword_matches = get_keyword_matches(resume_text, jd_text)
    keyword_score = round(len(keyword_matches) / len(extract_keywords(jd_text)) * 100, 2)
    
    # Apply weights
    final_score = round(
        (transformer_score * weights[0]) + 
        (spacy_score * weights[1]) + 
        (keyword_score * weights[2]), 
    2)
    
    return {
        'final_score': final_score,
        'breakdown': {
            'transformer': transformer_score,
            'spacy': spacy_score,
            'keyword': keyword_score,
            'weights': weights
        },
        'keyword_matches': keyword_matches
    }

def get_best_match_result(results):
    if not results:
        return default_result()
    
    best = max(results, key=lambda x: x['score'])
    return {
        'score': min(float(best['score']), 100.0),
        'breakdown': best.get('breakdown', {}),
        'skills': best.get('skills', []),
        'experience': best.get('experience', []),
        'keyword_matches': best.get('keyword_matches', [])
    }
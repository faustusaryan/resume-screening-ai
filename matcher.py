from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import defaultdict

# Load transformer model only
transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_keywords(text, top_n=20):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform([text])
    features = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]
    return [features[i] for i in np.argsort(scores)[-top_n:]]

def get_keyword_matches(resume_text, jd_text):
    jd_keywords = extract_keywords(jd_text)
    resume_keywords = extract_keywords(resume_text)
    return list(set(jd_keywords) & set(resume_keywords))

def get_transformer_score(resume_text, jd_text):
    jd_embedding = transformer_model.encode(jd_text, convert_to_tensor=True)
    resume_embedding = transformer_model.encode(resume_text, convert_to_tensor=True)
    similarity = util.cos_sim(jd_embedding, resume_embedding)
    return round(float(similarity[0][0]) * 100, 2)

def get_score(resume_text, jd_text, weights=(0.8, 0.2)):
    transformer_score = get_transformer_score(resume_text, jd_text)
    keyword_matches = get_keyword_matches(resume_text, jd_text)
    keyword_score = round(len(keyword_matches) / len(extract_keywords(jd_text)) * 100, 2)

    final_score = round(
        (transformer_score * weights[0]) +
        (keyword_score * weights[1]),
    2)

    return {
        'final_score': final_score,
        'breakdown': {
            'transformer': transformer_score,
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

def default_result():
    return {
        'score': 0.0,
        'breakdown': {},
        'skills': [],
        'experience': [],
        'keyword_matches': []
    }

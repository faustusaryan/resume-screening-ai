from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define your priority keywords (customize as needed)
PRIORITY_KEYWORDS = ['python', 'java', 'sql', 'machine learning', 'ai']

def calculate_match_score(resume_text, job_desc_text):
    # Combine job_desc + resume for TF-IDF vectorizer
    corpus = [job_desc_text, resume_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    job_vector = tfidf_matrix[0]  # Job description
    resume_vector = tfidf_matrix[1]  # Resume

    similarity = cosine_similarity(job_vector, resume_vector).flatten()[0]

    # Boost score for resume with priority keywords
    extra_weight = 0
    for keyword in PRIORITY_KEYWORDS:
        if keyword in resume_text.lower():
            extra_weight += 0.05  # 5% boost per keyword match
    
    score = min(similarity + extra_weight, 1.0)
    return round(score, 4)  # Return decimal between 0-1
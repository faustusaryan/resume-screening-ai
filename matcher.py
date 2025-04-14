from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(job_desc, resumes):
    # Combine job_desc + all resumes for TF-IDF vectorizer
    corpus = [job_desc] + resumes
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    job_vector = tfidf_matrix[0]  # Job description
    resume_vectors = tfidf_matrix[1:]  # All resumes

    similarities = cosine_similarity(job_vector, resume_vectors).flatten()

    # Optional: Boost priority keyword resumes
    boosted_scores = []
    for i, resume in enumerate(resumes):
        extra_weight = 0
        for keyword in PRIORITY_KEYWORDS:
            if keyword in resume.lower():
                extra_weight += 0.05  # 5% boost per match (tune as needed)
        score = min(similarities[i] + extra_weight, 1.0)
        boosted_scores.append(round(score * 100, 2))  # Normalize to 0-100%

    return boosted_scores

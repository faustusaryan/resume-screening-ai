from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string
import re

# Download stopwords (only once needed)
nltk.download('stopwords')

# Define skill keywords for higher weighting
PRIORITY_KEYWORDS = {
    "python", "java", "sql", "machine learning", "deep learning", "nlp",
    "data analysis", "pandas", "numpy", "tensorflow", "keras", "flask",
    "django", "docker", "api", "cloud", "aws", "azure", "linux"
}

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Tokenize and remove stopwords
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def weight_skills(text):
    for skill in PRIORITY_KEYWORDS:
        # Repeat skills to artificially increase weight (tune this factor as needed)
        pattern = re.compile(rf'\b{re.escape(skill)}\b', re.IGNORECASE)
        matches = len(pattern.findall(text))
        text += (" " + skill) * matches  # Append skill again to increase TF-IDF
    return text

def calculate_match_score(resume_text, jd_text):
    # Step 1: Preprocess
    resume_cleaned = preprocess(resume_text)
    jd_cleaned = preprocess(jd_text)

    # Step 2: Boost skills in both
    resume_weighted = weight_skills(resume_cleaned)
    jd_weighted = weight_skills(jd_cleaned)

    # Step 3: TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_weighted, jd_weighted])

    # Step 4: Cosine Similarity
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Step 5: Normalize to 0-100%
    return round(score * 100, 2)

def get_missing_keywords(resume_text, jd_text):
    resume_words = set(preprocess(resume_text).split())
    jd_words = set(preprocess(jd_text).split())
    missing = jd_words - resume_words
    return sorted(list(missing))

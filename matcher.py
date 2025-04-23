from sentence_transformers import SentenceTransformer, util
import spacy

# Load models globally (better performance)
transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

def get_spacy_score(resume_text, jd_text):
    """Calculate similarity using spaCy"""
    doc1 = nlp(resume_text.lower())
    doc2 = nlp(jd_text.lower())
    return round(doc1.similarity(doc2) * 100, 2)  # Percentage with 2 decimals

def get_transformer_score(resume_text, jd_text):
    """Calculate similarity using Sentence Transformers"""
    jd_embedding = transformer_model.encode(jd_text, convert_to_tensor=True)
    resume_embedding = transformer_model.encode(resume_text, convert_to_tensor=True)
    similarity = util.cos_sim(jd_embedding, resume_embedding)
    return round(float(similarity[0][0]) * 100, 2)

def get_score(resume_text, jd_text):
    """Hybrid scoring (70% Transformer + 30% spaCy)"""
    transformer_score = get_transformer_score(resume_text, jd_text)
    spacy_score = get_spacy_score(resume_text, jd_text)
    
    final_score = round((transformer_score * 0.7) + (spacy_score * 0.3), 2)
    
    return final_score, {
        'transformer': transformer_score,
        'spacy': spacy_score,
        'final': final_score
    }

def get_best_match_result(results):
    if not results:
        return {
            "score": 0.0,
            "breakdown": {
                "transformer": 0.0,
                "spacy": 0.0,
                "keyword": 0.0
            },
            "skills": [],
            "job_description": {}
        }

    best_match = results[0]

    # Score Fix
    score = best_match.get("score")
    if score is None:
        score = 0.0
    best_match["score"] = min(float(score), 100.0)

    # Breakdown Fix
    breakdown = best_match.get("breakdown") or {}
    best_match["breakdown"] = {
        "transformer": float(breakdown.get("transformer") or 0.0),
        "spacy": float(breakdown.get("spacy") or 0.0),
        "keyword": float(breakdown.get("keyword") or 0.0)
    }

    # Skills Fix
    skills = best_match.get("skills")
    if skills is None:
        best_match["skills"] = []

    # Job Description Fix
    job_description = best_match.get("job_description")
    if job_description is None:
        best_match["job_description"] = {}

    return best_match

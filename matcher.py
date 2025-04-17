from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast + accurate

def get_score(resume_text, jd_text):
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)

    similarity = util.cos_sim(jd_embedding, resume_embedding)
    score = float(similarity[0][0]) * 100  # Convert to percentage
    return round(score, 2)

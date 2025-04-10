def calculate_match_score(resume_text, jd_text):
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())

    if not jd_words:
        return 0.0

    matched_words = resume_words.intersection(jd_words)
    return min(len(matched_words) / len(jd_words), 1.0)  # Caps score at 100%
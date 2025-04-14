import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from parser import extract_text_from_pdf
from matcher import calculate_match_score

app = Flask(__name__)
app.secret_key = 'your-secret-key-123'

UPLOAD_FOLDER_RESUMES = 'resumes'
UPLOAD_FOLDER_JD = 'job_descriptions'
os.makedirs(UPLOAD_FOLDER_RESUMES, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_JD, exist_ok=True)
app.config['UPLOAD_FOLDER_RESUMES'] = UPLOAD_FOLDER_RESUMES
app.config['UPLOAD_FOLDER_JD'] = UPLOAD_FOLDER_JD

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'job_description' not in request.files or 'resumes' not in request.files:
        flash('Please upload the job description and at least one resume.', 'error')
        return redirect(url_for('home'))

    jd_file = request.files['job_description']
    resume_files = request.files.getlist('resumes')

    if not jd_file.filename.lower().endswith('.pdf'):
        flash('Only PDF files are allowed for job description!', 'error')
        return redirect(url_for('home'))

    resume_scores = []

    try:
        jd_filename = secure_filename(jd_file.filename)
        jd_path = os.path.join(app.config['UPLOAD_FOLDER_JD'], jd_filename)
        jd_file.save(jd_path)
        jd_text = extract_text_from_pdf(jd_path)

        for resume_file in resume_files:
            if resume_file.filename.lower().endswith('.pdf'):
                resume_filename = secure_filename(resume_file.filename)
                resume_path = os.path.join(app.config['UPLOAD_FOLDER_RESUMES'], resume_filename)
                resume_file.save(resume_path)

                resume_text = extract_text_from_pdf(resume_path)
                score = calculate_match_score(resume_text, jd_text)
                resume_scores.append({
                    'filename': resume_filename,
                    'score': round(score * 100, 2),
                    'text': resume_text
                })

        # Best matched resume
        top_resume = max(resume_scores, key=lambda x: x['score']) if resume_scores else None

        return render_template(
            'result.html',
            jd_text=jd_text,
            top_resume=top_resume,
            all_scores=resume_scores
        )

    except Exception as e:
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

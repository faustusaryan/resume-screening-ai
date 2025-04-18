import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from parser import extract_text_from_pdf
from matcher import get_score

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-123')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    jd_file = request.files.get('job_description')
    resume_files = request.files.getlist('resumes')

    if not jd_file or jd_file.filename == '' or not resume_files or resume_files[0].filename == '':
        flash('Please upload a job description and at least one resume (PDF).', 'error')
        return redirect(url_for('home'))

    if not allowed_file(jd_file.filename):
        flash('Only PDF files are allowed for the job description.', 'error')
        return redirect(url_for('home'))

    try:
        # Save JD
        jd_filename = secure_filename(jd_file.filename)
        jd_path = os.path.join(app.config['UPLOAD_FOLDER'], jd_filename)
        jd_file.save(jd_path)
        jd_text = extract_text_from_pdf(jd_path)

        results = []
        for resume_file in resume_files:
            if resume_file and allowed_file(resume_file.filename):
                resume_filename = secure_filename(resume_file.filename)
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_filename)
                resume_file.save(resume_path)

                resume_text = extract_text_from_pdf(resume_path)
                score = get_score(resume_text, jd_text)

                results.append({
                    'filename': resume_filename,
                    'score': round(score, 2)
                })

        if not results:
            flash('No valid resumes were uploaded.', 'error')
            return redirect(url_for('home'))

        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        best_match = results[0]

        return render_template('result.html',
                               results=results,
                               best_match=best_match,
                               jd_text=jd_text)

    except Exception as e:
        flash(f"An error occurred while processing: {str(e)}", 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

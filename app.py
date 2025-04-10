import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from parser import extract_text_from_pdf
from matcher import calculate_match_score

app = Flask(__name__)
app.secret_key = 'your-secret-key-123'  # Required for flash messages

# Upload folders
UPLOAD_FOLDER_RESUMES = 'resumes'
UPLOAD_FOLDER_JD = 'job_descriptions'

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER_RESUMES, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_JD, exist_ok=True)

app.config['UPLOAD_FOLDER_RESUMES'] = UPLOAD_FOLDER_RESUMES
app.config['UPLOAD_FOLDER_JD'] = UPLOAD_FOLDER_JD

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'resume' not in request.files or 'job_description' not in request.files:
        flash('Please upload both files!', 'error')
        return redirect(url_for('home'))

    resume_file = request.files['resume']
    jd_file = request.files['job_description']

    # Check for PDF files only
    if not (resume_file.filename.lower().endswith('.pdf') and jd_file.filename.lower().endswith('.pdf')):
        flash('Only PDF files are allowed!', 'error')
        return redirect(url_for('home'))

    try:
        # Secure filenames
        resume_filename = secure_filename(resume_file.filename)
        jd_filename = secure_filename(jd_file.filename)

        # Save files
        resume_path = os.path.join(app.config['UPLOAD_FOLDER_RESUMES'], resume_filename)
        jd_path = os.path.join(app.config['UPLOAD_FOLDER_JD'], jd_filename)
        
        resume_file.save(resume_path)
        jd_file.save(jd_path)

        # Extract text
        resume_text = extract_text_from_pdf(resume_path)
        jd_text = extract_text_from_pdf(jd_path)

        # Calculate score
        score = calculate_match_score(resume_text, jd_text)
        return render_template('result.html', score=round(score * 100, 2))

    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
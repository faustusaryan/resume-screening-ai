import os
from flask import Flask, render_template, request, flash, redirect, url_for, Response
from werkzeug.utils import secure_filename
from parser import extract_text_from_pdf, extract_structured_data
from matcher import get_score, get_best_match_result
from twilio.rest import Client
import logging
from dotenv import load_dotenv
import json

# Initialize
load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-fallback-key')

# Config
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Twilio Setup
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_TOKEN')
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None

# Logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Get scoring weights
        weights = (
            float(request.form.get('transformer_weight', 0.5)),
            float(request.form.get('spacy_weight', 0.3)),
            float(request.form.get('keyword_weight', 0.2))
        )
        
        # Process files
        jd_file = request.files['job_description']
        resume_files = request.files.getlist('resumes')
        
        # Save and process JD
        jd_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(jd_file.filename))
        jd_file.save(jd_path)
        jd_text = extract_text_from_pdf(jd_path)
        
        # Process resumes
        results = []
        for resume_file in resume_files:
            if resume_file and allowed_file(resume_file.filename):
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(resume_file.filename))
                resume_file.save(resume_path)
                
                resume_text = extract_text_from_pdf(resume_path)
                structured_data = extract_structured_data(resume_text)
                score_data = get_score(resume_text, jd_text, weights)
                
                results.append({
                    'filename': resume_file.filename,
                    'score': score_data['final_score'],
                    'breakdown': score_data['breakdown'],
                    'skills': structured_data['skills'],
                    'experience': structured_data['experience'],
                    'keyword_matches': score_data['keyword_matches']
                })
        
        # Get best match
        best_match = get_best_match_result(results) if results else None
        
        return render_template('result.html',
                            results=results,
                            best_match=best_match,
                            jd_text=jd_text)
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        flash(f'Processing error: {str(e)}', 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
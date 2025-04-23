import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from parser import extract_text_from_pdf, extract_structured_data
from matcher import get_score
from twilio.rest import Client
import logging
from dotenv import load_dotenv

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

def send_whatsapp_msg(phone, message):
    """Send WhatsApp message with error handling"""
    if not twilio_client:
        logging.warning("Twilio credentials missing - WhatsApp disabled")
        return False
    
    try:
        twilio_client.messages.create(
            body=f"Resume Screener Result:\n{message}",
            from_='whatsapp:+14155238886',
            to=f'whatsapp:+91{phone}'
        )
        return True
    except Exception as e:
        logging.error(f"WhatsApp failed: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Validate files
    if 'job_description' not in request.files or 'resumes' not in request.files:
        flash('Please upload both JD and resumes', 'error')
        return redirect(url_for('home'))
    
    jd_file = request.files['job_description']
    resume_files = request.files.getlist('resumes')

    if jd_file.filename == '' or not resume_files:
        flash('No selected files', 'error')
        return redirect(url_for('home'))

    try:
        # Process JD
        jd_filename = secure_filename(jd_file.filename)
        jd_path = os.path.join(app.config['UPLOAD_FOLDER'], jd_filename)
        jd_file.save(jd_path)
        jd_text = extract_text_from_pdf(jd_path)

        # Process resumes
        results = []
        for resume_file in resume_files:
            if resume_file and allowed_file(resume_file.filename):
                resume_filename = secure_filename(resume_file.filename)
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_filename)
                resume_file.save(resume_path)
                
                # Extract and analyze
                resume_text = extract_text_from_pdf(resume_path)
                structured_data = extract_structured_data(resume_text)
                score, breakdown = get_score(resume_text, jd_text)
                
                results.append({
                    'filename': resume_filename,
                    'score': score,
                    'breakdown': breakdown,
                    'skills': structured_data['skills'],
                    'experience': structured_data['experience']
                })

        # Sort and get best match
        results.sort(key=lambda x: x['score'], reverse=True)
        best_match = results[0] if results else None

        # WhatsApp notification
        phone = request.form.get('phone', '').strip()
        if phone and request.form.get('whatsapp_notify') and best_match:
            msg = f"Top Match: {best_match['filename']}\nScore: {best_match['score']}%"
            send_whatsapp_msg(phone, msg)

        return render_template('result.html',
                            results=results,
                            best_match=best_match,
                            jd_text=jd_text)

    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
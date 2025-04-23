import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    """Extract raw text from PDF with error handling"""
    try:
        with fitz.open(pdf_path) as doc:
            return " ".join([page.get_text() for page in doc])
    except Exception as e:
        raise Exception(f"PDF read error: {str(e)}")

def extract_structured_data(text):
    """Smart parsing for resume sections"""
    # Case-insensitive regex with section detection
    skills_match = re.search(r"(?i)skills[:\s]*(.*?)(?:\n\n|$)", text, re.DOTALL)
    exp_match = re.search(r"(?i)experience[:\s]*(.*?)(?:\n\n|$)", text, re.DOTALL)
    
    # Extract bullet points
    skills = []
    if skills_match:
        skills = [s.strip() for s in skills_match.group(1).split('\n') if s.strip()]
    
    # Experience lines cleanup
    experience = []
    if exp_match:
        experience = [e.replace('•', '').strip() 
                     for e in exp_match.group(1).split('\n') 
                     if e.strip()]
    
    return {
        'raw_text': text,
        'skills': skills[:10],  # Limit to top 10 skills
        'experience': experience[:5]  # Top 5 experience points
    }
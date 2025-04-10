import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with fitz.open(pdf_path) as doc:  # Auto-closes the file
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        raise Exception(f"Failed to read PDF: {str(e)}")
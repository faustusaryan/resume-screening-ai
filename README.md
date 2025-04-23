# 📄 Resume Screening AI

An AI-powered resume screening web application that uses Natural Language Processing to match job descriptions with candidate resumes.

## 🚀 Live Demo

👉 [Click to open app](https://resume-screening-ai-npp8.onrender.com)

## 📌 Features

- Upload **one job description** and **multiple resumes**
- Get **AI-generated match scores** for each resume
- See the **best match visualized with an animated ring**
- **Semantic matching** using spaCy NLP
- Clean and responsive user interface

## 🧠 How It Works

- Extracts meaningful words (nouns, verbs) using spaCy
- Computes **semantic similarity** between job description and each resume
- Ranks resumes based on relevance

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **NLP**: spaCy (`en_core_web_md`), cosine similarity
- **Frontend**: HTML, Bootstrap (Jinja templates), JavaScript
- **Deployment**: Render

## 🧪 How to Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/resume-screening-ai.git
   cd resume-screening-ai

# Install Dependencies
!pip install transformers datasets scikit-learn torchmetrics tensorboard pytorch-lightning spacy nltk onnx onnxruntime fastapi uvicorn python-multipart -q
!python -m spacy download en_core_web_sm
!python -m nltk.downloader words

# Imports
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import numpy as np
import random
import os
import spacy
import nltk
from nltk.corpus import words
from typing import List, Dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy, Precision, Recall, F1Score
import onnxruntime
import onnx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from datetime import datetime
import json
from transformers import pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Spacy model and NLTK word list
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt', quiet=True)
english_words = set(words.words())

# --- 1. Data Level Enhancements ---

def generate_realistic_pair(domains: List[str], english_words: set) -> Dict[str, str]:
    """Generates a realistic JD-Resume pair (can be match or non-match)."""
    domain = random.choice(domains)
    num_skills = random.randint(3, 8)
    all_skills = get_domain_skills(domain, english_words)
    if not all_skills:
        return None  # Handle cases where no skills are available for the domain

    jd_skills = random.sample(all_skills, num_skills)
    resume_skills_base = random.sample(all_skills, random.randint(2, 7))

    # Introduce some realistic variations and potential mismatches
    resume_skills = set(resume_skills_base)
    if random.random() < 0.7:  # Introduce some overlap for potential matches
        overlap = random.sample(jd_skills, random.randint(0, min(len(jd_skills), 3)))
        resume_skills.update(overlap)
    if random.random() < 0.3:  # Introduce some unrelated skills
        unrelated_skills = get_domain_skills(random.choice([d for d in domains if d != domain]), english_words)
        if unrelated_skills:
            resume_skills.update(random.sample(unrelated_skills, random.randint(0, 2)))
    resume_skills = list(resume_skills)

    jd = f"Looking for a professional with expertise in {', '.join(jd_skills)} within the {domain} domain."
    resume = f"Experienced individual with skills including {', '.join(resume_skills)}."
    label = 1 if set(jd_skills).issubset(set(resume_skills)) and len(set(jd_skills)) > 0 else 0
    return {"jd": jd, "resume": resume, "label": label}



def get_domain_skills(domain: str, english_words: set) -> List[str]:
    """Provides a list of example skills for a given domain."""
    domain_skills = {
        "IT": ["Python", "Java", "SQL", "AWS", "Azure", "Docker", "Kubernetes", "Machine Learning", "Deep Learning", "NLP", "JavaScript", "React", "Angular"],
        "Finance": ["Financial Modeling", "Risk Management", "Investment Analysis", "Accounting", "Budgeting", "Forecasting", "Valuation", "Economics", "Data Analysis"],
        "Healthcare": ["Patient Care", "Medical Records", "Diagnosis", "Treatment Planning", "Pharmacology", "Anatomy", "Physiology", "Clinical Trials", "Healthcare Management"],
        "Marketing": ["Digital Marketing", "Social Media Marketing", "SEO", "Content Creation", "Email Marketing", "Market Research", "Brand Management", "Advertising"],
        "Sales": ["Sales Strategy", "Customer Relationship Management", "Account Management", "Business Development", "Negotiation", "Sales Forecasting", "Lead Generation"],
        "HR": ["Talent Acquisition", "Employee Relations", "Compensation and Benefits", "Performance Management", "HR Strategy", "Training and Development"]
    }
    return [skill.lower() for skill in domain_skills.get(domain, []) if skill.lower() in english_words]

def generate_realistic_pair(domains: List[str], english_words: set) -> Dict[str, str]:
    """Generates a realistic JD-Resume pair (can be match or non-match)."""
    domain = random.choice(domains)
    all_skills = get_domain_skills(domain, english_words)
    if not all_skills:
        return None  # Handle cases where no skills are available for the domain

    max_skills_to_sample = min(random.randint(3, 8), len(all_skills)) # Ensure we don't sample more than available
    jd_skills = random.sample(all_skills, max_skills_to_sample)

    resume_skills_base = random.sample(all_skills, min(random.randint(2, 7), len(all_skills)))
    resume_skills = set(resume_skills_base)
    if random.random() < 0.7:  # Introduce some overlap for potential matches
        overlap = random.sample(jd_skills, min(random.randint(0, min(len(jd_skills), 3)), len(jd_skills)))
        resume_skills.update(overlap)
    if random.random() < 0.3:  # Introduce some unrelated skills
        other_domains = [d for d in domains if d != domain]
        if other_domains:
            unrelated_skills = get_domain_skills(random.choice(other_domains), english_words)
            if unrelated_skills:
                resume_skills.update(random.sample(unrelated_skills, min(random.randint(0, 2), len(unrelated_skills))))
    resume_skills = list(resume_skills)

    jd = f"Looking for a professional with expertise in {', '.join(jd_skills)} within the {domain} domain."
    resume = f"Experienced individual with skills including {', '.join(resume_skills)}."
    label = 1 if set(jd_skills).issubset(set(resume_skills)) and len(set(jd_skills)) > 0 else 0
    return {"jd": jd, "resume": resume, "label": label}

def generate_synthetic_data(num_samples: int = 1000, domains: List[str] = ["IT", "Finance", "Healthcare", "Marketing", "Sales", "HR"], english_words: set = english_words) -> List[Dict[str, str]]:
    """Generates synthetic JD-Resume pairs across multiple domains."""
    synthetic_data = []
    for _ in range(num_samples):
        pair = generate_realistic_pair(domains, english_words)
        if pair:
            synthetic_data.append(pair)
    return synthetic_data

# Generate synthetic data
synthetic_data = generate_synthetic_data(num_samples=1500)



def back_translate(text: str, src_lang: str = 'en', target_lang: str = 'fr') -> str:
    """Applies back-translation for text augmentation."""
    try:
        translator_forward = pipeline('translation', model=f'{src_lang}-{target_lang}')
        translator_backward = pipeline('translation', model=f'{target_lang}-{src_lang}')
        translated = translator_forward(text, max_length=512)[0]['translation_text']
        back_translated = translator_backward(translated, max_length=512)[0]['translation_text']
        return back_translated
    except Exception as e:
        logging.warning(f"Back-translation failed for '{text}': {e}")
        return text

def gpt_paraphrase(text: str) -> str:
    """
    Placeholder for GPT paraphrasing. Requires access to a GPT model.
    For demonstration, it returns the original text.
    """
    # In a real implementation, you would interact with a GPT API here.
    logging.info(f"GPT Paraphrasing (placeholder) applied to: '{text}'")
    return text

# Generate synthetic data
synthetic_data = generate_synthetic_data(num_samples=1500)

# Augment data
augmented_data = []
for entry in synthetic_data:
    augmented_data.append(entry) # Keep original
    if random.random() < 0.5:
        augmented_data.append({
            "jd": back_translate(entry['jd']),
            "resume": entry['resume'],
            "label": entry['label']
        })
    if random.random() < 0.5:
        augmented_data.append({
            "jd": entry['jd'],
            "resume": back_translate(entry['resume']),
            "label": entry['label']
        })
    if random.random() < 0.3:
        augmented_data.append({
            "jd": gpt_paraphrase(entry['jd']),
            "resume": entry['resume'],
            "label": entry['label']
        })
    if random.random() < 0.3:
        augmented_data.append({
            "jd": entry['jd'],
            "resume": gpt_paraphrase(entry['resume']),
            "label": entry['label']
        })

full_dataset = synthetic_data + augmented_data
random.shuffle(full_dataset)

# Handle class imbalance using SMOTE
labels = [item['label'] for item in full_dataset]
if labels.count(0) > 0 and labels.count(1) > 0 and labels.count(0) != labels.count(1):
    jd_texts = [item['jd'] for item in full_dataset]
    resume_texts = [item['resume'] for item in full_dataset]
    combined_texts = [f"{jd} [SEP] {resume}" for jd, resume in zip(jd_texts, resume_texts)]
    tokenizer_for_smote = AutoTokenizer.from_pretrained("roberta-base") # Using a base tokenizer
    encoded_texts = tokenizer_for_smote(combined_texts, padding=True, truncation=True, return_tensors='pt')['input_ids']
    encoded_texts_reshaped = encoded_texts.view(encoded_texts.size(0), -1).numpy() # Flatten for SMOTE

    smote = SMOTE(random_state=42)
    resampled_texts, resampled_labels = smote.fit_resample(encoded_texts_reshaped, labels)

    # Decode back to text and reconstruct dataset (simplified - consider feature-based SMOTE for better results)
    resampled_dataset = []
    decoded_texts = tokenizer_for_smote.batch_decode(resampled_texts, skip_special_tokens=True)
    for text, label in zip(decoded_texts, resampled_labels):
        jd, resume = text.split("[SEP]")
        resampled_dataset.append({"jd": jd.strip(), "resume": resume.strip(), "label": label})
    full_dataset = resampled_dataset
    logging.info(f"Class imbalance handled with SMOTE. Dataset size increased to {len(full_dataset)}.")
else:
    logging.info("Class imbalance handling not applied or not needed.")

# --- 2. Model Architecture ---

class MatchingDataset(Dataset):
    """Dataset for JD-Resume matching."""
    def __init__(self, data: List[Dict[str, str]], tokenizer: AutoTokenizer, max_len: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        jd = item['jd']
        resume = item['resume']
        label = item['label']

        # Option B: Cross-attention model with special tokens
        encoded = self.tokenizer(
            f"[JD] {jd} [RESUME] {resume}",
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label).long()
        }

class MatchingModel(pl.LightningModule):
    """PyTorch Lightning module for JD-Resume matching using a Transformer model."""
    def __init__(self, model_name: str, learning_rate: float = 2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'])
        loss = torch.nn.functional.cross_entropy(outputs.logits, batch['labels'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'])
        preds = torch.argmax(outputs.logits, dim=1)
        labels = batch['labels']
        self.accuracy(preds, labels)
        self.precision(preds, labels)
        self.recall(preds, labels)
        self.f1(preds, labels)
        self.log_dict({
            'val_accuracy': self.accuracy,
            'val_precision': self.precision,
            'val_recall': self.recall,
            'val_f1': self.f1
        })
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

class MatchingDataModule(pl.LightningDataModule):
    def __init__(self, train_data: List[Dict[str, str]], val_data: List[Dict[str, str]], tokenizer: AutoTokenizer, batch_size: int = 32, max_len: int = 512):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len

    def prepare_data(self):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") # Ensure tokenizer is downloaded

    def setup(self, stage=None):
        self.train_dataset = MatchingDataset(self.train_data, self.tokenizer, self.max_len)
        self.val_dataset = MatchingDataset(self.val_data, self.tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

# Split data
train_data, val_data = train_test_split(full_dataset, test_size=0.2, random_state=42, stratify=[item['label'] for item in full_dataset])

# Initialize tokenizer and data module
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
data_module = MatchingDataModule(train_data, val_data, tokenizer, batch_size=16, max_len=512)

# Initialize model and trainer
model = MatchingModel("roberta-base", learning_rate=2e-5)
checkpoint_callback = ModelCheckpoint(monitor='val_f1', mode='max', save_top_k=1, dirpath='./checkpoints')
trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback], accelerator='auto') # Use 'auto' for GPU if available

# Train the model
trainer.fit(model, data_module)

# Load the best checkpoint
best_model_path = checkpoint_callback.best_model_path
if best_model_path:
    trained_model = MatchingModel.load_from_checkpoint(best_model_path)
    trained_model.eval()
else:
    trained_model = model.eval()

# --- 3. Key Features ---

# Skill/Entity Extraction
def extract_skills_entities(text: str, nlp_model=nlp, english_words_set=english_words) -> List[str]:
    """Extracts skills and entities from text using Spacy and NLTK word list."""
    doc = nlp_model(text)
    skills = set()
    for ent in doc.ents:
        if ent.label_ in ["SKILL", "ORG", "PRODUCT", "TECHNOLOGY"] and ent.text.lower() in english_words_set:
            skills.add(ent.text.lower())
        elif ent.label_ in ["JOB"]: # Custom NER label if you train one
            skills.add(ent.text.lower())
    # Add noun chunks that might be skills
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3 and chunk.root.lemma_ not in nlp.vocab and chunk.text.lower() in english_words_set:
            skills.add(chunk.text.lower())
    return list(skills)

def explain_match(jd: str, resume: str, model: MatchingModel, tokenizer: AutoTokenizer, threshold: float = 0.5) -> Dict[str, any]:
    """Explains the match prediction by highlighting overlapping skills/entities."""
    jd_skills = extract_skills_entities(jd)
    resume_skills = extract_skills_entities(resume)
    common_skills = list(set(jd_skills) & set(resume_skills))

    encoded = tokenizer(f"[JD] {jd} [RESUME] {resume}", truncation=True, padding='max_length', max_length=512, return_tensors='pt')
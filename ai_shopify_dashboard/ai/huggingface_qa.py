from transformers import pipeline

# Initialize Hugging Face QA model (can be GPT, RoBERTa, etc.)
qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2')

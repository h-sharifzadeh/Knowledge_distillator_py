import os
import csv
import time
from groq import Groq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer  # Updated import
from datasets import Dataset
from dotenv import load_dotenv  # Added import

# Load environment variables from .env file
load_dotenv()  # Added line

# Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile" # Check latest model name
CSV_FILENAME = "qa_pairs.csv"
number_of_qa_pairs = 200
STUDENT_MODEL_NAME = "google/flan-t5-small"  # Student model for distillation

Topics="Psychology"


# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def generate_qa_pairs():
    qa_pairs = []
    prompt = f"""Generate a diverse set of {number_of_qa_pairs} question and answer pairs covering topics of {Topics}.
    Format should be:
    Q: [question]
    A: [answer]
    ---"""
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL_NAME,
        temperature=0.7,
        max_tokens=4096
    )

    # Parse the response
    content = response.choices[0].message.content
    pairs = content.split('---')
    
    for pair in pairs:
        lines = pair.strip().split('\n')
        if len(lines) >= 2:
            q = lines[0].replace('Q:', '').strip()
            a = lines[1].replace('A:', '').strip()
            if q and a:
                qa_pairs.append({"question": q, "answer": a})
    
    return qa_pairs[:number_of_qa_pairs]  # Ensure exactly number_of_qa_pairs pairs

def save_to_csv(qa_pairs):
    with open(CSV_FILENAME, 'w', newline='') as csvfile:
        fieldnames = ['question', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qa_pairs)

def distill_model():
    # Load dataset
    dataset = Dataset.from_csv(CSV_FILENAME)
    
    # Load student model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_MODEL_NAME)  # Updated model class
    
    # Tokenization function
    def tokenize_function(examples):
        inputs = tokenizer(
            ["question: " + q + " answer: " for q in examples["question"]],
            padding="max_length",
            truncation=True,
            max_length=128
        )
        labels = tokenizer(
            examples["answer"],
            padding="max_length",
            truncation=True,
            max_length=128
        ).input_ids
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels
        }
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        logging_dir="./logs",
        learning_rate=2e-5,
        push_to_hub=True,
        hub_token=HF_API_KEY
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train and push to Hub
    trainer.train()
    trainer.push_to_hub()

if __name__ == "__main__":
    # Generate and save QA pairs
    qa_pairs = generate_qa_pairs()
    save_to_csv(qa_pairs)
    
    # Distill model
    distill_model()

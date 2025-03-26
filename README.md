# Distillator Py

This project is a Python-based tool for generating question-answer (QA) pairs and fine-tuning a student language model using knowledge distillation. It leverages the Groq API for generating QA pairs and the Hugging Face Transformers library for model training.

## Features

- **QA Pair Generation**: Automatically generates 100 diverse QA pairs using the Groq API.
- **Save to CSV**: Saves the generated QA pairs to a CSV file.
- **Model Distillation**: Fine-tunes a student model (`google/flan-t5-small` by default) using the generated QA pairs.

## Requirements

- Python 3.8 or higher
- API keys for:
  - [Groq API](https://groq.com/)
  - [Hugging Face Hub](https://huggingface.co/)
- Required Python libraries:
  - `groq`
  - `transformers`
  - `datasets`
  - `csv`
  - `os`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/distillator_py.git
   cd distillator_py
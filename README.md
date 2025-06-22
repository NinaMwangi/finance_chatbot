# 💬 Finance QA Chatbot

A domain-specific question-answering chatbot built using a fine-tuned FLAN-T5 Transformer model on a financial Q&A dataset. This chatbot answers finance-related questions with accuracy and context awareness through a clean and interactive UI powered by Gradio.

## Dataset
- Name: financial-qa-10K
- Size: 10,000+ Q&A pairs
- Fields:
  - question: The finance-related question.
  - answer: The correct answer extracted from filings.
  - context: A supporting context sentence (used in training for better accuracy).
  - ticker: Company ticker (e.g., NVDA).
  - filing: Source filing (e.g., 2023_10K).

## Model Architecture
- Model Base: google/flan-t5-base
- Fine-Tuned On: Financial QA dataset
- Training Framework: TensorFlow / Keras
- Prompt Format: `Q: {question} Context: {context} A:`

## Performance Metrics
| Experiment   | Learning Rate | Batch Size | Epochs | Input Length | ROUGE-1 | BLEU   | Exact Match |
|--------------|---------------|------------|--------|--------------|---------|--------|-------------|
| Exp 5 (Final) | 3e-5          | 8          | 10     | 256          | 69.61%  | 43.75% | 25.43%      |


Metric Definitions:

- ROUGE-1: Measures the overlap of unigrams (words).
- BLEU: Captures how close generated responses are to human answers.
- Exact Match (EM): Measures how many answers exactly match the reference.

## How to run locally 
Requirements
`pip install gradio datasets transformers tensorflow`

Run the Chatbot
`python app.py`

On Notebooks
`interface.launch(share=True)`

## Deployment
- Model hosted on: Hugging Face Model Hub - https://huggingface.co/NinaMwangi/T5_finbot
- UI deployed via: Hugging Face Spaces using Gradio
- Link https://huggingface.co/spaces/NinaMwangi/Finance-chatbot

## Example Conversations
- User: What area did NVIDIA initially focus on before expanding to other computationally intensive fields?
  - Bot: NVIDIA initially focused on PC graphics.
- User: What was the net income of Apple in 2022?
  - Bot: The Net income of Apple in 2022 was $27 Billion.
- User: What was the amount of cash generated from operations by the company in Fiscal year 2023?
  - Bot: $18,085 million.

## Demo Link
https://drive.google.com/file/d/1-cGV_9A8aixESLD99gXj4PrqsRKA6ho3/view?usp=sharing

# 🧑‍💻 Author

Built with ❤️ by Nina Mwangi

import gradio as gr
print("Gradio version:", gr.__version__)
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

model_name = "NinaMwangi/T5_finbot"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("virattt/financial-qa-10K")["train"]

# Function to retrieve matching context
def get_context_for_question(question):
    for item in dataset:
        if item["question"].strip().lower() == question.strip().lower():
            return item["context"]
    return "No relevant context found."

# Define the prediction function (inference)
def generate_answer(question, chat_history):
    context = get_context_for_question(question)
    prompt = f"Q: {question} Context: {context} A:"


    inputs = tokenizer(
        prompt,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=256
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    chat_history.append((question, answer))
    return "", chat_history

with gr.Blocks(theme=gr.themes.Base()) as interface:
    gr.Markdown(
        """
        # 💬 Finance QA Chatbot
        Ask a finance-related question and get an accurate, concise response.
        Built using a fine-tuned T5 Transformer on financial Q&A data.
        """,
    )

    chatbot = gr.Chatbot(label="Finance Chatbot", height=400, bubble_full_width=False)
    with gr.Row():
        with gr.Column(scale=8):
            question_box = gr.Textbox(
                placeholder="Ask a finance question...", show_label=False, lines=2
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Send")

    clear_btn = gr.Button("Clear Chat")

    # Chat state
    state = gr.State([])

    # Bind function
    submit_btn.click(
        generate_answer,
        inputs=[question_box, state],
        outputs=[question_box, chatbot],
    )

    clear_btn.click(lambda: [], inputs=[], outputs=[chatbot, state])

# Run app
interface.launch(share=True)
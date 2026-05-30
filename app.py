import gradio as gr
import time
from transformers import MarianMTModel, MarianTokenizer

print("Loading AI Translation Models... Please wait.")

# English to Hindi
en_hi_model_name = "Helsinki-NLP/opus-mt-en-hi"
en_hi_tokenizer = MarianTokenizer.from_pretrained(en_hi_model_name)
en_hi_model = MarianMTModel.from_pretrained(en_hi_model_name)

# Hindi to English
hi_en_model_name = "Helsinki-NLP/opus-mt-hi-en"
hi_en_tokenizer = MarianTokenizer.from_pretrained(hi_en_model_name)
hi_en_model = MarianMTModel.from_pretrained(hi_en_model_name)

print("Models Loaded Successfully!")

def translate_text(text, direction, style):
    start = time.time()

    if not text.strip():
        return "Please enter text", 0, 0, "0 sec"

    # Style modification
    if style == "Formal":
        text = "Please translate formally: " + text
    elif style == "Casual":
        text = "Translate casually: " + text

    try:
        if direction == "English ➜ Hindi":
            tokenizer = en_hi_tokenizer
            model = en_hi_model
        else:
            tokenizer = hi_en_tokenizer
            model = hi_en_model

        inputs = tokenizer([text], return_tensors="pt", padding=True)

        translated = model.generate(**inputs)

        output = tokenizer.batch_decode(
            translated,
            skip_special_tokens=True
        )[0]

        end = time.time()

        words = len(text.split())
        chars = len(text)
        process_time = round(end - start, 2)

        return (
            output,
            words,
            chars,
            f"{process_time} sec"
        )

    except Exception as e:
        return f"Error: {str(e)}", 0, 0, "0 sec"

custom_css = """
body{
    background: linear-gradient(135deg,#0f172a,#1e293b,#111827);
}

.gradio-container{
    font-family: 'Poppins', sans-serif;
}

.main-box{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 25px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0px 8px 32px rgba(0,0,0,0.3);
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    background: linear-gradient(to right,#38bdf8,#818cf8,#c084fc);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    text-align:center;
    color:white;
    font-size:18px;
    margin-bottom:20px;
}

.stats{
    background: rgba(255,255,255,0.08);
    padding:15px;
    border-radius:18px;
    color:white;
    text-align:center;
}

textarea{
    border-radius:18px !important;
}

button{
    border-radius:14px !important;
    font-weight:bold !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
        <div class="title">🌍 AI Translator Pro</div>
        <div class="subtitle">
            Advanced English ↔ Hindi Translator
        </div>
    """)

    with gr.Column(elem_classes="main-box"):

        direction = gr.Radio(
            ["English ➜ Hindi", "Hindi ➜ English"],
            value="English ➜ Hindi",
            label="Translation Direction"
        )

        style = gr.Radio(
            ["Normal", "Formal", "Casual"],
            value="Normal",
            label="Translation Style"
        )

        input_text = gr.Textbox(
            lines=8,
            placeholder="Type your text here...",
            label="Input Text"
        )

        with gr.Row():
            translate_btn = gr.Button("🚀 Translate")
            clear_btn = gr.Button("🗑 Clear")

        output_text = gr.Textbox(
            lines=8,
            label="Translated Output"
        )

        copy_btn = gr.Button("📋 Copy Output")

        gr.Markdown("## 📊 Translation Analytics")

        with gr.Row():
            word_count = gr.Number(label="Word Count")
            char_count = gr.Number(label="Character Count")
            processing_time = gr.Textbox(label="Processing Time")

        translate_btn.click(
            fn=translate_text,
            inputs=[input_text, direction, style],
            outputs=[
                output_text,
                word_count,
                char_count,
                processing_time
            ]
        )

        clear_btn.click(
            fn=lambda: ("", "", 0, 0, ""),
            outputs=[
                input_text,
                output_text,
                word_count,
                char_count,
                processing_time
            ]
        )

demo.launch(share=True)
from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None   # global chat memory

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chat():
    global chat_history_ids

    user_input = request.form['msg']
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # If bot has no previous history
    if chat_history_ids is None:
        bot_input_ids = new_input_ids
    else:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)

    # Generate response
    attention_mask = (bot_input_ids != tokenizer.eos_token_id).long()

    chat_history_ids = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_p=0.95,
        top_k=60
    )


    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return reply

if __name__ == '__main__':
    app.run(debug=True)

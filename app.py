import os
from datetime import datetime, timedelta
import random
from functools import partial
import gradio as gr
from huggingface_hub import InferenceClient
import threading

css = """
gradio-app {
    background: none !important;
}

.md .container {
    border:1px solid #ccc; 
    border-radius:5px; 
    min-height:300px;
    color: #666;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    font-family: monospace;
    padding: 10px;
}

#hf_token_box {
    transition: height 1s ease-out, opacity 1s ease-out;
}

#hf_token_box.abc {
    height: 0;
    opacity: 0;
    overflow: hidden;
}

#generate_button {
    transition: background-color 1s ease-out, color 1s ease-out; border-color 1s ease-out;
}

#generate_button.changed {
    background: black !important;
    border-color: black !important; 
    color: white !important;
}
"""

js = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') === 'dark') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

system_prompt = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

code = """
```python
from huggingface_hub import InferenceClient

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
PROMPT = "{PROMPT}"
MODEL_NAME = "meta-llama/Meta-Llama-3-70b-Instruct"  # or "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO" or "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}, 
    {"role": "user", "content": PROMPT}
]
client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)
for c in client.chat_completion(messages, max_tokens=200, stream=True):
    token = c.choices[0].delta.content
    print(token, end="")
```
"""

ip_requests = {}
ip_requests_lock = threading.Lock()

def allow_ip(request: gr.Request, show_error=True):
    ip = request.headers.get("X-Forwarded-For")
    now = datetime.now()
    window = timedelta(hours=24)
    with ip_requests_lock:
        if ip in ip_requests:
            ip_requests[ip] = [timestamp for timestamp in ip_requests[ip] if now - timestamp < window]
        if len(ip_requests.get(ip, [])) >= 15:
            raise gr.Error("Rate limit exceeded. Please try again tomorrow or use your Hugging Face Pro token.", visible=show_error)
        ip_requests.setdefault(ip, []).append(now)
        print("ip_requests", ip_requests)
    return True

def inference(prompt, hf_token, model, model_name):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    if hf_token is None or not hf_token.strip():
        hf_token = os.getenv("HF_TOKEN")
    client = InferenceClient(model=model, token=hf_token)
    tokens = f"**`{model_name}`**\n\n"
    for completion in client.chat_completion(messages, max_tokens=200, stream=True):
        token = completion.choices[0].delta.content
        tokens += token
        yield tokens

def random_prompt():
    return random.choice([
        "Give me 5 very different ways to say the following sentence: 'The quick brown fox jumps over the lazy dog.'",
        "Write a summary of the plot of the movie 'Inception' using only emojis.",
        "Write a sentence with the words 'serendipity', 'baguette', and 'C++'.",
        "Explain the concept of 'quantum entanglement' to a 5-year-old.",
        "Write a couplet about Python"
    ])

with gr.Blocks(css=css, theme="NoCrypt/miku", js=js) as demo:
    gr.Markdown("<center><h1>🔮 Open LLM Explorer</h1></center>")
    gr.Markdown("Every LLM has its own personality! Type your prompt below and compare results from the 3 leading open models from the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) that are on the Hugging Face Inference API. You can sign up for [Hugging Face Pro](https://huggingface.co/pricing#pro) and get a token to avoid rate limits.")
    prompt = gr.Textbox(random_prompt, lines=2, show_label=False, info="Type your prompt here.")
    hf_token_box = gr.Textbox(lines=1, placeholder="Your Hugging Face token (not required, but a HF Pro account avoids rate limits):", show_label=False, elem_id="hf_token_box", type="password")
    with gr.Group():
        with gr.Row():
            generate_btn = gr.Button(value="Generate", elem_id="generate_button", variant="primary", size="sm")
            code_btn = gr.Button(value="View Code", elem_id="code_button", variant="secondary", size="sm")

    with gr.Row() as output_row:
        llama_output = gr.Markdown("<div class='container'>Llama 3-70B Instruct</div>", elem_classes=["md"], height=300)
        nous_output = gr.Markdown("<div class='container'>Nous Hermes 2 Mixtral 8x7B DPO</div>", elem_classes=["md"], height=300)
        zephyr_output = gr.Markdown("<div class='container'>Zephyr ORPO 141B A35B</div>", elem_classes=["md"], height=300)

    with gr.Row(visible=False) as code_row:
        code_display = gr.Markdown(code, elem_classes=["md"], height=300)

    output_visible = gr.State(True)
    code_btn.click(
        lambda x: (not x, gr.Row(visible=not x), gr.Row(visible=x), "View Results" if x else "View Code"),
        output_visible,
        [output_visible, output_row, code_row, code_btn],
        api_name=False,
    )

    false = gr.State(False)

    gr.on(
        [prompt.submit, generate_btn.click],
        None,
        None, 
        None,
        api_name=False,
        js="""
            function disappear() {
            var element = document.getElementById("hf_token_box");
            var height = element.offsetHeight;
            var step = height / 30; // Adjust this value to change the speed of disappearance
            var padding_top = parseFloat(getComputedStyle(element).paddingTop); // Get initial padding
            var padding_bottom = parseFloat(getComputedStyle(element).paddingBottom); // Get initial padding
            var step_padding = padding_top / 30; // Adjust this value to change the speed of disappearance

            var interval = setInterval(function() {
                if (height > 0) {
                    height -= step;
                    element.style.height = height + "px";
                    padding_bottom -= step_padding;
                    element.style.paddingBottom = padding_bottom + "px";
                    console.log("height", height);
                } else {
                    clearInterval(interval);
                }
            }, 20); // Adjust this value to change the smoothness of the animation
            }
        """
    )    

    gr.on(
        [prompt.submit, generate_btn.click],
        allow_ip,
        false,
    ).success(
        partial(inference, model="meta-llama/Meta-Llama-3-70b-Instruct", model_name="Llama 3-70B Instruct"),
        [prompt, hf_token_box],
        llama_output,
        show_progress="hidden",
        api_name=False
    )

    gr.on(
        [prompt.submit, generate_btn.click],
        allow_ip,
        false,
    ).success(
        partial(inference, model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", model_name="Nous Hermes 2 Mixtral 8x7B DPO"),
        [prompt, hf_token_box],
        nous_output,
        show_progress="hidden",
        api_name=False
    )

    gr.on(
        [prompt.submit, generate_btn.click],
        allow_ip,
    ).success(
        partial(inference, model="HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1", model_name="Zephyr ORPO 141B A35B"),
        [prompt, hf_token_box],
        zephyr_output,
        show_progress="hidden",
        api_name=False
    )

    gr.on(
        triggers=[prompt.submit, generate_btn.click],
        fn=lambda x: (code.replace("{PROMPT}", x), True, gr.Row(visible=True), gr.Row(visible=False), "View Code"),
        inputs=[prompt],
        outputs=[code_display, output_visible, output_row, code_row, code_btn],
        api_name=False
    )


demo.launch(show_api=False)



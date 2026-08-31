import ast
import json
import operator
import os
import re
import threading
from urllib.parse import quote

from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context
from gpt4all import GPT4All


HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))
MODEL_PATH = os.environ.get("MODEL_PATH") or os.environ.get("MODEL_NAME") or "orca-mini-3b-gguf2-q4_0.gguf"
MODEL_LOCK = threading.Lock()
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "500"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "8"))


def resolve_model_path(model_name):
    candidate = os.path.expanduser(model_name).strip()
    if not candidate:
        return None

    if os.path.isfile(candidate):
        return candidate

    candidates = [
        os.path.join(os.getcwd(), candidate),
        os.path.join(os.getcwd(), "models", candidate),
        os.path.join("/", "models", candidate),
        os.path.join("/", "workspace", candidate),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    return candidate


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

    if isinstance(node, ast.BinOp):
        operation = OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported operator")
        return operation(
            safe_eval(node.left),
            safe_eval(node.right),
        )

    if isinstance(node, ast.UnaryOp):
        operation = OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported operator")
        return operation(safe_eval(node.operand))

    raise ValueError("Unsupported expression")


def calculate(expression):
    try:
        expression = expression.replace("^", "**")
        tree = ast.parse(expression, mode="eval")
        return str(safe_eval(tree.body))
    except Exception as exc:
        return f"Calculator error: {exc}"


def make_image_url(prompt):
    encoded_prompt = quote(prompt, safe="")
    return (
        "https://image.pollinations.ai/prompt/"
        + encoded_prompt
        + "?width=768"
        + "&height=768"
        + "&nologo=true"
    )


SYSTEM_PROMPT = """
You are Nova AI, a helpful local AI assistant.

You can:
- answer questions
- write Python scripts
- write HTML, CSS and JavaScript
- write other programming languages
- explain code
- debug programs
- perform calculations
- generate image prompts for the application's image tool

Always put generated code inside a properly closed Markdown code block.

Never leave a code block open.

Do not claim to have searched the web unless the application actually provides search results.

Do not claim that you created or downloaded a file.

When the user asks to generate an image, respond with exactly:

[IMAGE: description]

Do not say that you cannot create Python scripts.
"""


class NovaAgent:
    def __init__(self):
        resolved_path = resolve_model_path(MODEL_PATH)
        if resolved_path is None or not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                "Model file not found. Set MODEL_PATH to a valid .gguf file path or upload the model into the workspace. "
                f"Tried: {MODEL_PATH}"
            )

        print(f"Loading Nova AI model from {resolved_path}...")
        self.model = GPT4All(resolved_path)
        print("Nova AI model loaded!")

    def wants_image(self, text):
        text = text.lower()
        phrases = [
            "generate an image",
            "generate image",
            "create an image",
            "make an image",
            "draw an image",
            "generate a picture",
            "create a picture",
            "make a picture",
            "generate a logo",
            "create a logo",
            "make a logo",
            "draw a logo",
        ]
        return any(phrase in text for phrase in phrases)

    def extract_image_prompt(self, text):
        patterns = [
            r"generate\s+(?:an?\s+)?image\s*(?:of|for)?\s*",
            r"create\s+(?:an?\s+)?image\s*(?:of|for)?\s*",
            r"make\s+(?:an?\s+)?image\s*(?:of|for)?\s*",
            r"draw\s+(?:an?\s+)?image\s*(?:of|for)?\s*",
            r"generate\s+(?:an?\s+)?picture\s*(?:of|for)?\s*",
            r"create\s+(?:an?\s+)?picture\s*(?:of|for)?\s*",
            r"make\s+(?:an?\s+)?picture\s*(?:of|for)?\s*",
            r"generate\s+(?:a\s+)?logo\s*(?:of|for)?\s*",
            r"create\s+(?:a\s+)?logo\s*(?:of|for)?\s*",
            r"make\s+(?:a\s+)?logo\s*(?:of|for)?\s*",
            r"draw\s+(?:a\s+)?logo\s*(?:of|for)?\s*",
        ]

        prompt = text.strip()
        for pattern in patterns:
            new_prompt = re.sub(pattern, "", prompt, count=1, flags=re.IGNORECASE)
            if new_prompt != prompt:
                prompt = new_prompt
                break
        return prompt.strip() or text.strip()

    def extract_math(self, text):
        match = re.search(
            r"(?:calculate|compute|what is)\s+([0-9+\-*/().%^ ]+)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None

        expression = match.group(1).strip()
        if re.fullmatch(r"[0-9+\-*/().%^ ]+", expression):
            return expression
        return None

    def chat(self, conversation):
        latest_user_message = ""
        for role, message in reversed(conversation):
            if role == "user":
                latest_user_message = message
                break

        if self.wants_image(latest_user_message):
            prompt = self.extract_image_prompt(latest_user_message)
            return f"[IMAGE: {prompt}]"

        tool_information = ""
        expression = self.extract_math(latest_user_message)
        if expression:
            result = calculate(expression)
            tool_information += f"\n\nCalculator result:\n{expression} = {result}\n"

        prompt = SYSTEM_PROMPT + "\n\nConversation:\n"
        recent_conversation = conversation[-MAX_HISTORY_MESSAGES:]
        for role, message in recent_conversation:
            name = "User" if role == "user" else "Nova"
            prompt += f"{name}: {message}\n"

        if tool_information:
            prompt += tool_information

        prompt += "\nNova:"

        try:
            with MODEL_LOCK:
                result = self.model.generate(
                    prompt,
                    max_tokens=MAX_TOKENS,
                    temp=TEMPERATURE,
                )
            return result.strip()
        except Exception as exc:
            return "Nova encountered an error:\n" + str(exc)


try:
    NOVA = NovaAgent()
except Exception as exc:
    NOVA = None
    print("FAILED TO LOAD NOVA:")
    print(str(exc))


app = Flask(__name__)


@app.get("/")
def index():
    return render_template_string("""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Nova AI</title>
        <style>
            :root {
                --bg: #0b1020;
                --panel: #121a2b;
                --panel-2: #1b2438;
                --text: #edf2ff;
                --muted: #a9b7d2;
                --accent: #7cc7ff;
                --accent-2: #7ef0c9;
                --border: #2a3853;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                background: var(--bg);
                color: var(--text);
                font-family: Arial, sans-serif;
            }
            .wrap {
                max-width: 980px;
                margin: 32px auto;
                padding: 20px;
            }
            .panel {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 18px;
                box-shadow: 0 16px 40px rgba(0,0,0,0.25);
            }
            h1 {
                margin: 0 0 10px;
                font-size: 2rem;
            }
            .status {
                color: var(--muted);
                margin-bottom: 14px;
            }
            textarea {
                width: 100%;
                min-height: 120px;
                background: var(--panel-2);
                border: 1px solid var(--border);
                border-radius: 10px;
                color: var(--text);
                padding: 14px;
                font-size: 1rem;
                resize: vertical;
            }
            .row {
                display: flex;
                gap: 12px;
                margin-top: 16px;
                flex-wrap: wrap;
            }
            button {
                background: linear-gradient(135deg, var(--accent), var(--accent-2));
                color: #08111d;
                border: none;
                border-radius: 10px;
                padding: 12px 18px;
                font-weight: 700;
                cursor: pointer;
            }
            button.secondary {
                background: transparent;
                color: var(--text);
                border: 1px solid var(--border);
            }
            .chat {
                margin-top: 24px;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            .bubble {
                background: var(--panel-2);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 14px 16px;
                white-space: pre-wrap;
                line-height: 1.5;
            }
            .bubble.user {
                background: #172236;
            }
            .bubble img {
                display: block;
                max-width: 100%;
                border-radius: 12px;
                margin-top: 10px;
                border: 1px solid var(--border);
            }
            .loading {
                color: var(--muted);
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-size: 0.9rem;
            }
            .typing-dots {
                display: inline-flex;
                gap: 4px;
                align-items: center;
                height: 16px;
            }
            .typing-dots span {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: var(--accent);
                display: block;
                animation: bounce 1.2s infinite ease-in-out;
            }
            .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
            .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
            @keyframes bounce {
                0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
                40% { transform: translateY(-4px); opacity: 1; }
            }
        </style>
    </head>
    <body>
        <div class="wrap">
            <div class="panel">
                <h1>Nova AI</h1>
                <div class="status" id="status">Ready</div>
                <textarea id="prompt" placeholder="Ask Nova AI anything...">Hello</textarea>
                <div class="row">
                    <button id="sendBtn">Send</button>
                    <button class="secondary" id="clearBtn" type="button">Clear</button>
                </div>
            </div>

            <div class="chat" id="chat"></div>
        </div>

        <script>
            const chat = document.getElementById('chat');
            const prompt = document.getElementById('prompt');
            const status = document.getElementById('status');
            const sendBtn = document.getElementById('sendBtn');
            const clearBtn = document.getElementById('clearBtn');

            const history = [];

            function addBubble(role, text) {
                const el = document.createElement('div');
                el.className = 'bubble ' + role;
                el.textContent = text;
                chat.appendChild(el);
                window.scrollTo(0, document.body.scrollHeight);
                return el;
            }

            function addImage(url, altText) {
                const el = document.createElement('div');
                el.className = 'bubble';
                const img = document.createElement('img');
                img.src = url;
                img.alt = altText;
                el.appendChild(img);
                chat.appendChild(el);
                window.scrollTo(0, document.body.scrollHeight);
            }

            function createLoadingBubble() {
                const el = document.createElement('div');
                el.className = 'bubble assistant';
                const loader = document.createElement('div');
                loader.className = 'loading';
                loader.innerHTML = '<span>Writing</span><span class="typing-dots"><span></span><span></span><span></span></span>';
                el.appendChild(loader);
                chat.appendChild(el);
                window.scrollTo(0, document.body.scrollHeight);
                return el;
            }

            function typeTextIntoBubble(el, text) {
                el.textContent = '';
                let i = 0;
                const interval = setInterval(() => {
                    if (i >= text.length) {
                        clearInterval(interval);
                        status.textContent = 'Ready';
                        sendBtn.disabled = false;
                        prompt.focus();
                        return;
                    }

                    el.textContent += text[i];
                    i += 1;
                    window.scrollTo(0, document.body.scrollHeight);
                }, 18);
            }

            async function sendMessage() {
                const text = prompt.value.trim();
                if (!text) return;

                history.push(['user', text]);
                addBubble('user', text);
                prompt.value = '';
                status.textContent = 'Thinking...';
                sendBtn.disabled = true;

                const typingBubble = createLoadingBubble();

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ conversation: history })
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.error || 'Request failed');
                    }

                    if (data.type === 'image') {
                        typingBubble.remove();
                        addBubble('assistant', data.prompt || 'Generated image');
                        addImage(data.image_url, data.prompt || 'Generated image');
                        history.push(['assistant', data.response || '']);
                        status.textContent = 'Ready';
                        sendBtn.disabled = false;
                        prompt.focus();
                        return;
                    }

                    const responseText = data.response || 'No response';
                    history.push(['assistant', responseText]);
                    typingBubble.textContent = '';
                    typeTextIntoBubble(typingBubble, responseText);
                } catch (error) {
                    typingBubble.textContent = 'Error: ' + error.message;
                    status.textContent = 'Error';
                    history.push(['assistant', 'Error: ' + error.message]);
                    sendBtn.disabled = false;
                    prompt.focus();
                }
            }

            sendBtn.addEventListener('click', sendMessage);
            clearBtn.addEventListener('click', () => {
                chat.innerHTML = '';
                history.length = 0;
                status.textContent = 'Ready';
            });

            prompt.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "nova_loaded": NOVA is not None,
        "model_path": resolve_model_path(MODEL_PATH),
        "port": PORT,
    })


def normalize_conversation(payload):
    data = payload or {}
    conversation = data.get("conversation")
    if conversation is None:
        message = data.get("message", "")
        if not isinstance(message, str):
            raise ValueError("Missing 'message' or 'conversation'")
        conversation = [("user", message)]

    if not isinstance(conversation, list):
        raise ValueError("'conversation' must be a list")

    cleaned = []
    for item in conversation:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each conversation item must contain role and message")

        role = str(item[0])
        message = str(item[1])
        if role not in ("user", "assistant"):
            raise ValueError("Role must be 'user' or 'assistant'")
        cleaned.append((role, message))

    if not cleaned:
        raise ValueError("Conversation is empty")

    return cleaned


@app.post("/api/chat")
def api_chat():
    if NOVA is None:
        return jsonify({"error": "Nova AI model is not loaded."}), 503

    try:
        conversation = normalize_conversation(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        response = NOVA.chat(conversation)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    image_match = re.search(r"\[IMAGE:\s*(.*?)\]", response, re.DOTALL)
    if image_match:
        prompt_text = image_match.group(1).strip()
        return jsonify({
            "response": response,
            "type": "image",
            "prompt": prompt_text,
            "image_url": make_image_url(prompt_text),
        })

    return jsonify({"response": response, "type": "text"})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False, threaded=True)

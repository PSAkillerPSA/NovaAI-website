import ast
import json
import operator
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

from gpt4all import GPT4All


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "0.0.0.0"

# FeatherPanel/Quaxly usually provides PORT.
# Fall back to 8000 if it doesn't.
PORT = int(os.environ.get("PORT", "8000"))

MODEL_NAME = "orca-mini-3b-gguf2-q4_0.gguf"

# Prevent multiple simultaneous generations from fighting
# over the same local GPT4All model.
MODEL_LOCK = threading.Lock()


# ============================================================
# SAFE CALCULATOR
# ============================================================

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
            safe_eval(node.right)
        )

    if isinstance(node, ast.UnaryOp):
        operation = OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operator")

        return operation(
            safe_eval(node.operand)
        )

    raise ValueError("Unsupported expression")


def calculate(expression):
    try:
        expression = expression.replace("^", "**")

        tree = ast.parse(
            expression,
            mode="eval"
        )

        return str(
            safe_eval(tree.body)
        )

    except Exception as exc:
        return f"Calculator error: {exc}"


# ============================================================
# IMAGE GENERATION
# ============================================================

def make_image_url(prompt):
    encoded_prompt = quote(
        prompt,
        safe=""
    )

    return (
        "https://image.pollinations.ai/prompt/"
        + encoded_prompt
        + "?width=768"
        + "&height=768"
        + "&nologo=true"
    )


# ============================================================
# PYTHON VALIDATION
# ============================================================

def validate_python(code):
    forbidden = [
        "open(",
        "os.remove",
        "os.unlink",
        "os.rename",
        "shutil.",
        "socket.",
        "requests.post",
        "requests.put",
        "requests.delete",
        "urllib.request.urlopen",
    ]

    lowered = code.lower()

    for item in forbidden:
        if item.lower() in lowered:
            return (
                False,
                "Preview blocked because the script contains "
                f"a restricted operation: {item}"
            )

    return True, ""


# ============================================================
# NOVA AI
# ============================================================

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
        print("Loading Nova AI model...")

        self.model = GPT4All(
            MODEL_NAME
        )

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

        return any(
            phrase in text
            for phrase in phrases
        )


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

            new_prompt = re.sub(
                pattern,
                "",
                prompt,
                count=1,
                flags=re.IGNORECASE
            )

            if new_prompt != prompt:
                prompt = new_prompt
                break

        return (
            prompt.strip()
            or text.strip()
        )


    def extract_math(self, text):
        match = re.search(
            r"(?:calculate|compute|what is)\s+"
            r"([0-9+\-*/().%^ ]+)",
            text,
            re.IGNORECASE
        )

        if not match:
            return None

        expression = match.group(1).strip()

        if re.fullmatch(
            r"[0-9+\-*/().%^ ]+",
            expression
        ):
            return expression

        return None


    def chat(self, conversation):

        latest_user_message = ""

        for role, message in reversed(conversation):

            if role == "user":
                latest_user_message = message
                break


        # ----------------------------------------------------
        # IMAGE REQUEST
        # ----------------------------------------------------

        if self.wants_image(
            latest_user_message
        ):

            prompt = self.extract_image_prompt(
                latest_user_message
            )

            return f"[IMAGE: {prompt}]"


        # ----------------------------------------------------
        # CALCULATOR
        # ----------------------------------------------------

        tool_information = ""

        expression = self.extract_math(
            latest_user_message
        )

        if expression:

            result = calculate(
                expression
            )

            tool_information += (
                "\n\nCalculator result:\n"
                f"{expression} = {result}\n"
            )


        # ----------------------------------------------------
        # BUILD PROMPT
        # ----------------------------------------------------

        prompt = SYSTEM_PROMPT

        prompt += "\n\nConversation:\n"


        # Keep the same 30-message limit
        # as the original NovaAI6.
        for role, message in conversation[-30:]:

            name = (
                "User"
                if role == "user"
                else "Nova"
            )

            prompt += (
                f"{name}: {message}\n"
            )


        if tool_information:
            prompt += tool_information


        prompt += "\nNova:"


        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        try:

            with MODEL_LOCK:

                result = self.model.generate(
                    prompt,
                    max_tokens=1500,
                    temp=0.3
                )

            return result.strip()

        except Exception as exc:

            return (
                "Nova encountered an error:\n"
                + str(exc)
            )


# ============================================================
# LOAD NOVA
# ============================================================

try:

    NOVA = NovaAgent()

except Exception as exc:

    NOVA = None

    print(
        "FAILED TO LOAD NOVA:"
    )

    print(
        str(exc)
    )


# ============================================================
# HTTP SERVER
# ============================================================

class NovaRequestHandler(BaseHTTPRequestHandler):


    def log_message(
        self,
        format,
        *args
    ):
        print(
            f"[HTTP] {self.address_string()} "
            + format % args
        )


    def send_json(
        self,
        status,
        data
    ):

        body = json.dumps(
            data,
            ensure_ascii=False
        ).encode("utf-8")


        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        # Allow your website to communicate
        # with the Quaxly API.
        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "POST, GET, OPTIONS"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )

        self.end_headers()

        self.wfile.write(
            body
        )


    def do_OPTIONS(self):

        self.send_response(
            204
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "POST, GET, OPTIONS"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )

        self.end_headers()


    def do_GET(self):

        if self.path == "/":

            self.send_json(
                200,
                {
                    "name": "Nova AI",
                    "status": "online",
                    "service": "NovaAI API"
                }
            )

            return


        if self.path == "/health":

            self.send_json(
                200,
                {
                    "status": "ok",
                    "nova_loaded": NOVA is not None
                }
            )

            return


        self.send_json(
            404,
            {
                "error": "Not found"
            }
        )


    def do_POST(self):

        if self.path != "/chat":

            self.send_json(
                404,
                {
                    "error": "Not found"
                }
            )

            return


        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

        except ValueError:

            self.send_json(
                400,
                {
                    "error": "Invalid Content-Length"
                }
            )

            return


        if content_length <= 0:

            self.send_json(
                400,
                {
                    "error": "Empty request body"
                }
            )

            return


        try:

            raw_body = self.rfile.read(
                content_length
            )

            data = json.loads(
                raw_body.decode("utf-8")
            )

        except Exception as exc:

            self.send_json(
                400,
                {
                    "error": (
                        "Invalid JSON: "
                        + str(exc)
                    )
                }
            )

            return


        if NOVA is None:

            self.send_json(
                503,
                {
                    "error": "Nova AI model is not loaded."
                }
            )

            return


        # ----------------------------------------------------
        # ACCEPT EITHER:
        #
        # {
        #   "message": "Hello"
        # }
        #
        # OR:
        #
        # {
        #   "conversation": [
        #       ["user", "Hello"]
        #   ]
        # }
        # ----------------------------------------------------

        conversation = data.get(
            "conversation"
        )


        if conversation is None:

            message = data.get(
                "message"
            )

            if not isinstance(
                message,
                str
            ):

                self.send_json(
                    400,
                    {
                        "error":
                        "Missing 'message' or 'conversation'"
                    }
                )

                return


            conversation = [
                (
                    "user",
                    message
                )
            ]


        # Convert JSON arrays into tuples.
        cleaned_conversation = []

        if not isinstance(
            conversation,
            list
        ):

            self.send_json(
                400,
                {
                    "error":
                    "'conversation' must be a list"
                }
            )

            return


        for item in conversation:

            if (
                not isinstance(
                    item,
                    (list, tuple)
                )
                or len(item) != 2
            ):

                self.send_json(
                    400,
                    {
                        "error":
                        "Each conversation item "
                        "must contain role and message"
                    }
                )

                return


            role = str(
                item[0]
            )

            message = str(
                item[1]
            )


            if role not in (
                "user",
                "assistant"
            ):

                self.send_json(
                    400,
                    {
                        "error":
                        "Role must be "
                        "'user' or 'assistant'"
                    }
                )

                return


            cleaned_conversation.append(
                (
                    role,
                    message
                )
            )


        if not cleaned_conversation:

            self.send_json(
                400,
                {
                    "error":
                    "Conversation is empty"
                }
            )

            return


        print(
            "Nova received:",
            cleaned_conversation[-1][1]
        )


        # ----------------------------------------------------
        # RUN NOVA
        # ----------------------------------------------------

        try:

            response = NOVA.chat(
                cleaned_conversation
            )

        except Exception as exc:

            self.send_json(
                500,
                {
                    "error":
                    str(exc)
                }
            )

            return


        # ----------------------------------------------------
        # IMAGE RESPONSE
        # ----------------------------------------------------

        image_match = re.search(
            r"\[IMAGE:\s*(.*?)\]",
            response,
            re.DOTALL
        )


        if image_match:

            prompt = (
                image_match
                .group(1)
                .strip()
            )

            image_url = make_image_url(
                prompt
            )

            self.send_json(
                200,
                {
                    "response": response,
                    "type": "image",
                    "prompt": prompt,
                    "image_url": image_url
                }
            )

            return


        # ----------------------------------------------------
        # NORMAL RESPONSE
        # ----------------------------------------------------

        self.send_json(
            200,
            {
                "response": response,
                "type": "text"
            }
        )


# ============================================================
# START SERVER
# ============================================================

def main():

    print()
    print("================================")
    print("        NOVA AI SERVER")
    print("================================")
    print()
    print(
        f"Listening on 0.0.0.0:{PORT}"
    )
    print()
    print(
        "Endpoints:"
    )
    print(
        "  GET  /"
    )
    print(
        "  GET  /health"
    )
    print(
        "  POST /chat"
    )
    print()


    server = ThreadingHTTPServer(
        (
            HOST,
            PORT
        ),
        NovaRequestHandler
    )


    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "\nStopping Nova AI server..."
        )

    finally:

        server.server_close()


if __name__ == "__main__":
    main()
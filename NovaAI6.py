import sys
import ast
import operator
import subprocess
import webbrowser
from urllib.parse import quote
import re

from gpt4all import GPT4All

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QListWidget,
    QSplitter,
    QScrollArea,
    QFrame,
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except ImportError:
    QWebEngineView = None
    WEBENGINE_AVAILABLE = False


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


class ImageWorker(QThread):
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        try:
            self.finished.emit(make_image_url(self.prompt))
        except Exception as exc:
            self.failed.emit(str(exc))


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


class PythonPreviewWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, code):
        super().__init__()
        self.code = code
        self.process = None
        self.stop_requested = False

    def run(self):
        allowed, reason = validate_python(self.code)

        if not allowed:
            self.finished.emit(reason)
            return

        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    self.code
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True
            )

            try:
                output, _ = self.process.communicate(timeout=10)

                if self.stop_requested:
                    result = (
                        "Preview stopped.\n"
                        + (output or "")
                    )
                elif output.strip():
                    result = output.rstrip()
                elif self.process.returncode == 0:
                    result = (
                        "Script finished successfully "
                        "with no console output."
                    )
                else:
                    result = (
                        "Script exited with code "
                        f"{self.process.returncode}."
                    )

            except subprocess.TimeoutExpired:
                self.process.kill()
                output, _ = self.process.communicate()

                result = (
                    "Preview stopped because the "
                    "10-second limit was reached.\n"
                    + (output or "")
                )

            self.finished.emit(result)

        except Exception as exc:
            self.finished.emit(f"Preview error: {exc}")

    def stop(self):
        self.stop_requested = True

        if (
            self.process is not None
            and self.process.poll() is None
        ):
            try:
                self.process.kill()
            except Exception:
                pass


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

The application displays generated images through an embedded browser view.

When the user asks to generate an image, respond with exactly:

[IMAGE: description]

Do not say that you cannot create Python scripts.
"""


class NovaAgent:
    def __init__(self):
        self.model = GPT4All(
            "orca-mini-3b-gguf2-q4_0.gguf"
        )

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

        return prompt.strip() or text.strip()

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

        if self.wants_image(latest_user_message):
            prompt = self.extract_image_prompt(
                latest_user_message
            )
            return f"[IMAGE: {prompt}]"

        tool_information = ""

        expression = self.extract_math(
            latest_user_message
        )

        if expression:
            result = calculate(expression)

            tool_information += (
                "\n\nCalculator result:\n"
                f"{expression} = {result}\n"
            )

        prompt = SYSTEM_PROMPT
        prompt += "\n\nConversation:\n"

        for role, message in conversation[-30:]:
            name = (
                "User"
                if role == "user"
                else "Nova"
            )

            prompt += f"{name}: {message}\n"

        if tool_information:
            prompt += tool_information

        prompt += "\nNova:"

        try:
            result = self.model.generate(
                prompt,
                max_tokens=1500,
                temp=0.3
            )

            return result.strip()

        except Exception as exc:
            return "Nova encountered an error:\n" + str(exc)


class AIWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, agent, conversation):
        super().__init__()
        self.agent = agent
        self.conversation = list(conversation)

    def run(self):
        try:
            response = self.agent.chat(
                self.conversation
            )
        except Exception as exc:
            response = f"Error:\n{exc}"

        self.finished.emit(response)


class CodeBlock(QFrame):
    def __init__(
        self,
        language,
        code,
        theme,
        parent=None
    ):
        super().__init__(parent)

        self.language = (
            language.strip().lower()
            if language
            else "text"
        )

        self.code = code
        self.theme = theme
        self.preview_worker = None

        self.setObjectName("CodeBlock")

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            10,
            8,
            10,
            10
        )

        layout.setSpacing(7)

        header = QHBoxLayout()

        language_label = QLabel(
            self.language.upper()
        )

        language_label.setFont(
            QFont(
                "Menlo",
                9,
                QFont.Weight.Bold
            )
        )

        header.addWidget(language_label)
        header.addStretch()

        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(self.copy_code)
        header.addWidget(copy_button)

        if self.language in ("python", "py"):
            self.preview_button = QPushButton(
                "Preview"
            )

            self.preview_button.clicked.connect(
                self.start_preview
            )

            header.addWidget(
                self.preview_button
            )

            self.stop_button = QPushButton(
                "Stop Preview"
            )

            self.stop_button.setEnabled(False)

            self.stop_button.clicked.connect(
                self.stop_preview
            )

            header.addWidget(
                self.stop_button
            )

        download_button = QPushButton(
            "Download"
        )

        download_button.setEnabled(False)

        download_button.setToolTip(
            "Disabled: Nova does not save "
            "generated files to your computer."
        )

        header.addWidget(download_button)

        layout.addLayout(header)

        self.editor = QTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setPlainText(code)
        self.editor.setFont(QFont("Menlo", 10))

        lines = max(
            1,
            code.count("\n") + 1
        )

        self.editor.setMinimumHeight(
            min(
                500,
                max(
                    100,
                    35 + lines * 18
                )
            )
        )

        layout.addWidget(self.editor)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Menlo", 9))
        self.output.setPlaceholderText(
            "Python preview output..."
        )
        self.output.setMinimumHeight(100)
        self.output.hide()

        layout.addWidget(self.output)

        self.status = QLabel("")
        layout.addWidget(self.status)

        self.apply_theme()

    def copy_code(self):
        QApplication.clipboard().setText(
            self.code
        )

        self.status.setText(
            "Code copied to clipboard."
        )

    def start_preview(self):
        if (
            self.preview_worker
            and self.preview_worker.isRunning()
        ):
            return

        self.output.clear()
        self.output.show()

        self.preview_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.status.setText(
            "Running preview..."
        )

        self.preview_worker = PythonPreviewWorker(
            self.code
        )

        self.preview_worker.finished.connect(
            self.preview_finished
        )

        self.preview_worker.start()

    def stop_preview(self):
        if self.preview_worker:
            self.preview_worker.stop()

        self.stop_button.setEnabled(False)
        self.preview_button.setEnabled(True)

        self.status.setText(
            "Stopping preview..."
        )

    def preview_finished(self, output):
        self.output.setPlainText(output)

        self.preview_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.status.setText(
            "Preview finished."
        )

    def apply_theme(self):
        if self.theme == "light":
            background = "#f0f3f7"
            border = "#ccd4de"
            editor_background = "#ffffff"
            text = "#202832"
            hover = "#e3e8ee"
        else:
            background = "#11161d"
            border = "#303844"
            editor_background = "#0a0d12"
            text = "#e7edf4"
            hover = "#29323d"

        self.setStyleSheet(
            f"""
            QFrame#CodeBlock {{
                background: {background};
                border: 1px solid {border};
                border-radius: 10px;
            }}

            QLabel {{
                color: {text};
                border: none;
            }}

            QPushButton {{
                background: {background};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 5px 9px;
            }}

            QPushButton:hover {{
                background: {hover};
            }}

            QPushButton:disabled {{
                color: #7a8490;
            }}

            QTextEdit {{
                background: {editor_background};
                color: {text};
                border: none;
                border-radius: 6px;
                padding: 8px;
            }}
            """
        )


class ImageMessage(QFrame):
    def __init__(
        self,
        url,
        prompt,
        theme,
        parent=None
    ):
        super().__init__(parent)

        self.url = url
        self.prompt = prompt
        self.theme = theme

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            10,
            10,
            10,
            10
        )

        title = QLabel("Generated Image")

        title.setFont(
            QFont(
                "Segoe UI",
                11,
                QFont.Weight.Bold
            )
        )

        layout.addWidget(title)

        prompt_label = QLabel(prompt)
        prompt_label.setWordWrap(True)
        layout.addWidget(prompt_label)

        if WEBENGINE_AVAILABLE:
            self.browser = QWebEngineView()

            self.browser.setMinimumHeight(560)

            self.browser.setUrl(
                QUrl(url)
            )

            layout.addWidget(
                self.browser
            )
        else:
            message = QLabel(
                "The embedded image view requires "
                "PyQt6-WebEngine.\n\n"
                "Install it with:\n"
                "pip3 install PyQt6-WebEngine"
            )

            message.setWordWrap(True)
            layout.addWidget(message)

        buttons = QHBoxLayout()

        open_button = QPushButton(
            "Open in Browser"
        )

        open_button.clicked.connect(
            lambda: webbrowser.open(self.url)
        )

        buttons.addWidget(open_button)
        buttons.addStretch()

        layout.addLayout(buttons)

        self.apply_theme()

    def apply_theme(self):
        if self.theme == "light":
            background = "#ffffff"
            border = "#ccd4de"
            text = "#202832"
        else:
            background = "#11161d"
            border = "#303844"
            text = "#e7edf4"

        self.setStyleSheet(
            f"""
            QFrame {{
                background: {background};
                color: {text};
                border: 1px solid {border};
                border-radius: 10px;
            }}

            QLabel {{
                color: {text};
                border: none;
            }}

            QPushButton {{
                background: {background};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 6px 10px;
            }}

            QPushButton:hover {{
                background: {border};
            }}
            """
        )


class MessageWidget(QFrame):
    def __init__(
        self,
        sender,
        text,
        theme,
        parent=None
    ):
        super().__init__(parent)

        self.sender = sender
        self.theme = theme

        self.setObjectName(
            "UserMessage"
            if sender == "You"
            else "NovaMessage"
        )

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            14,
            12,
            14,
            12
        )

        layout.setSpacing(8)

        title = QLabel(sender)

        title.setFont(
            QFont(
                "Segoe UI",
                9,
                QFont.Weight.Bold
            )
        )

        layout.addWidget(title)

        if sender == "You":
            self.add_markdown(
                layout,
                text
            )
        else:
            self.render_nova(
                layout,
                text
            )

        self.apply_theme()

    def add_markdown(self, layout, text):
        widget = QTextEdit()

        widget.setReadOnly(True)

        widget.setFrameStyle(
            QFrame.Shape.NoFrame
        )

        widget.setFont(
            QFont(
                "Segoe UI",
                11
            )
        )

        widget.setMarkdown(text)

        widget.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        widget.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        lines = max(
            1,
            text.count("\n") + 1
        )

        widget.setFixedHeight(
            max(
                42,
                min(
                    700,
                    27 * lines + 25
                )
            )
        )

        layout.addWidget(widget)

    def render_nova(self, layout, text):
        pattern = re.compile(
            r"```([A-Za-z0-9_+#.-]*)"
            r"[ \t]*\r?\n"
            r"(.*?)```",
            re.DOTALL
        )

        position = 0
        found_code = False

        for match in pattern.finditer(text):
            found_code = True

            before = text[
                position:
                match.start()
            ]

            if before.strip():
                self.add_markdown(
                    layout,
                    before
                )

            language = (
                match.group(1)
                or "text"
            )

            code = match.group(2)

            code_widget = CodeBlock(
                language,
                code,
                self.theme
            )

            layout.addWidget(code_widget)

            position = match.end()

        remaining = text[position:]

        if remaining.strip() or not found_code:
            self.add_markdown(
                layout,
                remaining
            )

    def apply_theme(self):
        if self.theme == "light":
            user_bg = "#e7f0fb"
            nova_bg = "#ffffff"
            border = "#d1d9e2"
            text = "#202832"
            user_name = "#276da8"
            nova_name = "#18734a"
        else:
            user_bg = "#1d2735"
            nova_bg = "#171c24"
            border = "#303844"
            text = "#edf2f7"
            user_name = "#8fc8ff"
            nova_name = "#9ee6bb"

        background = (
            user_bg
            if self.sender == "You"
            else nova_bg
        )

        self.setStyleSheet(
            f"""
            QFrame#{self.objectName()} {{
                background: {background};
                border: 1px solid {border};
                border-radius: 12px;
            }}

            QLabel {{
                color: {text};
                border: none;
            }}

            QTextEdit {{
                background: transparent;
                color: {text};
                border: none;
            }}
            """
        )

        title = (
            self.layout()
            .itemAt(0)
            .widget()
        )

        if isinstance(title, QLabel):
            name_color = (
                user_name
                if self.sender == "You"
                else nova_name
            )

            title.setStyleSheet(
                f"color: {name_color};"
            )


class NovaWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Nova AI")

        self.resize(
            1200,
            820
        )

        self.theme = "dark"

        self.agent = None
        self.ai_worker = None
        self.image_worker = None

        self.busy = False

        self.chats = []
        self.current_chat = -1

        self.build_ui()

        self.new_chat()

        self.status.setText(
            "Loading local AI model..."
        )

        QApplication.processEvents()

        try:
            self.agent = NovaAgent()

            self.status.setText("Ready")

        except Exception as exc:
            self.status.setText(
                "Model error"
            )

            self.add_system_message(
                "Could not load the local model:\n"
                + str(exc)
            )

    def build_ui(self):
        central = QWidget()

        root = QVBoxLayout(central)

        root.setContentsMargins(
            0,
            0,
            0,
            0
        )

        self.setCentralWidget(central)

        splitter = QSplitter(
            Qt.Orientation.Horizontal
        )

        root.addWidget(splitter)

        sidebar = QWidget()

        sidebar.setMinimumWidth(230)
        sidebar.setMaximumWidth(310)

        side_layout = QVBoxLayout(sidebar)

        side_layout.setContentsMargins(
            12,
            14,
            12,
            12
        )

        logo = QLabel("✦ NOVA AI")

        logo.setFont(
            QFont(
                "Segoe UI",
                18,
                QFont.Weight.Bold
            )
        )

        side_layout.addWidget(logo)

        subtitle = QLabel(
            "Local AI assistant"
        )

        side_layout.addWidget(subtitle)

        self.new_chat_button = QPushButton(
            "＋  New Chat"
        )

        self.new_chat_button.clicked.connect(
            self.new_chat
        )

        side_layout.addWidget(
            self.new_chat_button
        )

        self.chat_list = QListWidget()

        self.chat_list.currentRowChanged.connect(
            self.switch_chat
        )

        side_layout.addWidget(
            self.chat_list,
            1
        )

        privacy = QLabel(
            "🔒 Privacy\n\n"
            "Generated scripts are kept in "
            "memory only.\n\n"
            "No temporary .py file is created "
            "for Python previews.\n\n"
            "Images are displayed in the "
            "embedded browser."
        )

        privacy.setWordWrap(True)

        side_layout.addWidget(privacy)

        splitter.addWidget(sidebar)

        main = QWidget()

        main_layout = QVBoxLayout(main)

        main_layout.setContentsMargins(
            14,
            12,
            14,
            10
        )

        header = QHBoxLayout()

        title = QLabel("Nova AI")

        title.setFont(
            QFont(
                "Segoe UI",
                17,
                QFont.Weight.Bold
            )
        )

        header.addWidget(title)
        header.addStretch()

        self.status = QLabel("Ready")

        header.addWidget(self.status)

        self.theme_button = QPushButton(
            "☀ Light"
        )

        self.theme_button.clicked.connect(
            self.toggle_theme
        )

        header.addWidget(
            self.theme_button
        )

        main_layout.addLayout(header)

        self.scroll = QScrollArea()

        self.scroll.setWidgetResizable(True)

        self.scroll.setFrameShape(
            QFrame.Shape.NoFrame
        )

        self.messages_widget = QWidget()

        self.messages_layout = QVBoxLayout(
            self.messages_widget
        )

        self.messages_layout.setContentsMargins(
            8,
            8,
            8,
            8
        )

        self.messages_layout.setSpacing(12)

        self.messages_layout.addStretch()

        self.scroll.setWidget(
            self.messages_widget
        )

        main_layout.addWidget(
            self.scroll,
            1
        )

        options = QHBoxLayout()

        bold = QPushButton("B")

        bold.clicked.connect(
            lambda: self.wrap_selection("**")
        )

        options.addWidget(bold)

        italic = QPushButton("I")

        italic.clicked.connect(
            lambda: self.wrap_selection("*")
        )

        options.addWidget(italic)

        code = QPushButton("</>")

        code.clicked.connect(
            lambda: self.wrap_selection("`")
        )

        options.addWidget(code)

        bullet = QPushButton("• List")

        bullet.clicked.connect(
            self.add_bullet
        )

        options.addWidget(bullet)

        number = QPushButton("1. List")

        number.clicked.connect(
            self.add_number
        )

        options.addWidget(number)

        quote = QPushButton("Quote")

        quote.clicked.connect(
            self.add_quote
        )

        options.addWidget(quote)

        options.addStretch()

        main_layout.addLayout(options)

        self.input = QTextEdit()

        self.input.setPlaceholderText(
            "Message Nova..."
        )

        self.input.setFont(
            QFont(
                "Segoe UI",
                11
            )
        )

        self.input.setFixedHeight(95)

        self.input.installEventFilter(self)

        main_layout.addWidget(self.input)

        send_bar = QHBoxLayout()

        self.info = QLabel(
            "Enter = send • "
            "Shift+Enter = new line • "
            "One message at a time"
        )

        send_bar.addWidget(self.info)
        send_bar.addStretch()

        self.send_button = QPushButton("Send")

        self.send_button.setFont(
            QFont(
                "Segoe UI",
                10,
                QFont.Weight.Bold
            )
        )

        self.send_button.clicked.connect(
            self.send_message
        )

        send_bar.addWidget(
            self.send_button
        )

        main_layout.addLayout(send_bar)

        splitter.addWidget(main)

        splitter.setStretchFactor(
            1,
            1
        )

        self.apply_theme()

    def wrap_selection(self, marker):
        cursor = self.input.textCursor()

        selected = cursor.selectedText()

        if selected:
            cursor.insertText(
                marker
                + selected
                + marker
            )
        else:
            cursor.insertText(
                marker
                + marker
            )

            cursor.movePosition(
                QTextCursor.MoveOperation.Left,
                QTextCursor.MoveMode.MoveAnchor,
                len(marker)
            )

            self.input.setTextCursor(cursor)

        self.input.setFocus()

    def add_bullet(self):
        cursor = self.input.textCursor()

        cursor.movePosition(
            QTextCursor.MoveOperation.StartOfLine
        )

        cursor.insertText("- ")

        self.input.setTextCursor(cursor)
        self.input.setFocus()

    def add_number(self):
        cursor = self.input.textCursor()

        cursor.movePosition(
            QTextCursor.MoveOperation.StartOfLine
        )

        cursor.insertText("1. ")

        self.input.setTextCursor(cursor)
        self.input.setFocus()

    def add_quote(self):
        cursor = self.input.textCursor()

        cursor.movePosition(
            QTextCursor.MoveOperation.StartOfLine
        )

        cursor.insertText("> ")

        self.input.setTextCursor(cursor)
        self.input.setFocus()

    def eventFilter(self, obj, event):
        if (
            obj is self.input
            and event.type()
            == event.Type.KeyPress
        ):
            if event.key() in (
                Qt.Key.Key_Return,
                Qt.Key.Key_Enter
            ):
                if event.modifiers() & (
                    Qt.KeyboardModifier.ShiftModifier
                ):
                    return False

                self.send_message()

                return True

        return super().eventFilter(
            obj,
            event
        )

    def new_chat(self):
        if self.busy:
            return

        chat = {
            "title": "New Chat",
            "messages": [
                (
                    "assistant",
                    "Hi! I'm Nova AI. "
                    "How can I help?"
                )
            ]
        }

        self.chats.append(chat)

        self.chat_list.blockSignals(True)

        self.chat_list.addItem(
            chat["title"]
        )

        self.current_chat = (
            len(self.chats) - 1
        )

        self.chat_list.setCurrentRow(
            self.current_chat
        )

        self.chat_list.blockSignals(False)

        self.rebuild_messages()

    def switch_chat(self, row):
        if self.busy:
            self.chat_list.blockSignals(True)

            if self.current_chat >= 0:
                self.chat_list.setCurrentRow(
                    self.current_chat
                )

            self.chat_list.blockSignals(False)

            return

        if (
            row < 0
            or row >= len(self.chats)
        ):
            return

        self.current_chat = row

        self.rebuild_messages()

    def rename_current_chat(self, message):
        if self.current_chat < 0:
            return

        chat = self.chats[
            self.current_chat
        ]

        if chat["title"] != "New Chat":
            return

        title = re.sub(
            r"\s+",
            " ",
            message
        ).strip()

        if len(title) > 30:
            title = title[:30] + "..."

        if not title:
            title = "New Chat"

        chat["title"] = title

        item = self.chat_list.item(
            self.current_chat
        )

        if item:
            item.setText(title)

    def rebuild_messages(self):
        while self.messages_layout.count():
            item = self.messages_layout.takeAt(0)

            widget = item.widget()

            if widget:
                widget.deleteLater()

        if self.current_chat < 0:
            return

        messages = self.chats[
            self.current_chat
        ]["messages"]

        for role, message in messages:
            if role == "user":
                widget = MessageWidget(
                    "You",
                    message,
                    self.theme
                )
            else:
                widget = MessageWidget(
                    "Nova AI",
                    message,
                    self.theme
                )

            self.messages_layout.addWidget(widget)

        self.messages_layout.addStretch()

        QApplication.processEvents()

        self.scroll_to_bottom()

    def send_message(self):
        if self.busy:
            return

        if self.current_chat < 0:
            return

        text = self.input.toPlainText().strip()

        if not text:
            return

        self.busy = True

        self.input.setEnabled(False)
        self.send_button.setEnabled(False)
        self.new_chat_button.setEnabled(False)

        self.status.setText("Thinking...")

        self.rename_current_chat(text)

        self.chats[
            self.current_chat
        ]["messages"].append(
            (
                "user",
                text
            )
        )

        self.input.clear()

        self.rebuild_messages()

        if self.agent is None:
            self.finish_request(
                "The AI model is not loaded."
            )
            return

        conversation = list(
            self.chats[
                self.current_chat
            ]["messages"]
        )

        self.ai_worker = AIWorker(
            self.agent,
            conversation
        )

        self.ai_worker.finished.connect(
            self.ai_finished
        )

        self.ai_worker.start()

    def ai_finished(self, response):
        if self.current_chat < 0:
            self.finish_request()
            return

        self.chats[
            self.current_chat
        ]["messages"].append(
            (
                "assistant",
                response
            )
        )

        self.rebuild_messages()

        image_match = re.search(
            r"\[IMAGE:\s*(.*?)\]",
            response,
            re.DOTALL
        )

        if image_match:
            prompt = (
                image_match.group(1).strip()
            )

            self.status.setText(
                "Opening embedded image preview..."
            )

            self.image_worker = ImageWorker(
                prompt
            )

            self.image_worker.finished.connect(
                lambda url:
                self.image_finished(
                    url,
                    prompt
                )
            )

            self.image_worker.failed.connect(
                self.image_failed
            )

            self.image_worker.start()
        else:
            self.finish_request()

    def image_finished(self, url, prompt):
        image_widget = ImageMessage(
            url,
            prompt,
            self.theme
        )

        if self.current_chat >= 0:
            messages = self.chats[
                self.current_chat
            ]["messages"]

            if messages:
                role, message = messages[-1]

                if (
                    role == "assistant"
                    and message.startswith("[IMAGE:")
                ):
                    messages.pop()

        self.rebuild_messages()

        self.messages_layout.insertWidget(
            self.messages_layout.count() - 1,
            image_widget
        )

        self.scroll_to_bottom()

        self.finish_request()

    def image_failed(self, error):
        self.add_system_message(
            "Image preview error:\n"
            + error
        )

        self.finish_request()

    def finish_request(self, message=None):
        if message:
            self.add_system_message(message)

        self.busy = False

        self.input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.new_chat_button.setEnabled(True)

        self.status.setText("Ready")

        self.input.setFocus()

    def add_system_message(self, message):
        if self.current_chat < 0:
            return

        widget = MessageWidget(
            "Nova AI",
            message,
            self.theme
        )

        index = max(
            0,
            self.messages_layout.count() - 1
        )

        self.messages_layout.insertWidget(
            index,
            widget
        )

        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        QApplication.processEvents()

        scrollbar = (
            self.scroll.verticalScrollBar()
        )

        scrollbar.setValue(
            scrollbar.maximum()
        )

    def toggle_theme(self):
        if self.theme == "dark":
            self.theme = "light"
        else:
            self.theme = "dark"

        self.apply_theme()
        self.rebuild_messages()

    def apply_theme(self):
        if self.theme == "light":
            self.setStyleSheet(
                """
                QMainWindow, QWidget {
                    background: #f5f7fa;
                    color: #202832;
                }

                QListWidget {
                    background: #e9edf2;
                    border: none;
                    color: #26313d;
                    padding: 8px;
                }

                QListWidget::item {
                    padding: 10px;
                    border-radius: 7px;
                    margin: 2px 0;
                }

                QListWidget::item:selected {
                    background: #d5e3f2;
                    color: #17212b;
                }

                QPushButton {
                    background: #ffffff;
                    color: #26313d;
                    border: 1px solid #c8d1dc;
                    border-radius: 7px;
                    padding: 7px 11px;
                }

                QPushButton:hover {
                    background: #e8edf3;
                }

                QPushButton:disabled {
                    color: #929ba6;
                    background: #edf0f3;
                }

                QTextEdit {
                    background: #ffffff;
                    color: #202832;
                    border: 1px solid #cbd3dd;
                    border-radius: 10px;
                    padding: 10px;
                }

                QLabel {
                    color: #26313d;
                }
                """
            )

            self.theme_button.setText("☾ Dark")

        else:
            self.setStyleSheet(
                """
                QMainWindow, QWidget {
                    background: #0c0f14;
                    color: #edf2f7;
                }

                QListWidget {
                    background: #10141a;
                    border: none;
                    color: #dfe6ee;
                    padding: 8px;
                }

                QListWidget::item {
                    padding: 10px;
                    border-radius: 7px;
                    margin: 2px 0;
                }

                QListWidget::item:selected {
                    background: #263344;
                    color: white;
                }

                QPushButton {
                    background: #202731;
                    color: #edf2f7;
                    border: 1px solid #343e4a;
                    border-radius: 7px;
                    padding: 7px 11px;
                }

                QPushButton:hover {
                    background: #2a3441;
                }

                QPushButton:disabled {
                    color: #626c78;
                    background: #171b21;
                }

                QTextEdit {
                    background: #12171e;
                    color: #edf2f7;
                    border: 1px solid #303844;
                    border-radius: 10px;
                    padding: 10px;
                }

                QLabel {
                    color: #edf2f7;
                }
                """
            )

            self.theme_button.setText("☀ Light")

        self.info.setText(
            "Enter = send • "
            "Shift+Enter = new line • "
            "One message at a time"
        )


def main():
    app = QApplication(sys.argv)

    app.setApplicationName("Nova AI")

    window = NovaWindow()

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

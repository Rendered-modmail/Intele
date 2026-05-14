import os
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context
from groq import Groq
from tavily import TavilyClient


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

app = Flask(__name__)

SYSTEM_PROMPT = (
    "You are a professional essayist. Use the provided search context to write a "
    "detailed response. Use inline citations like [1] and [2] and list all source "
    "URLs at the end. If the context doesn't contain the answer, say so."
)

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Research Assistant</title>
  <style>
    :root {
      --bg: #ffffff;
      --sidebar: #f7f7f8;
      --border: #e5e5e5;
      --text: #111111;
      --muted: #6b7280;
      --soft: #f4f4f5;
      --button: #111111;
      --button-text: #ffffff;
      --accent: #10a37f;
      --shadow: 0 18px 50px rgba(0, 0, 0, 0.08);
      --quick: 160ms cubic-bezier(.2,.8,.2,1);
      --smooth: 520ms cubic-bezier(.2,.8,.2,1);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      height: 100vh;
      overflow: hidden;
    }

    .app {
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      height: 100vh;
      animation: appIn 500ms ease both;
    }

    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 12px;
      background: var(--sidebar);
      border-right: 1px solid var(--border);
      padding: 10px 8px;
      min-width: 0;
      animation: slideRight 520ms cubic-bezier(.2,.8,.2,1) both;
    }

    .brand {
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 36px;
      padding: 0 8px;
      font-size: 14px;
      font-weight: 650;
    }

    .mark {
      display: inline-grid;
      place-items: center;
      width: 28px;
      height: 28px;
      border: 1px solid var(--border);
      border-radius: 50%;
      background: #fff;
      font-size: 14px;
      animation: softPop 420ms 120ms cubic-bezier(.2,.8,.2,1) both;
    }

    .nav {
      display: grid;
      gap: 2px;
    }

    .nav button,
    .new-chat {
      display: flex;
      align-items: center;
      gap: 10px;
      width: 100%;
      min-height: 36px;
      border: 0;
      border-radius: 8px;
      background: transparent;
      color: var(--text);
      cursor: pointer;
      padding: 8px 10px;
      font: inherit;
      text-align: left;
      transform: translateZ(0);
      transition: background var(--quick), transform var(--quick);
    }

    .new-chat {
      background: #ededee;
    }

    .nav button:hover,
    .new-chat:hover {
      background: #ececee;
      transform: translateX(3px);
    }

    .icon {
      flex: 0 0 auto;
      width: 18px;
      height: 18px;
      display: inline-grid;
      place-items: center;
      color: #141414;
    }

    .sidebar-bottom {
      margin-top: auto;
      display: grid;
      gap: 2px;
    }

    .sidebar-note {
      margin: 8px -8px -10px;
      border-top: 1px solid var(--border);
      padding: 18px 14px 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    .sidebar-note strong {
      display: block;
      margin-bottom: 8px;
      color: var(--text);
      font-size: 13px;
    }

    .main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-width: 0;
      height: 100vh;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 52px;
      padding: 8px 18px 8px 28px;
    }

    .title {
      font-weight: 650;
      font-size: 17px;
    }

    .model {
      color: var(--muted);
      font-size: 13px;
      margin-left: 8px;
    }

    .auth {
      display: flex;
      gap: 10px;
    }

    .pill {
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 999px;
      min-height: 36px;
      padding: 0 14px;
      font-weight: 650;
      cursor: pointer;
      transition: transform var(--quick), box-shadow var(--quick), background var(--quick);
    }

    .pill:hover {
      transform: translateY(-1px);
      box-shadow: 0 8px 22px rgba(0, 0, 0, 0.08);
    }

    .pill.primary {
      border-color: #000;
      background: #000;
      color: #fff;
    }

    .conversation {
      overflow-y: auto;
      padding: 24px 24px 120px;
      scroll-behavior: smooth;
    }

    .empty {
      min-height: 66vh;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 24px;
      animation: fadeUp 620ms 120ms var(--smooth) both;
    }

    .empty h1 {
      margin: 0;
      font-size: clamp(24px, 3vw, 32px);
      font-weight: 500;
      letter-spacing: 0;
      animation: breathe 4.8s ease-in-out infinite;
    }

    .messages {
      width: min(860px, 100%);
      margin: 0 auto;
      display: grid;
      gap: 24px;
    }

    .message {
      display: grid;
      gap: 10px;
      line-height: 1.68;
      font-size: 16px;
      white-space: pre-wrap;
      animation: messageIn 300ms var(--quick) both;
    }

    .message.user {
      justify-items: end;
    }

    .bubble {
      max-width: min(680px, 100%);
      border-radius: 18px;
      padding: 12px 16px;
      background: var(--soft);
      transition: transform var(--quick), box-shadow var(--quick);
    }

    .message.user .bubble {
      animation: userBubble 280ms var(--quick) both;
    }

    .assistant .bubble {
      background: transparent;
      padding: 0;
      max-width: 100%;
    }

    .assistant.thinking .bubble::after {
      content: "";
      display: inline-block;
      width: 6px;
      height: 1em;
      margin-left: 2px;
      border-radius: 99px;
      background: var(--accent);
      vertical-align: -0.12em;
      animation: cursorPulse 900ms ease-in-out infinite;
    }

    .meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }

    .composer-wrap {
      position: fixed;
      left: 260px;
      right: 0;
      bottom: 0;
      padding: 20px 24px 24px;
      background: linear-gradient(180deg, rgba(255,255,255,0), #fff 34%);
      animation: fadeUp 520ms 160ms var(--smooth) both;
    }

    .composer {
      width: min(780px, 100%);
      margin: 0 auto;
      display: flex;
      align-items: end;
      gap: 10px;
      border: 1px solid #d7d7d9;
      border-radius: 28px;
      background: #fff;
      padding: 8px 8px 8px 14px;
      box-shadow: var(--shadow);
      transition: border-color var(--quick), box-shadow var(--quick), transform var(--quick);
    }

    .composer:focus-within {
      border-color: #b8b8bb;
      box-shadow: 0 22px 60px rgba(0, 0, 0, 0.11);
      transform: translateY(-1px);
    }

    .tool-btn,
    .send {
      flex: 0 0 auto;
      display: inline-grid;
      place-items: center;
      width: 34px;
      height: 34px;
      border: 0;
      border-radius: 50%;
      cursor: pointer;
      font-size: 20px;
      transition: transform var(--quick), background var(--quick), opacity var(--quick);
    }

    .tool-btn:hover,
    .send:hover:not(:disabled) {
      transform: scale(1.06);
    }

    .tool-btn {
      background: transparent;
      color: #222;
    }

    .send {
      background: var(--button);
      color: var(--button-text);
      font-size: 17px;
    }

    .send:disabled {
      background: #d7d7d9;
      cursor: not-allowed;
      animation: softPulse 1s ease-in-out infinite;
    }

    textarea {
      flex: 1;
      min-height: 38px;
      max-height: 180px;
      border: 0;
      outline: 0;
      resize: none;
      padding: 9px 2px;
      font: inherit;
      line-height: 1.35;
    }

    .status {
      width: min(780px, 100%);
      margin: 8px auto 0;
      min-height: 18px;
      color: var(--muted);
      font-size: 12px;
      padding-left: 18px;
    }

    .status.active::after {
      content: "";
      display: inline-block;
      width: 18px;
      height: 3px;
      margin-left: 8px;
      border-radius: 999px;
      background: linear-gradient(90deg, transparent, var(--accent), transparent);
      animation: statusSweep 1s ease-in-out infinite;
    }

    @keyframes appIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }

    @keyframes slideRight {
      from { opacity: 0; transform: translateX(-14px); }
      to { opacity: 1; transform: translateX(0); }
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(16px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes messageIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @keyframes userBubble {
      from { transform: scale(.98); }
      to { transform: scale(1); }
    }

    @keyframes softPop {
      from { opacity: 0; transform: scale(.86); }
      to { opacity: 1; transform: scale(1); }
    }

    @keyframes breathe {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-3px); }
    }

    @keyframes cursorPulse {
      0%, 100% { opacity: .25; }
      50% { opacity: 1; }
    }

    @keyframes softPulse {
      0%, 100% { opacity: .62; }
      50% { opacity: 1; }
    }

    @keyframes statusSweep {
      0% { transform: translateX(-6px); opacity: .35; }
      50% { opacity: 1; }
      100% { transform: translateX(6px); opacity: .35; }
    }

    @media (prefers-reduced-motion: reduce) {
      *,
      *::before,
      *::after {
        animation-duration: 1ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 1ms !important;
      }
    }

    @media (max-width: 760px) {
      .app {
        grid-template-columns: 1fr;
      }

      .sidebar {
        display: none;
      }

      .composer-wrap {
        left: 0;
      }

      .topbar {
        padding-left: 18px;
      }

      .auth {
        display: none;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <span class="mark">AI</span>
        <span class="icon">[]</span>
      </div>
      <button class="new-chat" id="newChat"><span class="icon">+</span>New chat</button>
      <nav class="nav">
        <button type="button"><span class="icon">/</span>Search chats</button>
        <button type="button"><span class="icon">E</span>Essays</button>
        <button type="button"><span class="icon">R</span>Research</button>
        <button type="button"><span class="icon">*</span>Deep research</button>
      </nav>
      <div class="sidebar-bottom nav">
        <button type="button"><span class="icon">S</span>Sources</button>
        <button type="button"><span class="icon">:</span>Settings</button>
        <button type="button"><span class="icon">?</span>Help</button>
        <div class="sidebar-note">
          <strong>Daily research chat</strong>
          Ask anything. Write an essay. BryanAI is still in beta, expect problems.
        </div>
      </div>
    </aside>

    <main class="main">
      <header class="topbar">
        <div>
          <span class="title"InteleAI</span>
          <span class="model">{{ model }}</span>
        </div>
        <div class="auth">
          <button class="pill primary" type="button">Log in</button>
          <button class="pill" type="button">Sign up for free</button>
        </div>
      </header>

      <section class="conversation" id="conversation">
        <div class="empty" id="empty">
          <h1>What's on your mind today?</h1>
        </div>
        <div class="messages" id="messages"></div>
      </section>

      <div class="composer-wrap">
        <form class="composer" id="chatForm">
          <button class="tool-btn" type="button" title="Add">+</button>
          <textarea id="query" rows="1" placeholder="Ask anything"></textarea>
          <button class="send" id="send" type="submit" title="Send">^</button>
        </form>
        <div class="status" id="status"></div>
      </div>
    </main>
  </div>

  <script>
    const form = document.getElementById("chatForm");
    const query = document.getElementById("query");
    const messages = document.getElementById("messages");
    const empty = document.getElementById("empty");
    const status = document.getElementById("status");
    const send = document.getElementById("send");
    const conversation = document.getElementById("conversation");
    const newChat = document.getElementById("newChat");

    function autoSize() {
      query.style.height = "auto";
      query.style.height = `${Math.min(query.scrollHeight, 180)}px`;
    }

    function scrollToBottom() {
      conversation.scrollTop = conversation.scrollHeight;
    }

    function addMessage(role, text = "") {
      empty.style.display = "none";
      const row = document.createElement("div");
      row.className = `message ${role}`;
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;
      row.appendChild(bubble);
      messages.appendChild(row);
      scrollToBottom();
      return { row, bubble };
    }

    function addMeta(text) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = text;
      messages.appendChild(meta);
      scrollToBottom();
    }

    function setStatus(text) {
      status.textContent = text;
      status.classList.toggle("active", Boolean(text));
    }

    function makeWordStreamer(target) {
      const queue = [];
      let running = false;

      function pump() {
        const next = queue.shift();
        if (next) {
          target.textContent += next;
          scrollToBottom();
          window.setTimeout(pump, next.trim() ? 18 : 6);
          return;
        }

        running = false;
      }

      return {
        push(text) {
          const tokens = text.match(/\\S+\\s*|\\s+/g) || [];
          queue.push(...tokens);
          if (!running) {
            running = true;
            pump();
          }
        },
        flush() {
          while (queue.length) {
            target.textContent += queue.shift();
          }
          scrollToBottom();
          running = false;
        }
      };
    }

    query.addEventListener("input", autoSize);
    query.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    newChat.addEventListener("click", () => {
      messages.innerHTML = "";
      empty.style.display = "grid";
      setStatus("");
      query.value = "";
      autoSize();
      query.focus();
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = query.value.trim();
      if (!text) return;

      addMessage("user", text);
      query.value = "";
      autoSize();
      send.disabled = true;
      setStatus("Researching with Tavily...");

      const assistant = addMessage("assistant thinking", "");
      const streamer = makeWordStreamer(assistant.bubble);

      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: text })
        });

        if (!response.ok || !response.body) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.error || "The assistant could not start.");
        }

        setStatus("Writing with Groq...");
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          streamer.push(decoder.decode(value, { stream: true }));
        }

        streamer.flush();
        assistant.row.classList.remove("thinking");
        addMeta(`Completed ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`);
        setStatus("");
      } catch (error) {
        assistant.row.classList.remove("thinking");
        assistant.bubble.textContent = `Error: ${error.message}`;
        setStatus("");
      } finally {
        send.disabled = false;
        query.focus();
      }
    });

    query.focus();
  </script>
</body>
</html>
"""


def require_keys():
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    return missing


def tavily_search(query):
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    return tavily.search(
        query=query,
        search_depth="advanced",
        include_raw_content=True,
        max_results=3,
    )


def build_search_context(search_response):
    results = search_response.get("results", [])
    context_blocks = []

    for index, result in enumerate(results, start=1):
        title = result.get("title") or "Untitled source"
        url = result.get("url") or "No URL"
        content = result.get("raw_content") or result.get("content") or ""
        content = " ".join(content.split())
        if len(content) > 6000:
            content = content[:6000] + "..."

        context_blocks.append(
            f"[{index}] Title: {title}\nURL: {url}\nContent: {content}"
        )

    if not context_blocks:
        return "No Tavily search results were returned."

    return "\n\n".join(context_blocks)


def groq_messages(query, context):
    today = datetime.now().strftime("%B %d, %Y")
    user_prompt = (
        f"Today's date: {today}\n\n"
        f"User query:\n{query}\n\n"
        f"Search context:\n{context}\n\n"
        "Write the answer now. Use only the search context for factual claims that "
        "need citations, cite with bracketed numbers, and end with a Sources section "
        "containing every source URL you used."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


@app.get("/")
def index():
    return render_template_string(HTML, model=GROQ_MODEL)


@app.get("/health")
def health():
    missing = require_keys()
    return jsonify({"ok": not missing, "missing": missing, "model": GROQ_MODEL})


@app.post("/chat")
def chat():
    missing = require_keys()
    if missing:
        return jsonify({"error": f"Missing environment variable(s): {', '.join(missing)}"}), 400

    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Please enter a question."}), 400

    def generate():
        try:
            search_response = tavily_search(query)
            context = build_search_context(search_response)

            client = Groq(api_key=GROQ_API_KEY)
            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=groq_messages(query, context),
                temperature=0.4,
                max_tokens=4096,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            yield f"\n\nError: {exc}"

    return Response(stream_with_context(generate()), mimetype="text/plain")


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=True, threaded=True)

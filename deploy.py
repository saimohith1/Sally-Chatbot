import os
import csv
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./qwen2-faq-finetuned"
LOG_FILE = "conversation_logs.csv"
REPORT_FILE = "daily_report.txt"
SUMMARY_LENGTH = 250

# --- THEME CSS: Clean, Bubbly, Customer Service Inspired ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
body, .stApp {
    background: #f4f8fb;
    font-family: 'Inter', sans-serif;
}
.sally-header {
    background: #fff;
    border-radius: 32px 32px 0 0;
    padding: 32px 32px 18px 32px;
    box-shadow: 0 6px 24px rgba(100, 160, 255, 0.07);
    margin-bottom: 0;
}
.sally-title-row {
    display: flex;
    align-items: center;
    gap: 16px;
}
.sally-avatar {
    width: 48px;
    height: 48px;
    border-radius: 16px;
    background: linear-gradient(135deg, #b6e0fe 0%, #ffe0f7 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(127, 187, 255, 0.13);
}
.sally-avatar img {
    width: 28px;
    height: 28px;
}
.sally-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #2d3748;
    letter-spacing: -0.5px;
}
.sally-tag {
    font-size: 0.95rem;
    color: #7fbbff;
    font-weight: 600;
    margin-top: 3px;
}
.sally-chatbox {
    background: #fff;
    border-radius: 0 0 32px 32px;
    box-shadow: 0 8px 32px rgba(100, 160, 255, 0.09);
    padding: 0;
    margin-bottom: 24px;
    min-height: 520px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.sally-chat-window {
    flex: 1;
    padding: 24px 32px;
    overflow-y: auto;
    background: #f8fafc;
    min-height: 400px;
    max-height: 470px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.sally-bubble-user {
    background: linear-gradient(135deg, #fff8e1 0%, #ffe6fa 100%);
    color: #744210;
    padding: 15px 20px;
    border-radius: 22px;
    border-bottom-right-radius: 8px;
    margin-left: auto;
    max-width: 75%;
    font-weight: 500;
    line-height: 1.5;
    word-break: break-word;
    box-shadow: 0 2px 8px rgba(255, 193, 7, 0.09);
}
.sally-bubble-bot {
    background: linear-gradient(135deg, #e3f6fd 0%, #d1ecf1 100%);
    color: #0d4f8c;
    padding: 15px 20px;
    border-radius: 22px;
    border-bottom-left-radius: 8px;
    margin-right: auto;
    max-width: 75%;
    font-weight: 500;
    line-height: 1.5;
    word-break: break-word;
    box-shadow: 0 2px 8px rgba(3, 102, 214, 0.09);
}
.sally-input-section {
    background: #f8fafc;
    padding: 20px 32px;
    border-top: 1px solid #eaf4fe;
}
.sally-input-row {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #fff;
    border-radius: 20px;
    padding: 8px 8px 8px 16px;
    box-shadow: 0 2px 8px rgba(100, 160, 255, 0.08);
}
.stTextInput > div > div > input {
    background: transparent !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 12px 0 !important;
    font-size: 1rem !important;
    color: #2d3748 !important;
    font-weight: 500 !important;
    outline: none !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: #a0aec0 !important;
    font-weight: 400 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7fbbff 0%, #ffe0f7 100%) !important;
    color: #2d3748 !important;
    border: none !important;
    border-radius: 50% !important;
    width: 44px !important;
    height: 44px !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(127, 187, 255, 0.13) !important;
    transition: background 0.2s;
    padding: 0 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #b6e0fe 0%, #ffe0f7 100%) !important;
    color: #0d4f8c !important;
}
@media (max-width: 700px) {
    .sally-header, .sally-chatbox, .sally-input-section { padding: 12px !important; }
    .sally-chat-window { padding: 12px !important; }
}
#MainMenu, footer, .stDeployButton, .stDecoration {display: none;}
</style>
""", unsafe_allow_html=True)

# --- LOGGING SYSTEM ---
def log_conversation(user_query, bot_response):
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["timestamp", "user_query", "bot_response", "response_summary"])
    timestamp = datetime.now().isoformat()
    response_summary = bot_response[:SUMMARY_LENGTH] + "..." if len(bot_response) > SUMMARY_LENGTH else bot_response
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([timestamp, user_query, bot_response, response_summary])

# --- ANALYTICS FUNCTION ---
def generate_daily_report():
    try:
        report_date = (datetime.now()).strftime('%Y-%m-%d')
        if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
            return f"📊 **Daily Report ({report_date})**\n\n⚠️ No conversation data available yet. Start chatting to see analytics!"
        df = pd.read_csv(LOG_FILE, quoting=csv.QUOTE_ALL, on_bad_lines='skip')
        for col in ["timestamp", "user_query", "bot_response"]:
            if col not in df.columns:
                return f"📊 **Daily Report ({report_date})**\n\n⚠️ Log file missing required columns."
        if "response_summary" not in df.columns:
            df["response_summary"] = df["bot_response"].apply(
                lambda x: x[:SUMMARY_LENGTH] + "..." if len(str(x)) > SUMMARY_LENGTH else str(x)
            )
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        start_date = pd.to_datetime(report_date)
        end_date = start_date + timedelta(days=1)
        daily_logs = df[(df['timestamp'] >= start_date) & (df['timestamp'] < end_date)]
        if len(daily_logs) == 0:
            return f"📊 **Daily Report ({report_date})**\n\n💬 **Total Conversations:** 0\n\nℹ️ No conversations recorded today. Start chatting!"
        report_content = f"📊 **Daily Conversation Report ({report_date})**\n\n"
        report_content += f"💬 **Total Conversations:** {len(daily_logs)}\n\n"
        question_counts = Counter(daily_logs['user_query'])
        top_questions = question_counts.most_common(10)
        if top_questions:
            report_content += "🔥 **Top Questions:**\n"
            for i, (question, count) in enumerate(top_questions[:5], 1):
                report_content += f"{i}. {question} *(asked {count} times)*\n"
        report_content += "\n💡 **Recent Conversation Samples:**\n"
        sample_size = min(3, len(daily_logs))
        sample_conversations = daily_logs.tail(sample_size)
        for idx, row in sample_conversations.iterrows():
            report_content += f"\n**Q:** {row['user_query']}\n**A:** {row['response_summary']}\n"
            report_content += f"*[Full response: {len(str(row['bot_response']))} characters]*\n"
            report_content += "---\n"
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(report_content)
        return report_content
    except Exception as e:
        return f"❌ **Report generation failed:** {str(e)}"

# --- RESPONSE CLEANING AND GENERATION ---
def clean_response(response):
    import re
    patterns = [
        r'^system\b.*?user\b.*?assistant\n',
        r'^You are a helpful customer support assistant\.?\s*'
    ]
    for pattern in patterns:
        response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
    return response.strip()

@st.cache_resource(show_spinner="Loading Sally...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).eval()
    return model, tokenizer

def generate_response(user_query, model, tokenizer):
    # Customer service script logic for fallback and empathy
    prompt = (
        "<|im_start|>system\n"
        "You are Sally, an AI-powered customer service assistant. "
        "Always greet warmly, clarify user needs, and offer to connect with a human agent if you can't help. "
        "Use short, friendly, and clear sentences. If you don't understand, apologize and ask for clarification. "
        "Tagline: Customer support, anytime. <|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=96,
        temperature=0.7,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_reply = full_response.split("<|im_start|>assistant\n")[-1].strip()
    return clean_response(bot_reply)

# --- MAIN UI ---

# Header Section
st.markdown('''
<div class="sally-header">
    <div class="sally-title-row">
        <div class="sally-avatar">
            <img src="https://cdn-icons-png.flaticon.com/512/3177/3177361.png" alt="Sally">
        </div>
        <div>
            <div class="sally-title">Sally</div>
            <div class="sally-tag">Customer support, anytime.</div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# Main Chat Container
st.markdown('<div class="sally-chatbox">', unsafe_allow_html=True)

# Chat Window
st.markdown('<div class="sally-chat-window">', unsafe_allow_html=True)

# Initialize chat history
if "chat_display" not in st.session_state:
    st.session_state.chat_display = [
        {"role": "assistant", "content": "Hello! 👋 I’m Sally, your customer service assistant. How can I help you today?"}
    ]

# Display chat messages
for msg in st.session_state.chat_display:
    if msg["role"] == "user":
        st.markdown(f'<div class="sally-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sally-bubble-bot">{msg["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input Section
st.markdown('<div class="sally-input-section">', unsafe_allow_html=True)
st.markdown('<div class="sally-input-row">', unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input("", placeholder="Type your question or 'report' for analytics...", key="user_input", label_visibility="collapsed")
    with col2:
        send_clicked = st.form_submit_button("➤")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- CHATBOT LOGIC ---
model, tokenizer = load_model()

if send_clicked and user_input.strip():
    st.session_state.chat_display.append({"role": "user", "content": user_input})
    st.rerun()

if st.session_state.chat_display and st.session_state.chat_display[-1]["role"] == "user":
    last_user_message = st.session_state.chat_display[-1]["content"]
    if last_user_message.strip().lower() == "report":
        with st.spinner("Generating your daily analytics report..."):
            report = generate_daily_report()
        st.session_state.chat_display.append({"role": "assistant", "content": report})
        st.rerun()
    else:
        with st.spinner("Sally is helping you..."):
            bot_reply = generate_response(last_user_message, model, tokenizer)
        st.session_state.chat_display.append({"role": "assistant", "content": bot_reply})
        try:
            log_conversation(last_user_message, bot_reply)
        except Exception as e:
            st.error(f"Logging failed: {e}")
        st.rerun()

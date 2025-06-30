# Sally: AI Customer Service Chatbot 🤖💬

Sally is an AI-powered customer service chatbot built with Streamlit and Hugging Face Transformers. Sally provides instant, friendly, and helpful support to users, answering questions, resolving issues, and generating daily analytics reports for your business.

---

## 🚀 Features

- **Conversational AI:** Uses advanced language models to understand and respond to customer queries in real time.
- **Customer Service Focus:** Empathetic, clear, and concise responses tailored for support scenarios.
- **Conversation Logging:** Every interaction is logged for quality assurance and analytics.
- **Daily Analytics Reports:** Generates a daily summary of conversations, top questions, and sample Q&A for team review.
- **Easy Deployment:** Launchable on Streamlit Community Cloud or your own server.
- **Customizable:** Easily adapt Sally for your brand, FAQ, or support workflow.

---

## 🛠️ Tech Stack

- Python 3.9+
- Streamlit for the interactive web UI
- Hugging Face Transformers for natural language understanding
- PyTorch (for local inference)
- pandas for analytics and reporting
- CSV for lightweight conversation logs

---

## 📦 Project Structure

```
sally-chatbot/
├── deploy.py                # Main Streamlit app
├── requirements.txt         # Python dependencies
├── conversation_logs.csv    # Conversation log (auto-created)
├── daily_report.txt         # Daily analytics report (auto-created)
├── qwen2-faq-finetuned/     # (Optional) Local model files (not on Streamlit Cloud)
└── README.md                # This file
```

---

## 🚦 Deployment Guide

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sally-chatbot.git
cd sally-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Add Your Model

Place your Hugging Face model files in `./qwen2-faq-finetuned/` (must include `config.json`, `pytorch_model.bin` or `model.safetensors`, and tokenizer files).

### 4. Run Locally

```bash
streamlit run deploy.py
```

### 5. Deploy to Streamlit Cloud

- Push your code to GitHub.
- Go to Streamlit Cloud, connect your repo, and deploy.

> **Note:** Large models may not work on Streamlit Cloud due to resource limits. For production, use a hosted model API or run on your own server.

---

## 📊 Analytics & Reporting

- All conversations are logged in `conversation_logs.csv`.
- Type `report` in the chat to generate a daily analytics report (`daily_report.txt`), including:
  - Total conversations
  - Top customer questions
  - Sample Q&A

---

## ⚠️ Limitations

- Model files are not included in the repository due to size constraints.
- Streamlit Cloud does not support large models. For best results, use an API-based model or deploy on your own infrastructure.
- No live agent handoff (but Sally can suggest escalation).

---


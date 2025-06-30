# Sally: AI Customer Service Chatbot 🤖💬

Sally is an AI-powered customer service chatbot built with Streamlit and **Qwen 2 1.5B** from Hugging Face Transformers. Sally provides instant, friendly, and helpful support to users, answering questions, resolving issues, and generating daily analytics reports for your business.

---

## 🚀 Features

- **Conversational AI (Qwen 2 1.5B):** Uses Alibaba’s advanced open-source Qwen 2 1.5B model to understand and respond to customer queries in real time with balanced performance and efficiency.
- **Customer Service Focus:** Empathetic, clear, and concise responses tailored for support scenarios.
- **Conversation Logging:** Every interaction is logged for quality assurance and analytics.
- **Daily Analytics Reports:** Generates a daily summary of conversations, top questions, and sample Q&A for team review.
- **Easy Deployment:** Launchable on Streamlit Community Cloud or your own server.
- **Customizable:** Easily adapt Sally for your brand, FAQ, or support workflow.

---

## 🛠️ Tech Stack

- Python 3.9+
- Streamlit for interactive web UI
- Hugging Face Transformers with **Qwen 2 1.5B** model for natural language understanding
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
├── qwen2-faq-finetuned/     # Local model files for Qwen 2 1.5B
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

### 3. Add Qwen 2 1.5B Model Files

Place your **Qwen 2 1.5B** Hugging Face model files in `./qwen2-faq-finetuned/` (must include `config.json`, `pytorch_model.bin` or `model.safetensors`, and tokenizer files).

### 4. Run Locally

```bash
streamlit run deploy.py
```

### 5. Deploy to Streamlit Cloud

- Push your code to GitHub.
- Go to Streamlit Cloud, connect your repo, and deploy.

> **Note:** Due to resource limits on Streamlit Cloud, large models like Qwen 2 1.5B may not work efficiently. For production, use a hosted inference API or deploy on your own infrastructure with GPU support.

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
- Streamlit Cloud does not support large models well. For optimal performance, use an API-based inference or a server with sufficient GPU resources.
- No live agent handoff (but Sally can suggest escalation).

---

## 📜 License & Credits

- **Model:** [Qwen 2 1.5B](https://huggingface.co/Qwen) by Alibaba (open-source with respective licensing terms).
- **Frameworks:** Streamlit, PyTorch, Hugging Face Transformers.

---


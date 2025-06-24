# Info-Assistant API

![FastAPI](https://img.shields.io/badge/FastAPI-009485?logo=fastapi&logoColor=white&style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white&style=for-the-badge)
![Uvicorn](https://img.shields.io/badge/Uvicorn-0.34.3-0e2236?logo=uvicorn&logoColor=white&style=for-the-badge)
![Vercel](https://img.shields.io/badge/Vercel-Deployed-000?logo=vercel&logoColor=white&style=for-the-badge)
![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge)

---

> **Info-Assistant** is a FastAPI-powered backend that provides internet-augmented answers to user questions, designed for seamless integration with client-side applications.

---

## 🚀 Features
- **Ask Anything:** Query the internet and get AI-generated answers with cited sources.
- **Chat History:** Supports multi-turn conversations for context-aware responses.
- **Source Links:** Returns relevant URLs and sources for transparency.
- **Health Check:** Simple endpoint to verify service and model status.
- **CORS Enabled:** Ready for cross-origin requests from any client.

---

## 🛠️ Tech Stack
- **Framework:** FastAPI
- **Language:** Python 3.12+
- **LLM:** Groq (Llama3-70B)
- **Search:** Tavily
- **Server:** Uvicorn
- **Deployment:** Vercel

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd InfoAssistant
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Copy `.env.example` to `.env` and fill in your API keys:
     - `GROQ_API_KEY`
     - `TAVILY_API_KEY`

4. **Run the server locally:**
   ```bash
   uvicorn main:app --reload
   ```

---

## 🔑 Environment Variables
| Variable         | Description                |
|------------------|---------------------------|
| `GROQ_API_KEY`   | API key for Groq LLM       |
| `TAVILY_API_KEY` | API key for Tavily Search  |

---

## 📚 API Endpoints

### `POST /ask`
- **Description:** Get an AI-generated answer to a question, with internet search context.
- **Request Body:**
  ```json
  {
    "question": "string",
    "chat_history": [
      { "isUser": true, "text": "string" },
      { "isUser": false, "text": "string" }
    ]
  }
  ```
- **Response:**
  ```json
  {
    "answer": "string",
    "sources": ["url1", "url2"],
    "relevant_urls": ["url1", "url2"]
  }
  ```

### `GET /health`
- **Description:** Service health and model status.
- **Response:**
  ```json
  {
    "status": "healthy",
    "model": "llama3-70b-8192",
    "search_available": true,
    "groq_available": true
  }
  ```

### `GET /`
- **Description:** API root info and available endpoints.

---

**Check health:**
```bash
curl http://localhost:8000/health
```

## 💡 Notes
- Make sure to provide valid API keys for Groq and Tavily in your `.env` file.
- The server is CORS-enabled for easy integration with any frontend (e.g., React, Vue, mobile apps).
- For production, deploy on [Vercel](https://vercel.com/) or any cloud provider supporting Python.

---

## 👤 Author
- [Reggie](https://github.com/Kidddvenus)

---

> _Info-Assistant: Your AI-powered internet search assistant for smarter, cited answers._




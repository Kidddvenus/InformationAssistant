from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
import os
from groq import Groq
from tavily import TavilyClient
import logging
from dotenv import load_dotenv
import uvicorn
from urllib.parse import urlparse  # Added this import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize clients
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    logger.info("Clients initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize clients: {str(e)}")
    raise

app = FastAPI(
    title="Internet Information Assistant API",
    description="An AI assistant that searches the internet to answer questions"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", 'X-Requested-With'],
    expose_headers=["Content-Disposition"],
    max_age=600
)

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    relevant_urls: List[str] = []

def get_context(query: str) -> tuple[str, list[str], list[str]]:
    """Search the internet for relevant information"""
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_raw_content=True
        )

        context = "Relevant information:\n\n"
        sources = []
        relevant_urls = []

        for result in response.get('results', []):
            url = result.get('url', '')
            if not url:
                continue

            context += f"Title: {result.get('title', 'N/A')}\n"
            context += f"URL: {url}\n"
            context += f"Content: {result.get('content', 'N/A')}\n\n"
            sources.append(url)

            if (any(keyword in query.lower() for keyword in ['official', 'source', 'reference', 'document']) or
                    result.get('score', 0) > 0.8):
                relevant_urls.append(url)

        if not sources:
            return "No relevant information found.", [], []

        return context, sources, relevant_urls
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return f"Error searching: {str(e)}", [], []

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Endpoint for answering questions with search-enhanced LLM"""
    try:
        logger.info(f"Received question: {request.question}")

        context, sources, relevant_urls = get_context(request.question)

        messages = []
        if request.chat_history:
            for msg in request.chat_history:
                messages.append({
                    "role": "user" if msg.get('isUser', False) else "assistant",
                    "content": msg.get('text', '')
                })

        # Format URLs in context as markdown links
        formatted_context = context
        for url in sources:
            domain = urlparse(url).netloc
            formatted_context = formatted_context.replace(
                f"URL: {url}",
                f"URL: [{domain}]({url})"
            )

        messages.extend([
            {
                "role": "system",
                "content": "You are an interactive social AI assistant that helps answer questions by searching the internet. "
                           "Provide clear, concise and precise answers based on the provided information. "
                           "The conversation flow should be like two people talking normally in a friendly way."
                           "When including URLs, always format them as markdown links like [display text](url).\n\n"
                           f"Context:\n{formatted_context}"
            },
            {
                "role": "user",
                "content": request.question
            }
        ])

        completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=1000,
            top_p=1.0
        )

        answer = completion.choices[0].message.content
        answer = answer.replace('**', '').strip()

        # Format relevant URLs as markdown links
        formatted_urls = []
        for url in relevant_urls[:3]:
            domain = urlparse(url).netloc
            formatted_urls.append(f"[{domain}]({url})")

        if formatted_urls and not any(url in answer for url in relevant_urls):
            answer += "\n\nRelevant links:\n" + "\n".join([f"- {url}" for url in formatted_urls])

        if not answer.endswith(('.', '?', '!')):
            answer += '.'

        if not sources:
            answer = "I couldn't find any relevant information about that."

        logger.info(f"Generated answer: {answer[:200]}...")
        return AnswerResponse(
            answer=answer,
            sources=sources,
            relevant_urls=relevant_urls[:3]
        )

    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    """Service health check"""
    try:
        return {
            "status": "healthy",
            "model": "llama3-70b-8192",
            "search_available": tavily_client is not None,
            "groq_available": groq_client is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/")
async def root():
    return {
        "message": "Internet Information Assistant API is running",
        "endpoints": {
            "/ask": "POST - Submit text questions",
            "/health": "GET - Service health check"
        },
        "models": {
            "llm": "Groq/Llama3-70B",
            "search_engine": "Tavily"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
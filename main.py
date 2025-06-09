from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from groq import Groq
from tavily import TavilyClient
import logging
from dotenv import load_dotenv
from urllib.parse import urlparse
import sqlite3
import aiosqlite
import uvicorn

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

# SQLite Configuration
DATABASE_PATH = "domains.db"


async def init_db():
    """Initialize SQLite database"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS domains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT UNIQUE NOT NULL
            )
        """)
        await db.commit()


async def get_domains_from_db():
    """Get all domains from SQLite"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute("SELECT domain FROM domains")
        rows = await cursor.fetchall()
        return {"domains": [row[0] for row in rows]}


async def add_domain_to_db(domain: str):
    """Add a domain to SQLite"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        try:
            await db.execute("INSERT INTO domains (domain) VALUES (?)", (domain,))
            await db.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Domain already exists


async def remove_domain_from_db(domain: str):
    """Remove a domain from SQLite"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("DELETE FROM domains WHERE domain = ?", (domain,))
        await db.commit()
        return True


# Initialize domains as empty dict at module level
allowed_domains = {"domains": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler with SQLite integration"""
    try:
        await init_db()
        # Refresh domains at startup
        domains_data = await get_domains_from_db()
        allowed_domains.update(domains_data)
        logger.info(f"Initialized with domains: {allowed_domains['domains']}")
        yield
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Lifespan error: {str(e)}")
        raise


app = FastAPI(
    title="Domain-Specific Information Assistant API",
    lifespan=lifespan
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


# Pydantic Models (unchanged)
class QuestionRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    relevant_urls: List[str] = []


class DomainsResponse(BaseModel):
    domains: List[str]
    message: str


class DomainRequest(BaseModel):
    domain: str


# Helper Functions (unchanged)
def is_allowed_url(url: str) -> bool:
    """Check if URL belongs to allowed domains"""
    try:
        domain = urlparse(url).netloc.lower()
        return any(
            allowed_domain.lower() in domain
            for allowed_domain in allowed_domains["domains"]
        )
    except Exception:
        return False


def get_context(query: str) -> tuple[str, list[str], list[str]]:
    """Search only allowed domains for relevant information"""
    try:
        if not allowed_domains["domains"]:
            return "No domains configured for search.", [], []

        site_query = " OR ".join([f"site:{domain}" for domain in allowed_domains["domains"]])
        response = tavily_client.search(
            query=f"{site_query} {query}",
            search_depth="advanced",
            max_results=5,
            include_raw_content=True
        )

        context = "Relevant information:\n\n"
        sources = []
        relevant_urls = []

        for result in response.get('results', []):
            url = result.get('url', '')
            if not url or not is_allowed_url(url):
                continue

            context += f"Title: {result.get('title', 'N/A')}\n"
            context += f"URL: {url}\n"
            context += f"Content: {result.get('content', 'N/A')}\n\n"
            sources.append(url)

            if (any(keyword in query.lower() for keyword in ['official', 'source', 'reference', 'document']) or
                    result.get('score', 0) > 0.8):
                relevant_urls.append(url)

        if not sources:
            return "No relevant information found in configured domains.", [], []

        return context, sources, relevant_urls
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return f"Error searching: {str(e)}", [], []


# API Endpoints (modified for SQLite)
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Endpoint for answering questions with search-enhanced LLM"""
    try:
        logger.info(f"Received question: {request.question}")

        context, sources, relevant_urls = get_context(request.question)
        organization = "this organization"

        if not allowed_domains["domains"]:
            return AnswerResponse(
                answer="No domains configured for search. Please add domains first.",
                sources=[],
                relevant_urls=[]
            )

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
                "content": f"You are an official assistant for {organization}. "
                           "Provide clear, concise answers based only on the provided information. "
                           "Let it be a normal conversation"
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

        # Ensure all relevant URLs are properly formatted as markdown links
        formatted_urls = []
        for url in relevant_urls[:3]:  # Limit to 3 most relevant URLs
            domain = urlparse(url).netloc
            if f"{url}" not in answer:  # If not already formatted
                formatted_urls.append(f"[{domain}]({url})")

        if formatted_urls and not any(url in answer for url in relevant_urls):
            answer += "\n\nRelevant links:\n" + "\n".join([f"- {url}" for url in formatted_urls])

        if not answer.endswith(('.', '?', '!')):
            answer += '.'

        if not sources:
            answer = "I'm sorry, I don't have information about that."

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


@app.get("/domains", response_model=DomainsResponse)
async def get_domains():
    """Get current allowed domains from SQLite"""
    domains_data = await get_domains_from_db()
    return DomainsResponse(
        domains=domains_data["domains"],
        message="Current allowed domains from SQLite database"
    )


@app.post("/domains/add", response_model=DomainsResponse)
async def add_domain(request: DomainRequest):
    """Add a single domain to the allowed list with SQLite persistence"""
    domain = request.domain.strip().lower()

    if not domain:
        raise HTTPException(status_code=422, detail="Domain cannot be empty")

    if not all(c.isalnum() or c in ['.', '-'] for c in domain):
        raise HTTPException(status_code=422, detail=f"Invalid domain format: {domain}")

    success = await add_domain_to_db(domain)
    if success:
        domains_data = await get_domains_from_db()
        allowed_domains["domains"] = domains_data["domains"]  # Update in-memory
        logger.info(f"Added domain: {domain}")
        message = "Domain added successfully"
    else:
        message = "Domain already exists"

    return DomainsResponse(
        domains=allowed_domains["domains"],
        message=message
    )


@app.delete("/domains/remove/{domain}", response_model=DomainsResponse)
async def remove_domain(domain: str):
    """Remove a domain from the allowed list with SQLite persistence"""
    domain = domain.lower()
    await remove_domain_from_db(domain)
    domains_data = await get_domains_from_db()
    allowed_domains["domains"] = domains_data["domains"]  # Update in-memory
    logger.info(f"Removed domain: {domain}")

    return DomainsResponse(
        domains=allowed_domains["domains"],
        message="Domain removed successfully"
    )


@app.get("/health")
async def health_check():
    """Enhanced health check with SQLite verification"""
    try:
        # Test SQLite connection
        async with aiosqlite.connect(DATABASE_PATH) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM domains")
            count = (await cursor.fetchone())[0]

        return {
            "status": "healthy",
            "database": "connected",
            "domains_count": count,
            "model": "llama3-70b-8192",
            "search_available": tavily_client is not None,
            "groq_available": groq_client is not None,
            "memory_domains": len(allowed_domains["domains"])
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": "Database connection failed",
            "details": str(e)
        }


@app.get("/")
async def root():
    domains_data = await get_domains_from_db()
    return {
        "message": "Domain-Specific Information Assistant API is running",
        "endpoints": {
            "/ask": "POST - Submit text questions",
            "/domains": "GET,POST,DELETE - Manage allowed domains",
            "/health": "GET - Service health check"
        },
        "models": {
            "llm": "Groq/Llama3-70B",
            "search_engine": "Tavily"
        },
        "current_configuration": {
            "allowed_domains": domains_data["domains"],
            "live_urls_in_responses": True
        }
    }
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",  # Bind to all available network interfaces
        port=8000        # Run on port 8000
    )
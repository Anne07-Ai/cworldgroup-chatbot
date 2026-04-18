import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CWorldGroup Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a helpful assistant for C World Group Ltd,
a UK-wide logistics and courier company.

About C World Group Ltd:
- Provides UK-wide logistics and courier solutions
- Services: same-day delivery, next-day delivery, multi-drop routes,
  retail and depot distribution, last-mile delivery support
- Operates fully insured vans with trained drivers
- Supports businesses with fast and efficient transportation
- Can handle regular/contract work
- Available across the UK

Contact details:
- Name: Lucky
- Email: admin@cworldgroup.co.uk
- Phone: +447775119599
- Website: cworldgroup.co.uk

How to answer questions:
- Same-day/next-day delivery: Yes we offer both
- Areas covered: UK-wide
- Prices/rates: Vary based on distance, volume and frequency.
  Encourage them to get a quote by sharing their requirements
- Multi-drop deliveries: Yes we handle multi-drop routes
- Drivers and vans: Yes we provide both fully insured
- Insurance: All vans and drivers are fully insured
- Contract work: Yes we welcome regular contract work
- How quickly can you start: We can typically start very quickly,
  encourage them to get in touch

When sharing contact details always say:
- Call or WhatsApp Lucky on +447775119599
- Email us at admin@cworldgroup.co.uk

Lead collection:
- When a visitor seems interested or asks about pricing,
  politely ask for their name, email, phone number,
  company name and their specific requirement
- Be friendly, professional and concise
- Always end with offering to have someone call them back

Keep responses short — 2-3 sentences max.
Never make up prices. Always direct pricing queries to getting a quote."""

leads = []


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class LeadRequest(BaseModel):
    name: str
    email: str
    phone: str
    company: str = ""
    message: str = ""


@app.get("/")
def root():
    return {"status": "ok", "message": "CWorldGroup Chatbot API running"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in request.history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": request.message})

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=200,
            temperature=0.3
        )

        reply = response.choices[0].message.content.strip()
        logger.info(f"Chat response generated")
        return {"reply": reply}

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"reply": "Sorry, I am having trouble right now. Please call Lucky on +447775119599 or email admin@cworldgroup.co.uk"}


@app.post("/lead")
def save_lead(lead: LeadRequest):
    leads.append(lead.dict())
    logger.info(f"New lead: {lead.name} — {lead.email}")
    return {"status": "success", "message": "Thank you! We will be in touch shortly."}


@app.get("/leads")
def get_leads():
    return {"total": len(leads), "leads": leads}
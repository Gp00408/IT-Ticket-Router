"""
Round 5 — take an incoming ticket, classify it with the trained models,
decide which team it should go to, and ask GPT to draft a suggested
first reply for the support agent.

This plays the role of the "generation" step in the original plan
(send question + retrieved context to the GPT API, get an answer back).
Here there's no chunk retrieval — the "context" handed to GPT is the
ticket text plus the category/urgency the classifiers just predicted.
"""

import os

import joblib
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Which team a ticket gets routed to, based on the predicted category.
TEAM_ROUTING = {
    "Access": "Identity & Access Management",
    "Network": "Network Operations",
    "Software": "Application Support",
    "Hardware": "Desktop/Hardware Support",
}

GPT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are an IT helpdesk assistant. Given a support ticket and its "
    "predicted category/urgency, draft a short, professional first reply "
    "to the person who filed it. Acknowledge the issue, set an expectation "
    "based on urgency, and ask for any info needed to resolve it. Keep it "
    "under 100 words. Do not invent specific fix steps you aren't sure of."
)


def load_artifacts():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    category_model = joblib.load(os.path.join(MODELS_DIR, "category_model.joblib"))
    urgency_model = joblib.load(os.path.join(MODELS_DIR, "urgency_model.joblib"))
    return vectorizer, category_model, urgency_model


def classify(ticket_text, vectorizer, category_model, urgency_model):
    X = vectorizer.transform([ticket_text])
    category = category_model.predict(X)[0]
    urgency = urgency_model.predict(X)[0]
    return category, urgency


def generate_suggested_reply(ticket_text, category, urgency) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "[GPT reply skipped: OPENAI_API_KEY is not set. "
            "Add it to your .env file to enable this step.]"
        )

    client = OpenAI(api_key=api_key)
    user_prompt = (
        f"Ticket: {ticket_text}\n"
        f"Predicted category: {category}\n"
        f"Predicted urgency: {urgency}"
    )

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT reply failed: {e}]"


def route_ticket(ticket_text: str) -> dict:
    vectorizer, category_model, urgency_model = load_artifacts()
    category, urgency = classify(ticket_text, vectorizer, category_model, urgency_model)
    team = TEAM_ROUTING.get(category, "General IT Support")
    suggested_reply = generate_suggested_reply(ticket_text, category, urgency)

    return {
        "ticket_text": ticket_text,
        "category": category,
        "urgency": urgency,
        "routed_team": team,
        "suggested_reply": suggested_reply,
    }


def print_result(result: dict) -> None:
    print(f"Ticket:         {result['ticket_text']}")
    print(f"Category:       {result['category']}")
    print(f"Urgency:        {result['urgency']}")
    print(f"Routed to:      {result['routed_team']}")
    print("Suggested reply:")
    print(f"  {result['suggested_reply']}")


def main():
    import sys

    ticket_text = " ".join(sys.argv[1:]).strip()
    if not ticket_text:
        ticket_text = input("Enter ticket text: ").strip()

    result = route_ticket(ticket_text)
    print_result(result)


if __name__ == "__main__":
    main()

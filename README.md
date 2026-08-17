# AI Ticket Classification & Routing System

An AI system that reads an IT support ticket written in plain English and
automatically predicts its **category**, its **urgency**, and which **team**
should handle it — then drafts a suggested first reply. Built as a follow-up
to an earlier SQL/ETL/Power BI project, this one adds an actual trained
machine learning classifier instead of just querying and summarizing data.

![Architecture diagram](docs/architecture.svg)

## How it works

1. **Data prep** (`data_prep.py`) — 360 realistic synthetic tickets, evenly
   split across 4 categories × 3 urgency levels, cleaned and loaded into a
   PostgreSQL database.
2. **Vectorize** (`vectorize.py`) — pulls the tickets back out, splits them
   80/20 into train/test, and converts the ticket text into TF-IDF numeric
   features.
3. **Train category model** (`train_category_model.py`) — Logistic
   Regression, **79.2% accuracy** predicting Network / Software / Hardware /
   Access.
4. **Train urgency model** (`train_urgency_model.py`) — Logistic Regression,
   **81.9% accuracy** predicting Low / Medium / High.
5. **Route a ticket** (`route_ticket.py`) — runs both models on a new
   ticket, looks up which team handles that category, and asks GPT to draft
   a suggested reply.
6. **App** (`app.py`) — a Streamlit page where you type a ticket and see it
   classified and routed live.

Full evaluation details (confusion matrices, per-class scores, known
limitations) are in [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your own PostgreSQL credentials
and (optionally) an `OPENAI_API_KEY` — the suggested-reply feature in Round
5/6 degrades gracefully to a placeholder message if that key isn't set.

## Running it

Run once, in order, to build the dataset and train the models:

```bash
python data_prep.py
python vectorize.py
python train_category_model.py
python train_urgency_model.py
```

Then either try it from the command line:

```bash
python route_ticket.py "My laptop won't turn on and I have a meeting in an hour."
```

or launch the app:

```bash
streamlit run app.py
```

## Tech stack

Python, pandas, scikit-learn (TF-IDF + Logistic Regression), PostgreSQL,
SQLAlchemy, Streamlit, OpenAI API.

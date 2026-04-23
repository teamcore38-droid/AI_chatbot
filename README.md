# Educational Counselling Chatbot

Domain-based AI chatbot for educational counseling. The bot answers questions about courses, fees, duration, entry requirements, intake dates, contact details, and course recommendations.

## Features

- Web landing page and chatbot page
- Natural language chat interface
- Rule-based inference plus trained intent classification
- SQLite knowledge base for courses and contact details
- Logged interactions for learning and evaluation
- Teach-the-bot flow to add new Q&A knowledge
- Optional external LLM fallback for off-topic or unknown questions
- NLP improvements with synonym handling, lemmatization support, and better course extraction

## Project Structure

```
main.py            # Web app entry point
web_app.py         # Flask routes, API endpoints, and page rendering
templates/         # Landing page and chatbot page templates
static/            # CSS and JavaScript assets
inference_engine.py# Intent routing, response generation, learning flow
ml_model.py        # Intent classifier and text normalization
external_llm_client.py # Optional external LLM fallback client
database.py        # SQLite schema and knowledge base helpers
db-setup.py        # Database initialization script
education_counseling.db
requirements.txt
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

For better lemmatization, also download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

Initialize the database:

```bash
python db-setup.py
```

## Run

Start the web app:

```bash
python main.py
```

Then open the landing page in your browser at `http://127.0.0.1:5000`.

## Optional External LLM Fallback

If a question is outside the education domain or the local bot cannot classify it, the app can call an external language model through OpenRouter as a fallback.

Create a file named `.env` in the project root, next to `main.py`, and add:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free
OPENROUTER_HTTP_REFERER=http://localhost
OPENROUTER_TITLE=Education Counselling Chatbot
```

If you want a second model fallback, put multiple models in `OPENROUTER_MODEL` separated by commas. For example:

```env
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free
```

OpenRouter is OpenAI-compatible and can be used with the OpenAI SDK. The optional headers used in this project are `HTTP-Referer=http://localhost` and `X-Title=Education Counselling Chatbot`.

When the external model answers successfully, the chatbot page shows `Answered via external LLM` in the interface.

## NLP Notes

- Text is normalized with synonym replacement and stop-word removal.
- spaCy lemmatization is used when the `en_core_web_sm` model is installed.
- Course names are matched with alias hints such as `AI`, `IT`, `accounting`, and `tourism`.
- The classifier now trains from built-in examples plus [`intent_dataset.json`](./intent_dataset.json), which adds more natural student phrasing.

## Teaching the Bot

Use the **Teach Bot** button in the GUI to save a new question, intent label, and answer. The example is stored in SQLite and the intent model is refreshed.

CLI teaching format:

```text
teach|question|intent|answer
```

Example:

```text
teach|Do you offer scholarships?|scholarship|Yes, merit-based scholarships are available for selected courses.
```

## Packaging for Submission

To create a standalone executable for demo submission, use PyInstaller:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed main.py
```

The executable will be created in the `dist` folder.

## Sample Questions

- What courses do you offer?
- What is the fee for Computer Science?
- What are the entry requirements for Cyber Security?
- Recommend a course for coding
- How can I contact the counselor?

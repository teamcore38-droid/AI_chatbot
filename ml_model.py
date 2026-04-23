import re
import json
from collections import OrderedDict
from pathlib import Path

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy
except ImportError:
    spacy = None


STEMMER = PorterStemmer()
SPACY_NLP = None
if spacy is not None:
    try:
        SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except Exception:
        SPACY_NLP = None

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "show",
    "tell",
    "that",
    "the",
    "their",
    "there",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "you",
    "your",
}

SYNONYM_REPLACEMENTS = OrderedDict(
    [
        ("programmes", "course"),
        ("programs", "course"),
        ("programme", "course"),
        ("program", "course"),
        ("specialisation", "course"),
        ("specialization", "course"),
        ("major", "course"),
        ("field", "course"),
        ("subjects", "course"),
        ("subject", "course"),
        ("modules", "course"),
        ("module", "course"),
        ("study area", "course"),
        ("study areas", "course"),
        ("tuition", "fee"),
        ("tuition fee", "fee"),
        ("tuition fees", "fee"),
        ("costs", "fee"),
        ("cost", "fee"),
        ("price", "fee"),
        ("prices", "fee"),
        ("charges", "fee"),
        ("charge", "fee"),
        ("eligibility", "requirement"),
        ("eligible", "requirement"),
        ("entry", "requirement"),
        ("admission", "requirement"),
        ("admissions", "requirement"),
        ("prerequisite", "requirement"),
        ("prerequisites", "requirement"),
        ("length", "duration"),
        ("time", "duration"),
        ("years", "duration"),
        ("year", "duration"),
        ("recommend", "recommend"),
        ("suggest", "recommend"),
        ("advise", "recommend"),
        ("ai", "artificial intelligence"),
        ("ml", "machine learning"),
    ]
)

INTENT_DATASET_PATH = Path(__file__).with_name("intent_dataset.json")


def _apply_synonyms(text):
    normalized = text.lower()
    for source, target in SYNONYM_REPLACEMENTS.items():
        normalized = re.sub(r"\b{}\b".format(re.escape(source)), target, normalized)
    return normalized


def normalize_text(text):
    if text is None:
        return ""

    normalized = _apply_synonyms(text)

    if SPACY_NLP is not None:
        doc = SPACY_NLP(normalized)
        tokens = []
        for token in doc:
            if token.is_space or token.is_punct or token.like_num:
                continue
            lemma = token.lemma_.strip().lower()
            if not lemma or lemma == "-pron-":
                continue
            if lemma in STOP_WORDS:
                continue
            tokens.append(lemma)
        if tokens:
            return " ".join(tokens)

    tokens = re.findall(r"[a-z0-9']+", normalized)
    filtered = []
    for token in tokens:
        if token in STOP_WORDS:
            continue
        stemmed = STEMMER.stem(token)
        filtered.append(stemmed)
    return " ".join(filtered)


DEFAULT_INTENT_EXAMPLES = OrderedDict(
    [
        (
            "greeting",
            [
                "hello",
                "hi",
                "good morning",
                "good afternoon",
                "hey there",
                "hello counselor",
            ],
        ),
        (
            "goodbye",
            [
                "bye",
                "goodbye",
                "see you later",
                "exit",
                "close the chat",
            ],
        ),
        (
            "thanks",
            [
                "thank you",
                "thanks",
                "much appreciated",
                "thanks a lot",
            ],
        ),
        (
            "help",
            [
                "help me",
                "what can you do",
                "how can you help",
                "help",
                "show me options",
            ],
        ),
        (
            "capabilities",
            [
                "what can this chatbot do",
                "tell me your features",
                "what do you help with",
                "your capabilities",
                "what are you",
            ],
        ),
        (
            "course_list",
            [
                "what courses do you offer",
                "list available courses",
                "show all programs",
                "which courses are available",
                "available courses",
            ],
        ),
        (
            "course_detail",
            [
                "tell me about computer science",
                "give details about cyber security",
                "what is data science about",
                "information about software engineering",
                "course overview",
                "what do we have in cyber security course",
                "what is included in the computer science course",
                "show me cyber security details",
            ],
        ),
        (
            "course_fee",
            [
                "how much is computer science",
                "what is the fee for business management",
                "cost of data science",
                "tuition fee",
                "price of the course",
            ],
        ),
        (
            "most_expensive_course",
            [
                "what is the most expensive course",
                "which course is the most expensive",
                "highest fee course",
                "most costly course",
            ],
        ),
        (
            "cheapest_course",
            [
                "what is the cheapest course",
                "which course is the cheapest",
                "lowest fee course",
                "most cheap course",
            ],
        ),
        (
            "fee_list",
            [
                "give fees of courses",
                "show course fees",
                "list course fees",
                "what are the fees",
                "course fees",
                "fees of courses",
                "tell me about the fees",
                "tell me the fees",
                "show me the fees",
            ],
        ),
        (
            "course_duration",
            [
                "how long is software engineering",
                "what is the duration",
                "how many years is the course",
                "course length",
                "study period",
            ],
        ),
        (
            "duration_list",
            [
                "show course durations",
                "list course durations",
                "what are the durations",
                "duration of courses",
                "tell me about the durations",
                "tell me the durations",
                "show me the durations",
            ],
        ),
        (
            "requirements_list",
            [
                "tell me about the requirements",
                "show me the requirements",
                "what are the requirements",
                "what are the entry requirements",
                "course requirements",
            ],
        ),
        (
            "intake_list",
            [
                "tell me about the intake dates",
                "show me the intake dates",
                "what are the intake dates",
                "what are the intakes",
                "intake dates",
            ],
        ),
        (
            "course_requirements",
            [
                "what are the entry requirements",
                "eligibility for cyber security",
                "what do i need to apply",
                "admission requirements",
                "course requirements",
            ],
        ),
        (
            "course_intake",
            [
                "when is the intake for computer science",
                "when does business management start",
                "next admission date",
                "intake months",
            ],
        ),
        (
            "education_info",
            [
                "what is education",
                "tell me about education",
                "what is educational counseling",
                "what is educational counselling",
            ],
        ),
        (
            "recommendation",
            [
                "recommend a course for me",
                "suggest a course based on coding",
                "what should i study if i like business",
                "which program suits me",
                "best course for me",
                "data",
                "security",
                "business",
                "accounting",
                "design",
                "tourism",
                "coding",
                "ai",
            ],
        ),
        (
            "contact",
            [
                "how can i contact the counselor",
                "what is the phone number",
                "office hours",
                "email address",
                "where is the office",
            ],
        ),
        (
            "fee_list",
            [
                "give fees of courses",
                "show course fees",
                "list course fees",
                "what are the fees",
                "course fees",
                "fees of courses",
            ],
        ),
        (
            "most_expensive_course",
            [
                "what is the most expensive course",
                "which course is the most expensive",
                "highest fee course",
                "most costly course",
            ],
        ),
        (
            "cheapest_course",
            [
                "what is the cheapest course",
                "which course is the cheapest",
                "lowest fee course",
                "most cheap course",
            ],
        ),
        (
            "duration_list",
            [
                "show course durations",
                "list course durations",
                "what are the durations",
                "duration of courses",
            ],
        ),
    ]
)


def load_training_examples(database):
    examples = []

    for intent, phrases in DEFAULT_INTENT_EXAMPLES.items():
        for phrase in phrases:
            examples.append((phrase, intent))

    if database is not None:
        for row in database.get_learned_examples():
            examples.append((row["question"], row["intent"]))

    if INTENT_DATASET_PATH.exists():
        try:
            with INTENT_DATASET_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            for item in payload:
                intent = item.get("intent", "").strip()
                utterances = item.get("utterances", [])
                if not intent or not isinstance(utterances, list):
                    continue
                for utterance in utterances:
                    if isinstance(utterance, str) and utterance.strip():
                        examples.append((utterance, intent))
        except Exception:
            pass

    return examples


class IntentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000)
        self.is_trained = False
        self.training_texts = []
        self.training_labels = []
        self.training_matrix = None
        self.exact_intent_map = {}

    def train(self, examples):
        cleaned_examples = []
        for text, intent in examples:
            normalized = normalize_text(text)
            if normalized:
                cleaned_examples.append((normalized, intent))

        intents = sorted(set(intent for _, intent in cleaned_examples))
        if len(cleaned_examples) < 2 or len(intents) < 2:
            self.is_trained = False
            return False

        texts = [text for text, _ in cleaned_examples]
        labels = [intent for _, intent in cleaned_examples]
        features = self.vectorizer.fit_transform(texts)
        self.classifier.fit(features, labels)
        self.training_texts = texts
        self.training_labels = labels
        self.training_matrix = features
        self.exact_intent_map = {}
        for text, intent in cleaned_examples:
            self.exact_intent_map.setdefault(text, []).append(intent)
        self.is_trained = True
        return True

    def exact_predict(self, text):
        if not self.is_trained or not self.exact_intent_map:
            return None, 0.0

        normalized = normalize_text(text)
        if not normalized:
            return None, 0.0

        matches = self.exact_intent_map.get(normalized)
        if not matches:
            return None, 0.0

        best_intent = max(set(matches), key=matches.count)
        return best_intent, 1.0

    def semantic_predict(self, text, similarity_floor=0.30):
        if not self.is_trained or self.training_matrix is None:
            return None, 0.0

        normalized = normalize_text(text)
        if not normalized:
            return None, 0.0

        query_vector = self.vectorizer.transform([normalized])
        similarities = cosine_similarity(query_vector, self.training_matrix)[0]
        if similarities.size == 0:
            return None, 0.0

        best_index = similarities.argmax()
        best_score = float(similarities[best_index])
        if best_score < similarity_floor:
            return None, best_score

        return self.training_labels[best_index], best_score

    def predict(self, text, confidence_floor=0.45):
        if not self.is_trained:
            return None, 0.0

        exact_intent, exact_score = self.exact_predict(text)
        if exact_intent:
            return exact_intent, exact_score

        normalized = normalize_text(text)
        if not normalized:
            return None, 0.0

        semantic_intent, semantic_score = self.semantic_predict(normalized)
        if semantic_intent:
            return semantic_intent, semantic_score

        features = self.vectorizer.transform([normalized])
        probabilities = self.classifier.predict_proba(features)[0]
        best_index = probabilities.argmax()
        best_intent = self.classifier.classes_[best_index]
        confidence = float(probabilities[best_index])

        if confidence < confidence_floor:
            return None, confidence

        return best_intent, confidence

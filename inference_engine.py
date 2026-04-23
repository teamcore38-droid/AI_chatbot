from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import re

from database import Database
from external_llm_client import ExternalLLMClient
from ml_model import IntentModel, load_training_examples, normalize_text


@dataclass
class ChatResult:
    response: str
    intent: str
    source: str
    confidence: float = 1.0
    teachable: bool = False


class InferenceEngine:
    def __init__(self):
        self.db = Database()
        self.intent_model = IntentModel()
        self.external_llm_client = ExternalLLMClient()
        self.refresh_model()

    def refresh_model(self):
        examples = load_training_examples(self.db)
        self.intent_model.train(examples)

    def teach_example(self, question, intent, answer):
        if not question or not answer:
            return False

        self.db.save_learned_example(question, intent, answer)
        self.refresh_model()
        return True

    @staticmethod
    def _contains_phrase(normalized_text_value, phrase):
        normalized_phrase = normalize_text(phrase)
        return " {} ".format(normalized_phrase) in " {} ".format(normalized_text_value)

    @staticmethod
    def _clean_raw_text(user_input):
        text = (user_input or "").strip().lower()
        return " ".join(re.sub(r"[^a-z0-9']+", " ", text).split())

    @staticmethod
    def _contains_all(normalized_text_value, terms):
        return all(term in normalized_text_value for term in terms)

    @staticmethod
    def _contains_any(normalized_text_value, terms):
        return any(term in normalized_text_value for term in terms)

    @staticmethod
    def _fuzzy_match_token(token, target, threshold=0.75):
        if not token or not target:
            return False
        if token == target:
            return True
        if len(token) <= 3 or len(target) <= 3:
            return token.startswith(target) or target.startswith(token)
        return SequenceMatcher(None, token, target).ratio() >= threshold

    def _contains_fuzzy_term(self, normalized_text_value, term, threshold=0.75):
        normalized_term = normalize_text(term)
        if not normalized_term:
            return False

        haystack_tokens = normalized_text_value.split()
        target_tokens = normalized_term.split()

        if len(target_tokens) == 1:
            target = target_tokens[0]
            return any(self._fuzzy_match_token(token, target, threshold) for token in haystack_tokens)

        if len(target_tokens) == 2:
            first, second = target_tokens
            for index, token in enumerate(haystack_tokens):
                if not self._fuzzy_match_token(token, first, threshold):
                    continue
                for next_token in haystack_tokens[index + 1 :]:
                    if self._fuzzy_match_token(next_token, second, threshold):
                        return True
            return False

        return self._best_phrase_score(normalized_text_value, [normalized_term]) >= threshold

    @staticmethod
    def _best_phrase_score(normalized_text_value, phrases):
        best_score = 0.0
        for phrase in phrases:
            score = SequenceMatcher(None, normalized_text_value, normalize_text(phrase)).ratio()
            if score > best_score:
                best_score = score
        return best_score

    def _mentions_course_context(self, normalized_text_value):
        course_matches = self.db.find_course_mentions(normalized_text_value)
        course_terms = [
            "course",
            "program",
            "programme",
            "degree",
            "subject",
            "study",
            "studies",
        ]
        fuzzy_terms = ["course", "program", "degree", "study", "accounting", "finance", "computer", "security", "tourism"]
        return self._contains_any(normalized_text_value, course_terms) or any(
            self._contains_fuzzy_term(normalized_text_value, term) for term in fuzzy_terms
        ) or bool(course_matches)

    def _looks_like_recommendation_query(self, normalized_text_value):
        interest_markers = [
            "i like",
            "i enjoy",
            "i am interested in",
            "i'm interested in",
            "i want",
            "i prefer",
            "i want to study",
            "i want to work with",
            "i am looking for",
            "suggest",
            "recommend",
            "which course suits",
            "what should i study",
        ]
        topic_hints = [
            "coding",
            "programming",
            "software",
            "app",
            "data",
            "analytics",
            "statistics",
            "security",
            "network",
            "cloud",
            "system",
            "systems",
            "it",
            "ai",
            "artificial intelligence",
            "machine learning",
            "business",
            "marketing",
            "management",
            "finance",
            "accounting",
            "tax",
            "design",
            "creative",
            "media",
            "tourism",
            "hospitality",
            "travel",
        ]
        if not any(self._contains_phrase(normalized_text_value, marker) for marker in interest_markers):
            return False
        return any(self._contains_phrase(normalized_text_value, hint) for hint in topic_hints)

    def _looks_like_course_detail_query(self, normalized_text_value):
        detail_markers = [
            "what do we have in",
            "what do you have in",
            "what do we have",
            "what do you have",
            "what is in",
            "what is there in",
            "what is there about",
            "tell me about",
            "tell me more about",
            "give me details",
            "show me details",
            "course details",
            "details of",
            "details about",
            "more about",
            "info about",
            "information about",
        ]

        if not self._mentions_course_context(normalized_text_value):
            return False

        if any(self._contains_phrase(normalized_text_value, marker) for marker in detail_markers):
            return True

        return any(
            self._contains_fuzzy_term(normalized_text_value, term)
            for term in ["detail", "details", "about", "information", "info", "have", "contains"]
        )

    def _looks_like_price_extreme_query(self, normalized_text_value):
        expensive_markers = [
            "most expensive",
            "highest fee",
            "highest price",
            "most costly",
            "priciest",
        ]
        cheap_markers = [
            "cheapest",
            "lowest fee",
            "lowest price",
            "most cheap",
            "least expensive",
            "affordable",
        ]

        if any(self._contains_phrase(normalized_text_value, marker) for marker in expensive_markers):
            return "most_expensive_course"
        if any(self._contains_phrase(normalized_text_value, marker) for marker in cheap_markers):
            return "cheapest_course"

        if self._contains_any(normalized_text_value, ["expensive", "costly", "priciest", "highest"]):
            return "most_expensive_course"
        if self._contains_any(normalized_text_value, ["cheap", "cheapest", "lowest", "affordable", "least"]):
            return "cheapest_course"

        return None

    @staticmethod
    def _extract_name_from_intro(user_input):
        text = user_input.strip()
        match = re.match(r"^(i am|i'm)\s+(.+)$", text, flags=re.IGNORECASE)
        if not match:
            return None

        candidate = match.group(2).strip()
        if not candidate:
            return None

        if candidate.lower().startswith(("still", "not", "just")):
            return None

        return candidate

    def _detect_rule_intent(self, normalized_text_value, user_input):
        if not normalized_text_value:
            return None

        raw_text = self._clean_raw_text(user_input)

        if raw_text in ["how are you", "how are u", "how do you do"]:
            return "how_are_you"

        if any(
            self._contains_phrase(normalized_text_value, phrase)
            for phrase in ["how are you", "how are u", "how do you do"]
        ):
            return "how_are_you"

        if any(
            self._contains_phrase(normalized_text_value, phrase)
            for phrase in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        ):
            return "greeting"

        if any(
            self._contains_phrase(normalized_text_value, phrase)
            for phrase in ["bye", "goodbye", "see you", "exit", "quit", "close chat"]
        ):
            return "goodbye"

        if any(
            self._contains_phrase(normalized_text_value, phrase)
            for phrase in ["thank you", "thanks", "much appreciated", "thx"]
        ):
            return "thanks"

        if any(
            self._contains_phrase(normalized_text_value, phrase)
            for phrase in ["help", "what can you do", "how can you help", "show me options"]
        ):
            return "help"

        capability_phrases = [
            "what can this chatbot do",
            "your capabilities",
            "what do you help with",
            "what is this chatbot about",
            "what are you about",
            "who are you",
            "what type of educational counseling assistant are you",
        ]
        if any(phrase in raw_text for phrase in capability_phrases):
            return "capabilities"

        if raw_text.startswith(("what ", "who ", "can you ", "could you ")):
            if self._best_phrase_score(
                normalized_text_value,
                [
                    "what is this chatbot about",
                    "what type of educational counseling assistant are you",
                    "what can this chatbot do",
                    "what do you help with",
                    "who are you",
                ],
            ) >= 0.55:
                return "capabilities"

        if raw_text in [
            "what are you about",
            "what is this chatbot about",
            "what can you do",
            "who are you",
            "what do you help with",
        ]:
            return "capabilities"

        if raw_text in [
            "what is education",
            "what is educational counseling",
            "tell me about education",
            "tell me about what is education",
        ]:
            return "education_info"

        if any(
            phrase in raw_text
            for phrase in [
                "fees of courses",
                "course fees",
                "show fees",
                "list fees",
                "what are the fees",
                "give fees of courses",
                "tell me about the fees",
                "tell me about fees",
                "tell me the fees",
                "show me the fees",
                "what are the course fees",
            ]
        ):
            return "fee_list"

        if (
            (self._contains_any(normalized_text_value, ["fee"]) or self._contains_fuzzy_term(normalized_text_value, "fee"))
            and self._contains_any(normalized_text_value, ["about", "show", "list", "all", "course"])
            and not self._mentions_course_context(normalized_text_value)
        ):
            return "fee_list"

        if any(
            phrase in raw_text
            for phrase in [
                "duration of courses",
                "course durations",
                "show durations",
                "list durations",
                "what are the durations",
                "tell me about the durations",
                "tell me about duration",
                "tell me the durations",
                "show me the durations",
                "what are the course durations",
            ]
        ) and not self._mentions_course_context(normalized_text_value):
            return "duration_list"

        if (
            (self._contains_any(normalized_text_value, ["duration"]) or self._contains_fuzzy_term(normalized_text_value, "duration"))
            and self._contains_any(normalized_text_value, ["about", "show", "list", "all", "course"])
        ):
            if not self._mentions_course_context(normalized_text_value):
                return "duration_list"

        if any(
            phrase in raw_text
            for phrase in [
                "tell me about the requirements",
                "tell me about requirements",
                "tell me the requirements",
                "show me the requirements",
                "what are the requirements",
                "what are the entry requirements",
            ]
        ) and not self._mentions_course_context(normalized_text_value):
            return "requirements_list"

        if (
            (self._contains_any(normalized_text_value, ["requirement"]) or self._contains_fuzzy_term(normalized_text_value, "requirement"))
            and self._contains_any(normalized_text_value, ["about", "show", "list", "all", "course", "entry"])
            and not self._mentions_course_context(normalized_text_value)
        ):
            return "requirements_list"

        if any(
            phrase in raw_text
            for phrase in [
                "tell me about the intake",
                "tell me about intake",
                "tell me the intake dates",
                "show me the intake dates",
                "what are the intake dates",
                "what are the intakes",
                "when are the intakes",
            ]
        ) and not self._mentions_course_context(normalized_text_value):
            return "intake_list"

        if (
            (self._contains_any(normalized_text_value, ["intake", "admission"]) or self._contains_fuzzy_term(normalized_text_value, "intake"))
            and self._contains_any(normalized_text_value, ["about", "show", "list", "all", "course", "date", "dates"])
            and not self._mentions_course_context(normalized_text_value)
        ):
            return "intake_list"

        if self._mentions_course_context(normalized_text_value):
            if self._contains_any(normalized_text_value, ["duration"]) or self._contains_fuzzy_term(normalized_text_value, "duration"):
                return "duration"
            if self._contains_any(normalized_text_value, ["fee"]) or self._contains_fuzzy_term(normalized_text_value, "fee"):
                return "fee"
            if self._contains_any(normalized_text_value, ["requirement"]) or self._contains_fuzzy_term(normalized_text_value, "requirement"):
                return "requirements"
            if self._contains_any(normalized_text_value, ["intake", "admission"]) or self._contains_fuzzy_term(normalized_text_value, "intake"):
                return "intake"
            if self._looks_like_course_detail_query(normalized_text_value):
                return "description"

        price_extreme_intent = self._looks_like_price_extreme_query(normalized_text_value)
        if price_extreme_intent:
            return price_extreme_intent

        if any(
            self._contains_phrase(normalized_text_value, phrase)
            for phrase in ["contact", "phone number", "email address", "office hours", "where is the office"]
        ):
            return "contact"

        if self._contains_all(normalized_text_value, ["cours", "avail"]):
            return "course_list"

        if self._contains_all(normalized_text_value, ["cours"]) and self._contains_any(
            normalized_text_value,
            ["offer", "list", "show", "available", "program", "programs"],
        ):
            return "course_list"

        if self._best_phrase_score(
            normalized_text_value,
            [
                "what courses do you offer",
                "what courses are available",
                "which courses are available",
                "list available courses",
                "show all programs",
            ],
        ) >= 0.52:
            return "course_list"

        if any(
            self._contains_phrase(normalized_text_value, phrase)
            for phrase in ["what courses do you offer", "available courses", "list available courses", "show all programs"]
        ):
            return "course_list"

        if any(
            self._contains_phrase(normalized_text_value, phrase)
            for phrase in ["recommend", "suggest", "which program suits me", "best course for me", "what should i study"]
        ):
            return "recommendation"

        if self._looks_like_recommendation_query(normalized_text_value):
            return "recommendation"

        intro_name = self._extract_name_from_intro(user_input)
        if intro_name:
            return "self_intro"

        if any(
            self._contains_phrase(normalized_text_value, phrase)
            for phrase in ["can you understand i speak", "do you understand me", "can you understand me"]
        ):
            return "language_help"

        if self._mentions_course_context(normalized_text_value):
            field_map = self.db.course_field_map()
            for intent, phrases in field_map.items():
                if any(self._contains_phrase(normalized_text_value, phrase) or self._contains_fuzzy_term(normalized_text_value, phrase) for phrase in phrases):
                    return intent

        return None

    def _format_course_card(self, course):
        lines = [
            "{}".format(course["name"]),
            "Description: {}".format(course["description"]),
            "Duration: {}".format(course["duration"]),
            "Fee: {}".format(course["fee"]),
            "Entry requirements: {}".format(course["requirements"]),
            "Intake: {}".format(course["intake"]),
            "Career paths: {}".format(course["career_paths"]),
        ]
        return "\n".join(lines)

    def _list_courses(self):
        courses = self.db.list_courses()
        if not courses:
            return "I could not find any course data in the knowledge base."

        lines = ["Here are the courses I can help with:"]
        for course in courses:
            lines.append("- {} ({})".format(course["name"], course["duration"]))
        lines.append("Ask me about the fee, duration, requirements, intake, or details of any course.")
        return "\n".join(lines)

    def _list_course_fees(self):
        courses = self.db.list_courses()
        if not courses:
            return "I could not find any course fee data in the knowledge base."

        lines = ["Here are the course fees I can help with:"]
        for course in courses:
            lines.append("- {}: {}".format(course["name"], course["fee"]))
        lines.append("If you want, I can also show the duration, requirements, intake, or full details for any course.")
        return "\n".join(lines)

    def _list_course_durations(self):
        courses = self.db.list_courses()
        if not courses:
            return "I could not find any course duration data in the knowledge base."

        lines = ["Here are the course durations I can help with:"]
        for course in courses:
            lines.append("- {}: {}".format(course["name"], course["duration"]))
        lines.append("If you want, I can also show the fee, requirements, intake, or full details for any course.")
        return "\n".join(lines)

    def _list_course_requirements(self):
        courses = self.db.list_courses()
        if not courses:
            return "I could not find any course requirement data in the knowledge base."

        lines = ["Here are the entry requirements I can help with:"]
        for course in courses:
            lines.append("- {}: {}".format(course["name"], course["requirements"]))
        lines.append("If you want, I can also show the fee, duration, intake, or full details for any course.")
        return "\n".join(lines)

    def _list_course_intakes(self):
        courses = self.db.list_courses()
        if not courses:
            return "I could not find any course intake data in the knowledge base."

        lines = ["Here are the intake dates I can help with:"]
        for course in courses:
            lines.append("- {}: {}".format(course["name"], course["intake"]))
        lines.append("If you want, I can also show the fee, duration, requirements, or full details for any course.")
        return "\n".join(lines)

    def _match_course(self, user_input):
        mentions = self.db.find_course_mentions(user_input)
        if mentions:
            matched_courses = []
            for mention in mentions[:2]:
                course = self.db.get_course(mention["name"])
                if course:
                    matched_courses.append(course)
            if matched_courses:
                if len(matched_courses) > 1 and (mentions[0]["score"] - mentions[1]["score"]) < 1:
                    return matched_courses
                return matched_courses[:1]

        matches = self.db.search_courses(user_input)
        if not matches:
            return []
        if len(matches) > 1 and (matches[0]["score"] - matches[1]["score"]) < 0.12:
            return matches[:2]
        return matches[:1]

    def _recommend_courses(self, normalized_text_value):
        course_hint_map = {
            "Computer Science": [
                "coding",
                "programming",
                "software",
                "development",
                "logic",
                "algorithm",
                "computer",
                "app",
                "web",
            ],
            "Software Engineering": [
                "software",
                "engineering",
                "development",
                "testing",
                "architecture",
                "app",
                "web",
                "project",
            ],
            "Data Science": [
                "data",
                "analytics",
                "analysis",
                "statistics",
                "machine learning",
                "ml",
                "visualization",
                "prediction",
            ],
            "Cyber Security": [
                "security",
                "cyber",
                "hacking",
                "network",
                "forensics",
                "privacy",
                "protection",
                "ethical hacking",
            ],
            "Business Management": [
                "business",
                "marketing",
                "management",
                "entrepreneur",
                "leadership",
                "commerce",
                "operations",
            ],
            "Graphic Design": [
                "design",
                "creative",
                "media",
                "branding",
                "visual",
                "ui",
                "ux",
                "art",
            ],
            "Information Technology": [
                "it",
                "information technology",
                "networking",
                "system",
                "systems",
                "database",
                "cloud",
                "support",
                "hardware",
            ],
            "Artificial Intelligence": [
                "ai",
                "artificial intelligence",
                "machine learning",
                "deep learning",
                "automation",
                "neural",
                "intelligent",
                "ml",
                "data",
            ],
            "Accounting and Finance": [
                "accounting",
                "finance",
                "auditing",
                "tax",
                "taxation",
                "bookkeeping",
                "financial",
                "reports",
            ],
            "Tourism and Hospitality Management": [
                "tourism",
                "hospitality",
                "travel",
                "hotel",
                "event",
                "guest",
                "resort",
                "airline",
            ],
        }

        scores = {}
        for course_name, hints in course_hint_map.items():
            course = self.db.get_course(course_name)
            if not course:
                continue

            score = 0
            haystack = normalize_text(
                "{} {} {}".format(course["name"], course["description"], course["keywords"])
            )

            for hint in hints:
                if self._contains_phrase(normalized_text_value, hint):
                    score += 2
            for token in normalized_text_value.split():
                if token in haystack:
                    score += 1

            if score > 0:
                scores[course_name] = score

        if not scores:
            return (
                "Tell me what you are interested in, for example coding, AI, data, security, business, "
                "accounting, design, or tourism, and I will suggest a course."
            )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        chosen = []
        for course_name, _ in ranked[:3]:
            course = self.db.get_course(course_name)
            if course:
                chosen.append(course)

        if not chosen:
            return "I could not match your interests to a course right now."

        lines = ["Based on your interests, I would suggest:"]
        for course in chosen:
            lines.append("- {}: {}".format(course["name"], course["description"]))
        lines.append("If you want, I can compare the fee, duration, or career paths for these options.")
        return "\n".join(lines)

    def _answer_specific_course_question(self, user_input, normalized_text_value, intent):
        matches = self._match_course(user_input)
        if not matches:
            available = ", ".join(self.db.get_course_names())
            return (
                "I could not identify the course you mean. Available courses are: {}. "
                "Please mention one of these by name.".format(available)
            )

        if len(matches) > 1:
            names = ", ".join([course["name"] for course in matches])
            return "I found a few possible matches: {}. Which one did you mean?".format(names)

        course = matches[0]
        if intent == "course_detail":
            return self._format_course_card(course)
        if intent in ["fee", "course_fee"]:
            return "{} fee: {}".format(course["name"], course["fee"])
        if intent in ["duration", "course_duration"]:
            return "{} duration: {}".format(course["name"], course["duration"])
        if intent in ["requirements", "course_requirements"]:
            return "{} entry requirements: {}".format(course["name"], course["requirements"])
        if intent in ["intake", "course_intake"]:
            return "{} intake: {}".format(course["name"], course["intake"])
        if intent == "description":
            return "{} overview: {}".format(course["name"], course["description"])
        if intent == "career_paths":
            return "{} career paths: {}".format(course["name"], course["career_paths"])

        return self._format_course_card(course)

    def _respond_to_intent(self, intent, user_input, normalized_text_value):
        if intent in ["greeting", "goodbye", "thanks", "help", "capabilities"]:
            response = self.db.get_random_faq(intent)
            if response:
                return response

        if intent == "how_are_you":
            return "I am doing well, thanks for asking. How can I help with your course search today?"

        if intent == "self_intro":
            intro_name = self._extract_name_from_intro(user_input) or "there"
            return "Nice to meet you, {}. How can I help you with education counseling today?".format(intro_name)

        if intent == "language_help":
            return "I can understand typed English questions. Please keep your question short and related to courses, fees, or recommendations."

        if intent == "education_info":
            return (
                "Education is the process of learning knowledge, skills, and values. "
                "In this chatbot, I can help you explore study programs and career options."
            )

        if intent == "course_list":
            return self._list_courses()

        if intent == "fee_list":
            return self._list_course_fees()

        if intent == "duration_list":
            return self._list_course_durations()

        if intent == "requirements_list":
            return self._list_course_requirements()

        if intent == "intake_list":
            return self._list_course_intakes()

        if intent in [
            "course_detail",
            "course_fee",
            "course_duration",
            "course_requirements",
            "course_intake",
            "requirements_list",
            "intake_list",
            "fee",
            "duration",
            "requirements",
            "intake",
            "description",
            "career_paths",
        ]:
            return self._answer_specific_course_question(user_input, normalized_text_value, intent)

        if intent == "recommendation":
            return self._recommend_courses(normalized_text_value)

        if intent in ["most_expensive_course", "cheapest_course"]:
            mode = "max" if intent == "most_expensive_course" else "min"
            courses = self.db.get_extreme_fee_course(mode=mode)
            if not courses:
                return "I could not find course fee data in the knowledge base."
            if len(courses) > 1:
                course_names = ", ".join(course["name"] for course in courses)
                amount = courses[0]["fee"]
                if intent == "most_expensive_course":
                    return "The most expensive courses are: {}. They are priced at {}.".format(course_names, amount)
                return "The cheapest courses are: {}. They are priced at {}.".format(course_names, amount)

            course = courses[0]
            if intent == "most_expensive_course":
                return "The most expensive course is {} with a fee of {}.".format(course["name"], course["fee"])
            return "The cheapest course is {} with a fee of {}.".format(course["name"], course["fee"])

        if intent == "contact":
            info = self.db.get_contact_info()
            if not info:
                return "I could not find contact information right now."
            lines = ["Contact details:"]
            for key in ["contact", "email", "location", "hours", "counselor"]:
                if key in info:
                    lines.append(info[key])
            return "\n".join(lines)

        return None

    def get_response(self, user_input):
        raw_text = self._clean_raw_text(user_input)
        if raw_text in ["how are you", "how are u", "how do you do"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = "I am doing well, thanks for asking. How can I help with your course search today?"
            self.db.log_interaction(timestamp, user_input, raw_text, "how_are_you", response, "rule", 1.0)
            return ChatResult(response=response, intent="how_are_you", source="rule", confidence=1.0)

        normalized_input = normalize_text(user_input)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not normalized_input:
            response = "Please type a question about courses, fees, duration, requirements, or recommendations."
            self.db.log_interaction(timestamp, user_input, normalized_input, "empty", response, "system", 1.0)
            return ChatResult(response=response, intent="empty", source="system", confidence=1.0)

        learned_answer = self.db.find_learned_answer(user_input)
        if learned_answer:
            self.db.log_interaction(timestamp, user_input, normalized_input, "learned_qa", learned_answer, "learned", 1.0)
            return ChatResult(response=learned_answer, intent="learned_qa", source="learned", confidence=1.0)

        rule_intent = self._detect_rule_intent(normalized_input, user_input)
        if rule_intent:
            response = self._respond_to_intent(rule_intent, user_input, normalized_input)
            if response:
                source = "rule"
                confidence = 1.0
                self.db.log_interaction(timestamp, user_input, normalized_input, rule_intent, response, source, confidence)
                return ChatResult(response=response, intent=rule_intent, source=source, confidence=confidence)

        model_intent, confidence = self.intent_model.predict(user_input)
        if model_intent:
            response = self._respond_to_intent(model_intent, user_input, normalized_input)
            if response:
                source = "ml"
                self.db.log_interaction(timestamp, user_input, normalized_input, model_intent, response, source, confidence)
                return ChatResult(response=response, intent=model_intent, source=source, confidence=confidence)

        api_intent = self.external_llm_client.classify_intent(
            user_input,
            [
                "greeting",
                "goodbye",
                "thanks",
                "help",
                "capabilities",
                "education_info",
                "course_list",
                "course_detail",
                "course_fee",
                "fee_list",
                "duration_list",
                "requirements_list",
                "intake_list",
                "course_duration",
                "course_requirements",
                "course_intake",
                "most_expensive_course",
                "cheapest_course",
                "recommendation",
                "contact",
                "language_help",
                "unknown",
            ],
        )
        if api_intent and api_intent != "unknown":
            response = self._respond_to_intent(api_intent, user_input, normalized_input)
            if response:
                self.db.log_interaction(timestamp, user_input, normalized_input, api_intent, response, "external_llm_intent", 0.0)
                return ChatResult(
                    response=response,
                    intent=api_intent,
                    source="external_llm_intent",
                    confidence=0.0,
                    teachable=False,
                )

        response = self.db.get_random_faq("fallback")
        if not response:
            response = (
                "I am still learning. Ask me about courses, fees, duration, entry requirements, intake dates, "
                "or type a teaching example if you want to add new knowledge."
            )

        if self.external_llm_client.is_configured():
            llm_reply = self.external_llm_client.get_fallback_response(user_input)
            if llm_reply:
                self.db.log_unknown_query(timestamp, user_input, normalized_input, "external llm fallback used")
                self.db.log_interaction(timestamp, user_input, normalized_input, "unknown", llm_reply, "external_llm", 0.0)
                return ChatResult(
                    response=llm_reply,
                    intent="unknown",
                    source="external_llm",
                    confidence=0.0,
                    teachable=True,
                )

        self.db.log_unknown_query(timestamp, user_input, normalized_input, "fallback response used")
        self.db.log_interaction(timestamp, user_input, normalized_input, "unknown", response, "fallback", 0.0)
        return ChatResult(response=response, intent="unknown", source="fallback", confidence=0.0, teachable=True)

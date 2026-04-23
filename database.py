import random
import sqlite3
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
import re
from nltk.stem import PorterStemmer


DB_PATH = Path(__file__).with_name("education_counseling.db")
STEMMER = PorterStemmer()

DEFAULT_COURSES = [
    {
        "name": "Computer Science",
        "description": "Programming, algorithms, software engineering, and system design.",
        "duration": "3 years",
        "fee": "LKR 850,000 per year",
        "requirements": "G.C.E. A/L with Mathematics preferred",
        "intake": "January and September",
        "career_paths": "Software developer, systems analyst, web engineer",
        "keywords": "computer science computing programming software coding development",
    },
    {
        "name": "Software Engineering",
        "description": "Modern software development, testing, architecture, and teamwork.",
        "duration": "3 years",
        "fee": "LKR 900,000 per year",
        "requirements": "G.C.E. A/L with Mathematics or ICT preferred",
        "intake": "February and August",
        "career_paths": "Software engineer, full stack developer, QA engineer",
        "keywords": "software engineering programming app web development coding",
    },
    {
        "name": "Data Science",
        "description": "Data analysis, machine learning basics, visualization, and statistics.",
        "duration": "3 years",
        "fee": "LKR 920,000 per year",
        "requirements": "G.C.E. A/L with Mathematics or Statistics preferred",
        "intake": "March and October",
        "career_paths": "Data analyst, business analyst, machine learning associate",
        "keywords": "data science analytics machine learning statistics ai",
    },
    {
        "name": "Cyber Security",
        "description": "Network security, digital forensics, ethical hacking, and risk management.",
        "duration": "3 years",
        "fee": "LKR 880,000 per year",
        "requirements": "G.C.E. A/L with Mathematics or ICT preferred",
        "intake": "January and July",
        "career_paths": "Security analyst, SOC analyst, penetration tester",
        "keywords": "cyber security hacking network security forensics",
    },
    {
        "name": "Business Management",
        "description": "Management, marketing, entrepreneurship, and business operations.",
        "duration": "3 years",
        "fee": "LKR 760,000 per year",
        "requirements": "Any approved G.C.E. A/L stream",
        "intake": "February and September",
        "career_paths": "Business analyst, manager, entrepreneur, marketer",
        "keywords": "business management marketing finance entrepreneurship",
    },
    {
        "name": "Graphic Design",
        "description": "Visual communication, branding, digital media, and creative design tools.",
        "duration": "3 years",
        "fee": "LKR 700,000 per year",
        "requirements": "Portfolio preferred, any approved G.C.E. A/L stream",
        "intake": "March and November",
        "career_paths": "Graphic designer, UI artist, content creator",
        "keywords": "graphic design creative media branding ui ux animation",
    },
    {
        "name": "Information Technology",
        "description": "Networks, systems administration, databases, cloud basics, and business IT.",
        "duration": "3 years",
        "fee": "LKR 820,000 per year",
        "requirements": "G.C.E. A/L with Mathematics or ICT preferred",
        "intake": "January and August",
        "career_paths": "IT officer, systems administrator, support engineer",
        "keywords": "information technology it networking databases cloud systems",
    },
    {
        "name": "Artificial Intelligence",
        "description": "Machine learning, intelligent systems, data-driven decision making, and automation.",
        "duration": "3 years",
        "fee": "LKR 950,000 per year",
        "requirements": "G.C.E. A/L with Mathematics, Statistics, or ICT preferred",
        "intake": "March and September",
        "career_paths": "AI engineer, machine learning engineer, data scientist",
        "keywords": "artificial intelligence ai machine learning deep learning automation",
    },
    {
        "name": "Accounting and Finance",
        "description": "Financial reporting, auditing, taxation, and business finance fundamentals.",
        "duration": "3 years",
        "fee": "LKR 780,000 per year",
        "requirements": "Any approved G.C.E. A/L stream",
        "intake": "February and October",
        "career_paths": "Accountant, finance assistant, auditor, tax consultant",
        "keywords": "accounting finance auditing taxation business finance",
    },
    {
        "name": "Tourism and Hospitality Management",
        "description": "Travel operations, hotel management, customer service, and event planning.",
        "duration": "3 years",
        "fee": "LKR 740,000 per year",
        "requirements": "Any approved G.C.E. A/L stream",
        "intake": "January and July",
        "career_paths": "Hotel supervisor, tourism officer, event coordinator",
        "keywords": "tourism hospitality travel hotel event management",
    },
]

DEFAULT_FAQS = [
    ("greeting", "Hello! I am your education counseling assistant. How can I help today?"),
    ("greeting", "Hi there. Ask me about courses, fees, intake dates, requirements, or recommendations."),
    ("greeting", "Welcome back. I can guide you through course options and student support questions."),
    ("goodbye", "Good luck with your studies. Feel free to come back anytime."),
    ("goodbye", "Bye for now. I hope I helped you find the right course."),
    ("goodbye", "Take care. If you need more guidance later, I will be here."),
    ("thanks", "You're welcome."),
    ("thanks", "Happy to help."),
    ("thanks", "No problem at all."),
    ("help", "I can help with course lists, fees, duration, entry requirements, intake dates, contact details, and course recommendations."),
    ("capabilities", "Try questions like: 'What courses do you offer?', 'What is the fee for Computer Science?', or 'Recommend a course for me.'"),
    ("capabilities", "I can answer questions about study programs, student support, course fees, and career options."),
    ("about", "I am an education counseling assistant built to help students explore study options in a focused domain."),
    ("fallback", "I did not fully understand that. Please ask about courses, fees, duration, requirements, intake dates, or recommendations."),
    ("fallback", "I am still learning. You can ask about available courses, course fees, or entry requirements."),
    ("fallback", "Please keep your question related to education counseling, course guidance, or student support."),
]

DEFAULT_CONTACT_INFO = [
    ("contact", "Phone: +94 11 234 5678"),
    ("email", "Email: counseling@edu-assist.lk"),
    ("location", "Location: Student Support Center, Colombo"),
    ("hours", "Office Hours: Monday to Friday, 8:30 AM to 4:30 PM"),
    ("counselor", "Counselor: Ms. Nethmi Perera"),
]

COURSE_ALIAS_HINTS = {
    "Computer Science": ["computer science", "cs", "computing", "coding", "programming"],
    "Software Engineering": ["software engineering", "software", "app development", "web development"],
    "Data Science": ["data science", "data", "analytics", "statistics", "machine learning"],
    "Cyber Security": ["cyber security", "cybersecurity", "security", "network security", "ethical hacking"],
    "Business Management": ["business management", "business", "management", "commerce", "entrepreneurship"],
    "Graphic Design": ["graphic design", "design", "creative design", "branding", "ui", "ux"],
    "Information Technology": ["information technology", "it", "networking", "systems", "cloud", "database"],
    "Artificial Intelligence": ["artificial intelligence", "ai", "machine learning", "deep learning", "automation"],
    "Accounting and Finance": ["accounting and finance", "accounting", "finance", "taxation", "auditing"],
    "Tourism and Hospitality Management": ["tourism and hospitality management", "tourism", "hospitality", "travel", "hotel", "events"],
}


class Database:
    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.initialize()

    @staticmethod
    def normalize_text(text):
        if text is None:
            return ""
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        normalized = []
        for token in tokens:
            if token:
                normalized.append(STEMMER.stem(token))
        return " ".join(normalized)

    @staticmethod
    def _fuzzy_match_token(token, target, threshold=0.75):
        if not token or not target:
            return False
        if token == target:
            return True
        if len(token) <= 3 or len(target) <= 3:
            return token.startswith(target) or target.startswith(token)
        return SequenceMatcher(None, token, target).ratio() >= threshold

    def initialize(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS course_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                duration TEXT NOT NULL,
                fee TEXT NOT NULL,
                requirements TEXT NOT NULL,
                intake TEXT NOT NULL,
                career_paths TEXT NOT NULL,
                keywords TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS faq_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT NOT NULL,
                response TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS office_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                info_key TEXT UNIQUE NOT NULL,
                info_value TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_input TEXT NOT NULL,
                normalized_input TEXT NOT NULL,
                intent TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS learned_qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                normalized_question TEXT NOT NULL,
                intent TEXT NOT NULL,
                answer TEXT NOT NULL,
                approved INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS unknown_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_input TEXT NOT NULL,
                normalized_input TEXT NOT NULL,
                note TEXT NOT NULL
            )
            """
        )
        self.conn.commit()
        self._seed_courses()
        self._seed_faqs()
        self._seed_contact_info()

    def close(self):
        if self.conn:
            self.conn.close()

    def _table_count(self, table_name):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM {}".format(table_name))
        return cursor.fetchone()[0]

    def _seed_courses(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM course_catalog")
        existing_names = {row[0].lower() for row in cursor.fetchall()}
        for course in DEFAULT_COURSES:
            if course["name"].lower() in existing_names:
                continue
            cursor.execute(
                """
                INSERT INTO course_catalog (
                    name, description, duration, fee, requirements, intake, career_paths, keywords
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    course["name"],
                    course["description"],
                    course["duration"],
                    course["fee"],
                    course["requirements"],
                    course["intake"],
                    course["career_paths"],
                    course["keywords"],
                ),
            )
        self.conn.commit()

    def _seed_faqs(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT intent, response FROM faq_entries")
        existing = {(row[0], row[1]) for row in cursor.fetchall()}
        for intent, response in DEFAULT_FAQS:
            if (intent, response) in existing:
                continue
            cursor.execute(
                "INSERT INTO faq_entries (intent, response) VALUES (?, ?)",
                (intent, response),
            )
        self.conn.commit()

    def _seed_contact_info(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT info_key, info_value FROM office_info")
        existing = {(row[0], row[1]) for row in cursor.fetchall()}
        for info_key, info_value in DEFAULT_CONTACT_INFO:
            if (info_key, info_value) in existing:
                continue
            cursor.execute(
                "INSERT INTO office_info (info_key, info_value) VALUES (?, ?)",
                (info_key, info_value),
            )
        self.conn.commit()

    def list_courses(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM course_catalog ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]

    def get_course_names(self):
        return [course["name"] for course in self.list_courses()]

    def course_aliases(self):
        aliases = {}
        for course in self.list_courses():
            hints = COURSE_ALIAS_HINTS.get(course["name"], [])
            keywords = course.get("keywords", "")
            dynamic_keywords = [word.strip() for word in keywords.split() if word.strip()]
            combined_aliases = []
            for alias in hints + [course["name"]] + dynamic_keywords:
                alias_normalized = self.normalize_text(alias)
                if course["name"] != "Artificial Intelligence" and alias_normalized == "ai":
                    continue
                combined_aliases.append(alias)
            aliases[course["name"]] = list(dict.fromkeys(combined_aliases))
        return aliases

    def find_course_mentions(self, text):
        normalized_text = self.normalize_text(text)
        results = []

        for course_name, aliases in self.course_aliases().items():
            matched_aliases = []
            score = 0
            for alias in aliases:
                alias_text = self.normalize_text(alias)
                if not alias_text:
                    continue
                alias_tokens = alias_text.split()
                text_tokens = normalized_text.split()

                exact_match = re.search(r"\b{}\b".format(re.escape(alias_text)), normalized_text)
                fuzzy_match = False

                if not exact_match:
                    if len(alias_tokens) == 1:
                        target = alias_tokens[0]
                        fuzzy_match = any(self._fuzzy_match_token(token, target) for token in text_tokens)
                    else:
                        for alias_token in alias_tokens:
                            if any(self._fuzzy_match_token(token, alias_token) for token in text_tokens):
                                fuzzy_match = True
                                break

                if exact_match or fuzzy_match:
                    matched_aliases.append(alias)
                    if alias_text == self.normalize_text(course_name):
                        score += 5
                    elif " " in alias_text:
                        score += 3
                    elif len(alias_text) <= 3:
                        score += 2
                    else:
                        score += 1

            if score:
                results.append(
                    {
                        "name": course_name,
                        "score": score,
                        "aliases": matched_aliases,
                    }
                )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def search_courses(self, query):
        normalized_query = self.normalize_text(query)
        query_terms = set(normalized_query.split())
        matches = []

        for course in self.list_courses():
            haystack = self.normalize_text(
                " ".join(
                    [
                        course["name"],
                        course["description"],
                        course["keywords"],
                        course["career_paths"],
                    ]
                )
            )
            haystack_terms = set(haystack.split())
            overlap = len(query_terms.intersection(haystack_terms))
            ratio = SequenceMatcher(None, normalized_query, haystack).ratio()
            score = ratio + (overlap * 0.12)
            if score > 0.15:
                course_copy = dict(course)
                course_copy["score"] = score
                matches.append(course_copy)

        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches

    def get_course(self, name):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM course_catalog WHERE LOWER(name) = ?", (name.lower(),))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_random_faq(self, intent):
        cursor = self.conn.cursor()
        cursor.execute("SELECT response FROM faq_entries WHERE intent = ?", (intent,))
        rows = [row[0] for row in cursor.fetchall()]
        if not rows:
            return None
        return random.choice(rows)

    def get_contact_info(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT info_key, info_value FROM office_info ORDER BY info_key")
        return {row[0]: row[1] for row in cursor.fetchall()}

    @staticmethod
    def parse_fee_amount(fee_text):
        if not fee_text:
            return None

        match = re.search(r"(\d[\d,]*)", str(fee_text))
        if not match:
            return None

        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            return None

    def get_extreme_fee_course(self, mode="max"):
        courses = self.list_courses()
        priced_courses = []

        for course in courses:
            amount = self.parse_fee_amount(course.get("fee"))
            if amount is not None:
                priced_courses.append((amount, course))

        if not priced_courses:
            return []

        reverse = mode == "max"
        priced_courses.sort(key=lambda item: item[0], reverse=reverse)
        best_amount = priced_courses[0][0]
        return [course for amount, course in priced_courses if amount == best_amount]

    def log_interaction(self, timestamp, user_input, normalized_input, intent, bot_response, source, confidence):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO interaction_logs (
                timestamp, user_input, normalized_input, intent, bot_response, source, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                user_input,
                normalized_input,
                intent,
                bot_response,
                source,
                confidence,
            ),
        )
        self.conn.commit()

    def log_unknown_query(self, timestamp, user_input, normalized_input, note):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO unknown_queries (timestamp, user_input, normalized_input, note)
            VALUES (?, ?, ?, ?)
            """,
            (timestamp, user_input, normalized_input, note),
        )
        self.conn.commit()

    def save_learned_example(self, question, intent, answer, approved=1):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO learned_qa (
                question, normalized_question, intent, answer, approved, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                question,
                self.normalize_text(question),
                intent,
                answer,
                approved,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        self.conn.commit()

    def get_learned_examples(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT question, intent
            FROM learned_qa
            WHERE approved = 1
            ORDER BY created_at DESC
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def find_learned_answer(self, query, threshold=0.82):
        normalized_query = self.normalize_text(query)
        if not normalized_query:
            return None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT question, normalized_question, answer
            FROM learned_qa
            WHERE approved = 1
            ORDER BY created_at DESC
            """
        )
        best_answer = None
        best_score = 0.0

        for row in cursor.fetchall():
            exact_score = SequenceMatcher(None, normalized_query, row["normalized_question"]).ratio()
            word_overlap = len(set(normalized_query.split()).intersection(set(row["normalized_question"].split())))
            score = exact_score + (word_overlap * 0.08)
            if score > best_score:
                best_score = score
                best_answer = row["answer"]

        if best_score >= threshold:
            return best_answer
        return None

    def course_field_map(self):
        return {
            "fee": ["fee", "fees", "cost", "costs", "tuition", "price", "how much"],
            "duration": ["duration", "how long", "years", "length", "time"],
            "requirements": ["requirement", "requirements", "entry", "eligibility", "qualify"],
            "intake": ["intake", "start date", "admission", "begin", "commence"],
            "description": ["about", "detail", "details", "overview", "info", "information"],
            "career_paths": ["career", "job", "jobs", "after", "opportunities", "work"],
        }

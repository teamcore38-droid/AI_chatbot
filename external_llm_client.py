import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


class ExternalLLMClient:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        raw_model = model or os.environ.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
        self.models = [item.strip() for item in raw_model.split(",") if item.strip()]
        if not self.models:
            self.models = ["google/gemma-4-26b-a4b-it:free"]
        self.http_referer = os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost")
        self.app_title = os.environ.get("OPENROUTER_TITLE", "Education Counselling Chatbot")
        self._client = None

    def is_configured(self):
        return bool(self.api_key)

    def _get_client(self):
        if self._client:
            return self._client

        try:
            from openai import OpenAI
        except ImportError:
            return None

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "HTTP-Referer": self.http_referer,
                "X-Title": self.app_title,
            },
        )
        return self._client

    def get_response(self, user_message, conversation_style=None):
        if not self.is_configured():
            return None

        client = self._get_client()
        if client is None:
            return None

        system_prompt = (
            "You are a helpful assistant. Keep answers short, clear, and relevant. "
            "If possible, relate answers to students or education context."
        )
        if conversation_style:
            system_prompt = "{} {}".format(system_prompt, conversation_style)

        for model_name in self.models:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    stream=False,
                )
            except Exception:
                continue

            content = response.choices[0].message.content
            if content:
                return content.strip()

        return None

    def generate_reply(self, user_input, conversation_style=None):
        return self.get_response(user_input, conversation_style=conversation_style)

    def get_fallback_response(self, user_message, conversation_style=None):
        return self.get_response(user_message, conversation_style=conversation_style)

    def classify_intent(self, user_message, allowed_intents):
        if not self.is_configured():
            return None

        client = self._get_client()
        if client is None:
            return None

        if not allowed_intents:
            return None

        allowed_text = ", ".join(allowed_intents)
        system_prompt = (
            "You are an intent classifier for an educational counseling chatbot. "
            "Return exactly one label from the allowed labels. "
            "If the message is clearly outside the domain, return unknown. "
            "Do not add any extra words."
        )
        user_prompt = (
            "Allowed labels: {}\n"
            "User message: {}\n"
            "Return one label only."
        ).format(allowed_text, user_message)

        for model_name in self.models:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=16,
                    stream=False,
                )
            except Exception:
                continue

            content = response.choices[0].message.content
            if not content:
                continue

            label = content.strip().lower().split()[0].strip(".,:;!?\"'")
            normalized_allowed = {item.lower(): item for item in allowed_intents}
            if label in normalized_allowed:
                return normalized_allowed[label]

        return None

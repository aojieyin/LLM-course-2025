import csv
import re
import requests
from typing import List, Optional, Set
import dspy

# 1. DSPy - Ollama Adapter 


class DSPyOllamaLM(dspy.BaseLM):
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        super().__init__(model=model)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        **kwargs,
    ) -> List[str]:
        # DSPy usually calls LM in chat-style
        if messages is not None:
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in messages
            )

        if prompt is None:
            raise ValueError("DSPyOllamaLM called without prompt/messages")

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return [response.json()["response"]]


# 2. Configure DSPy


lm = DSPyOllamaLM(model="llama3")
dspy.settings.configure(lm=lm)


# 3. Abbreviations rules


KNOWN_ABBREVIATIONS: Set[str] = {
    "JFK", "LAX", "NBA", "NFL", "USA", "EU", "UK",
    "UN", "AI", "ML", "NLP", "BBC", "CNN", "NYC"
}


def breaks_abbreviation(original: str, variant: str) -> bool:
    for abbr in KNOWN_ABBREVIATIONS:
        if abbr in original and abbr not in variant:
            return True
    return False



# 4. DSPy Signature for misspelling generation


class MisspellingSignature(dspy.Signature):
    """
    Generate realistic misspellings of a web search query.
    """
    query: str = dspy.InputField(desc="Original search query")
    n: int = dspy.InputField(desc="Number of misspellings to generate")
    variants: List[str] = dspy.OutputField(
        desc="Misspelled variants of the query"
    )


class MisspellingGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(MisspellingSignature)

    def forward(self, query: str, n: int):
        prompt = (
            "You are generating synthetic misspellings of web search queries "
            "for search engine evaluation.\n\n"
            "Rules:\n"
            "- Keep the query understandable\n"
            "- Do NOT change known abbreviations like JFK, NBA, USA\n"
            "- Produce human-like spelling mistakes\n"
            "- Use a mix of:\n"
            "  * missing letters (omission)\n"
            "  * repeated letters\n"
            "  * transposed letters\n"
            "  * phonetic spellings\n"
            "  * multiple errors combined\n"
            "- Output ONLY the misspelled queries, one per line\n\n"
            f"Query: {query}"
        )

        return self.predict(query=prompt, n=n)


generator = MisspellingGenerator()



# 5. Misspelling generation logic


def generate_misspellings(
    query: str,
    max_variants: int = 5,
    max_retries: int = 3,
) -> List[str]:

    for _ in range(max_retries):
        try:
            result = generator(query=query, n=max_variants)
            raw_variants = result.variants

            cleaned = []
            seen = set()

            for v in raw_variants:
                v = v.strip()
                if not v:
                    continue
                if v == query:
                    continue
                if breaks_abbreviation(query, v):
                    continue
                if v in seen:
                    continue

                seen.add(v)
                cleaned.append(v)

                if len(cleaned) >= max_variants:
                    break

            if cleaned:
                return cleaned

        except Exception as e:
            print(f"[WARN] LLM generation failed: {e}")

    return []



# 6. CSV processing


import csv
from typing import List, Tuple

def process_csv(
    input_path: str,
    output_path: str,
    max_variants: int = 5,
):
    rows: List[Tuple[str, str]] = []

    # Read the two columns in csv（topic, query）
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        first = True
        for row in reader:
            if not row:
                continue

            if len(row) < 2:
                continue

            topic = (row[0] or "").strip()
            query = (row[1] or "").strip()

            if first:
                first = False
                if topic.lower() in {"topic", "category"} and query.lower() in {"query", "queries", "web_search_query"}:
                    continue

            if not query:
                continue

            rows.append((topic, query))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # output three columns：topic + original + misspelled
        writer.writerow(["topic", "original_query", "misspelled_query"])
       

        for topic, query in rows:
            variants = generate_misspellings(query, max_variants)
            for v in variants:
                writer.writerow([topic, query, v])
               



# 7. Entry point


if __name__ == "__main__":
    process_csv(
        input_path="web_search_queries.csv",
        output_path="web_search_queries_with_typos.csv",
        max_variants=5,
    )


import json
import logging
import re
import datetime
from typing import Optional

from groq import Groq
from dotenv import load_dotenv

from ..schemas import StructuredQuery
from ..core.config import settings

load_dotenv()
logger = logging.getLogger(__name__)

_client: Optional[Groq] = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not configured.")
        _client = Groq(api_key=api_key)
        logger.info("Groq client initialized.")
    return _client


_PARSE_SYSTEM = """\
You are a geospatial query parser. Extract structured fields from user questions.
Return ONLY a valid JSON object — no explanation, no markdown, no code fences.

JSON schema:
{
  "metric": one of ["vegetation_change","builtup_change","water_change"],
  "region": string or null,
  "start_date": "YYYY-MM-DD" or null,
  "end_date": "YYYY-MM-DD" or null
}

Rules:
- If no start year mentioned, use 5 years before today.
- If no end date, use today.
- "deforestation" for tree/forest loss queries.
- "drought_index" for drought, dry, water stress queries.
- "land_surface_temperature" for heat, temperature, urban heat island.

Today is {TODAY}.

Examples:
Input: "how much green cover did this area lose since 2020"
Output: {"metric": "vegetation_change", "region": null, "start_date": "2020-01-01", "end_date": "{TODAY}"}

Input: "[Metric: deforestation] how much deforestation has happened in this region over 5 years"
Output: {"metric": "deforestation", "region": null, "start_date": "{FIVE_YEARS_AGO}", "end_date": "{TODAY}"}
"""

_SUMMARY_SYSTEM = """\
You are a geospatial analyst writing concise findings for a dashboard.
Write exactly ONE sentence (max 25 words). State the primary numeric finding.
Round numbers to one decimal place. Be direct and factual.
Return only the sentence, no extra text.
"""


def _extract_json(text: str) -> dict:
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    text = re.sub(r'```\s*$', '', text).strip()
    
    return json.loads(text)


def parse_natural_language_query(text: str) -> StructuredQuery:
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    five_years_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=5*365)).strftime("%Y-%m-%d")
    
    system = _PARSE_SYSTEM.replace("{TODAY}", today).replace("{FIVE_YEARS_AGO}", five_years_ago)

    client = _get_client()
    logger.info(f"LLM: parsing '{text[:80]}'")

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content
        logger.info(f"LLM raw response: {raw}")
        parsed = _extract_json(raw)
        if not parsed.get("start_date"):
            parsed["start_date"] = five_years_ago
        if not parsed.get("end_date"):
            parsed["end_date"] = today

        return StructuredQuery(**parsed)

    except Exception as e:
        logger.error(f"LLM parse error: {e}")
        raise


def generate_summary(query: StructuredQuery, metrics: dict) -> str:
    client = _get_client()
    payload = {
        "metric": query.metric,
        "region": query.region,
        "metrics": metrics,
        "start_date": query.start_date,
        "end_date": query.end_date,
    }
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.1,
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        val = next(iter(metrics.values()), 0)
        return f"Detected {val:.1f} km² of change in {query.region or 'selected area'}."

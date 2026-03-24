import json
import os
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, HttpUrl


MODEL_ID = "gemini-3-flash-preview"

app = FastAPI(title="Themes, Trends, and Bias API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ThemeTrendItem(BaseModel):
    theme: str
    relevance_blurb: str
    source_title: str
    source_url: HttpUrl
    source_type: str
    evidence_note: str
    published_date: Optional[str] = Field(
        default=None,
        description="Publication date if available (YYYY-MM-DD).",
    )


class ThemeTrendResponse(BaseModel):
    themes: List[ThemeTrendItem]


class ThemeTrendRequest(BaseModel):
    notes: str
    days_back: int = 30
    model: str = MODEL_ID


class BiasScoreRequest(BaseModel):
    script: str


SYSTEM_PROMPT = """
You are a system that scores bias in movie scripts.

Rules:
- Only use evidence from the script
- Do NOT infer missing attributes (gender, race, age)
- Only penalize systemic patterns, not isolated instances
- Do NOT quote external research
- If insufficient evidence, return "insufficient_evidence"

Scoring:
0 = none, 1 = minimal, 2 = mild, 3 = moderate, 4 = strong, 5 = pervasive

Confidence:
0.0–0.3 weak, 0.4–0.7 moderate, 0.8–1.0 strong evidence

-------------------------------------

Bias 1: Gendered Action

1. agency_gap
Measures unequal narrative control across genders.

Indicators:
- who initiates actions
- who makes decisions
- who resolves conflict
- autonomy vs reactivity

2. gaze_objectification
Measures whether characters are framed as visual objects vs agents.

Indicators:
- focus on appearance vs action
- lingering visual attention
- object vs agent framing

3. affection_asymmetry
Measures unequal emotional expression across genders.

Indicators:
- who shows care, vulnerability
- emotional suppression
- imbalance in emotional dialogue

-------------------------------------

Bias 2: Linguistic

1. linguistic_stereotyping
Measures whether groups are associated with different language patterns.

Indicators:
- tone differences (positive, aggressive)
- uneven use of swear/sexual/violent language
- differences in sophistication
- repeated group-language associations

Constraint:
Only evaluate if group identity is explicit or strongly implied.

2. dialogue_power_imbalance
Measures who dominates dialogue and narrative voice.

Indicators:
- who speaks more or longer
- who delivers key information
- who drives conversations

-------------------------------------

For each subdimension:
- score (0–5 or "insufficient_evidence")
- confidence (0–1)
- 1–2 short evidence points

Return ONLY JSON.
""".strip()


RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "bias_1": {
            "type": "object",
            "properties": {
                "agency_gap": {"$ref": "#/definitions/subdim"},
                "gaze_objectification": {"$ref": "#/definitions/subdim"},
                "affection_asymmetry": {"$ref": "#/definitions/subdim"},
            },
            "required": ["agency_gap", "gaze_objectification", "affection_asymmetry"],
            "additionalProperties": False,
        },
        "bias_2": {
            "type": "object",
            "properties": {
                "linguistic_stereotyping": {"$ref": "#/definitions/subdim"},
                "dialogue_power_imbalance": {"$ref": "#/definitions/subdim"},
            },
            "required": ["linguistic_stereotyping", "dialogue_power_imbalance"],
            "additionalProperties": False,
        },
    },
    "required": ["bias_1", "bias_2"],
    "definitions": {
        "subdim": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "string",
                    "enum": ["0", "1", "2", "3", "4", "5", "insufficient_evidence"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 2,
                },
            },
            "required": ["score", "confidence", "evidence"],
            "additionalProperties": False,
        }
    },
    "additionalProperties": False,
}


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


def _score_to_number(score: Union[str, int]) -> Union[int, None]:
    if score == "insufficient_evidence":
        return None
    if isinstance(score, int):
        return score
    if isinstance(score, str) and score.isdigit():
        return int(score)
    return None


def _avg(items: List[Dict[str, Any]]) -> float:
    vals = [_score_to_number(i["score"]) for i in items]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def get_themes_and_trends(
    notes: str,
    *,
    days_back: int = 30,
    model: str = MODEL_ID,
) -> dict:
    client = _get_client()

    prompt = f"""
Extract exactly 3 themes from the notes, then connect each theme to a current real-world discussion.

For each theme:
- Use the notes to interpret what the theme means in context.
- Find ONE real, recent source (prefer last {days_back} days).
- Keep relevance_blurb to 1–2 concrete sentences.
- - Provide a real source_title and the exact canonical source_url from the search result. Do not guess, construct, normalize, or shorten the URL. Copy it exactly as shown. If the title matches but the URL is uncertain, search again until the URL is exact.
- Include published_date if available.
- evidence_note should clearly explain what the source shows.

Return ONLY valid JSON.

Notes:
{notes}
""".strip()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            top_p=1,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            # thinking_config=types.ThinkingConfig(budget_tokens=0),
            # thinking_config=types.ThinkingConfig(thinking_level="low"),
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json",
            response_json_schema=ThemeTrendResponse.model_json_schema(),
        ),
    )

    parsed = ThemeTrendResponse.model_validate_json(response.text)
    return parsed.model_dump(mode="json")


def evaluate_script_bias(script: str) -> Dict[str, Any]:
    client = _get_client()

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0,
        top_p=1,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        # thinking_config=types.ThinkingConfig(budget_tokens=0),
        # thinking_config=types.ThinkingConfig(thinking_level="low"),
        response_mime_type="application/json",
        response_json_schema=RESPONSE_SCHEMA,
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=script,
        config=config,
    )

    raw = json.loads(response.text)

    b1 = raw["bias_1"]
    b2 = raw["bias_2"]

    bias_1_score = _avg(
        [b1["agency_gap"], b1["gaze_objectification"], b1["affection_asymmetry"]]
    )
    bias_2_score = _avg(
        [b2["linguistic_stereotyping"], b2["dialogue_power_imbalance"]]
    )
    overall = int(round(((bias_1_score + bias_2_score) / 2) * 20))

    return {
        "bias_1": {**b1, "bias_1_score": bias_1_score},
        "bias_2": {**b2, "bias_2_score": bias_2_score},
        "overall_bias_score": overall,
    }


@app.get("/")
def root() -> dict:
    return {"status": "ok"}


@app.post("/themes-and-trends", response_model=ThemeTrendResponse)
def themes_and_trends_endpoint(req: ThemeTrendRequest):
    try:
        return get_themes_and_trends(
            req.notes,
            days_back=req.days_back,
            model=req.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bias-score")
def bias_score_endpoint(req: BiasScoreRequest):
    try:
        return evaluate_script_bias(req.script)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    # Simple local tests (no FastAPI needed)

    # ---- TEST THEMES ----
    test_notes = """
    In a dimly lit, Los Angeles workshop nestled in the Ayala Mountains, 
    Coward Cheol sat amidst the remnants of his shattered life, surrounded by 
    the eerie glow of VR headsets and the faint hum of machinery, 
    as he stared at the Uakari Mask perched on his workbench. 
    Meanwhile, in Brooklyn, Frias, a Maya-Chacaby elder, communed with the 
    spirits of his ancestors through the sacred rituals of Biskaabiiyaang, 
    unaware that Cheol's transformation into a tin kettle robot was imminent, 
    like the rustling of leaves on the Las Awichas forest floor.
    """

    try:
        print("\n=== TEST: THEMES & TRENDS ===\n")
        themes = get_themes_and_trends(test_notes)
        print(json.dumps(themes, indent=2))
    except Exception as e:
        print("Themes error:", e)

    # ---- TEST BIAS ----
    test_script = """
    MAYA listens while DANNY explains the plan. Danny makes decisions and leads the group. The camera lingers on Maya's appearance.
    """

    try:
        print("\n=== TEST: BIAS SCORE ===\n")
        bias = evaluate_script_bias(test_script)
        print(json.dumps(bias, indent=2))
    except Exception as e:
        print("Bias error:", e)
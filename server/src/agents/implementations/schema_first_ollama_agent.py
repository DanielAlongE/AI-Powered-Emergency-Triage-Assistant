"""
Schema-first structured extraction + deterministic ESI algorithm
ESI v5 rules, with **semantic** (embedding-based) detection for:
  • Decision B high-risk presentations
  • Complaint categorization for resource floors/caps

1) LLM extraction -> JSON (vitals, flags, resources, demographics, etc.).
2) Semantic Concept Classifiers:
   a) High-risk concepts (Decision B) scored via cosine similarity between the
      transcript and canonical descriptions/examples of each risk domain.
   b) Complaint categorization scored the same way to pick the best-fit
      category, which then applies a resource floor/cap.
3) Deterministic ESI logic (A/B/C/D) aligned with ESI v5.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterable

from agents.base import BaseTriageAgent
from models.esi_assessment import ESIAssessment, MedicalConversation
from logger import get_logger
from config import get_settings
from services.ollama_client import get_ollama_gateway
from services.red_flags import RedFlagDetector
from services.ollama_embeddings import get_ollama_embedding_service

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# JSON schema for the LLM extraction
# -----------------------------------------------------------------------------
EXTRACTION_SCHEMA = {
    "name": "esi_structured_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "demographics": {
                "type": "object",
                "properties": {
                    "age_years": {"type": ["number", "null"]},
                    "sex": {"type": ["string", "null"]},
                    "pregnant": {"type": ["boolean", "null"]},
                    "postpartum_weeks": {"type": ["number", "null"]},
                },
            },
            "chief_complaint": {"type": ["string", "null"]},
            "symptoms": {"type": "array", "items": {"type": "string"}},
            "mental_status": {"type": ["string", "null"], "description": "alert | confused | lethargic | unresponsive"},
            "vitals": {
                "type": "object",
                "properties": {
                    "hr": {"type": ["number", "null"]},
                    "rr": {"type": ["number", "null"]},
                    "bp_systolic": {"type": ["number", "null"]},
                    "bp_diastolic": {"type": ["number", "null"]},
                    "temp_f": {"type": ["number", "null"]},
                    "spo2": {"type": ["number", "null"]},
                    "pain_score": {"type": ["number", "null"]},
                },
            },
            # Accept dict OR list of string names
            "high_risk_flags": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "airway_compromise": {"type": ["boolean", "null"]},
                            "respiratory_distress": {"type": ["boolean", "null"]},
                            "shock_or_hypotension": {"type": ["boolean", "null"]},
                            "active_seizure": {"type": ["boolean", "null"]},
                            "post_ictal": {"type": ["boolean", "null"]},
                            "altered_mental_status": {"type": ["boolean", "null"]},
                            "intox_overdose": {"type": ["boolean", "null"]},
                            "severe_bleeding": {"type": ["boolean", "null"]},
                            "penetrating_head_neck_chest_abd": {"type": ["boolean", "null"]},
                            "high_risk_trauma": {"type": ["boolean", "null"]},
                            "stroke_symptoms": {"type": ["boolean", "null"]},
                            "chest_pain_concerning": {"type": ["boolean", "null"]},
                            "shortness_of_breath": {"type": ["boolean", "null"]},
                            "immunocompromised": {"type": ["boolean", "null"]},
                            "pregnancy_complication": {"type": ["boolean", "null"]},
                            "neonate_fever_under_28d": {"type": ["boolean", "null"]},
                        },
                    },
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
            "distress": {
                "type": "object",
                "properties": {
                    "severe_pain": {"type": ["boolean", "null"]},
                    "pain_score": {"type": ["number", "null"]},
                    "severe_psych_distress": {"type": ["boolean", "null"]},
                },
            },
            "anticipated_resources": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "ESI resource types only: labs, ekg, radiograph, ct, mri, ultrasound, angiography, iv_fluids, iv_im_or_neb_meds, specialty_consult, simple_procedure, complex_procedure. Ignore: oral_meds, H&P, splints/slings/crutches, fluorescein/wood lamp, saline lock, point-of-care tests.",
                },
            },
            "notes": {"type": ["string", "null"]},
        },
        "required": ["vitals", "high_risk_flags", "distress", "anticipated_resources"],
        "additionalProperties": True,
    },
}

SYSTEM_EXTRACTION_INSTRUCTIONS = """You are extracting structured triage facts from a nurse–patient transcript.
Respond with ONE JSON object only, matching the schema. Fill what is present; leave unknowns null/empty.

For anticipated_resources, list the MINIMAL initial ED resources likely in the first hour.
Use only ESI resource types (labs, ekg, radiograph, ct, mri, ultrasound, angiography,
iv_fluids, iv_im_or_neb_meds, specialty_consult, simple_procedure, complex_procedure).
Do NOT include non-resources (PO meds, tetanus, H&P, splints/slings/crutches, fluorescein/wood lamp, saline lock).

High-risk flags (set true only if clearly present):
  airway_compromise, respiratory_distress, shock_or_hypotension, active_seizure, post_ictal,
  altered_mental_status, intox_overdose, severe_bleeding, penetrating_head_neck_chest_abd,
  high_risk_trauma, stroke_symptoms, chest_pain_concerning, shortness_of_breath,
  immunocompromised, pregnancy_complication, neonate_fever_under_28d.

Examples of minimal resources:
- 0: Rx refill; dental pain without swelling/fever; simple rash/dermatitis; conjunctivitis;
     otitis externa; corneal abrasion (slit lamp/fluorescein are NOT resources).
- 1: uncomplicated UTI -> labs; simple laceration -> simple_procedure; wrist/ankle injury -> radiograph;
     detox consult without SI/HI -> specialty_consult.
- 2+: first-time seizure (labs + head CT); vomiting/diarrhea with dehydration (labs + IV fluids +/- IV antiemetic);
     significant abdominal pain (labs + US/CT); kidney stone (CT + IV meds/fluids);
     pneumonia (labs + chest radiograph +/- IV meds); cellulitis needing IV abx.
"""

# -----------------------------------------------------------------------------
# Data model for extracted fields
# -----------------------------------------------------------------------------
@dataclass
class Extracted:
    age_years: Optional[float] = None
    sex: Optional[str] = None
    pregnant: Optional[bool] = None
    postpartum_weeks: Optional[float] = None
    chief_complaint: Optional[str] = None
    symptoms: List[str] = field(default_factory=list)
    mental_status: Optional[str] = None
    hr: Optional[float] = None
    rr: Optional[float] = None
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    temp_f: Optional[float] = None
    spo2: Optional[float] = None
    pain_score: Optional[float] = None
    flags: Dict[str, Optional[bool]] = field(default_factory=dict)
    severe_pain: Optional[bool] = None
    severe_psych_distress: Optional[bool] = None
    anticipated_resources: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "Extracted":
        dem = d.get("demographics", {}) or {}
        vit = d.get("vitals", {}) or {}
        raw_flags = d.get("high_risk_flags", {}) or {}
        if isinstance(raw_flags, list):
            flags = {f: True for f in raw_flags if f}
        elif isinstance(raw_flags, dict):
            flags = {k: (bool(v) if v is not None else None) for k, v in raw_flags.items()}
        else:
            flags = {}
        dist = d.get("distress", {}) or {}

        return Extracted(
            age_years=dem.get("age_years"),
            sex=dem.get("sex"),
            pregnant=dem.get("pregnant"),
            postpartum_weeks=dem.get("postpartum_weeks"),
            chief_complaint=d.get("chief_complaint"),
            symptoms=d.get("symptoms") or [],
            mental_status=(d.get("mental_status") or "").lower() or None,
            hr=vit.get("hr"),
            rr=vit.get("rr"),
            bp_systolic=vit.get("bp_systolic"),
            bp_diastolic=vit.get("bp_diastolic"),
            temp_f=vit.get("temp_f"),
            spo2=vit.get("spo2"),
            pain_score=vit.get("pain_score"),
            flags=flags,
            severe_pain=dist.get("severe_pain"),
            severe_psych_distress=dist.get("severe_psych_distress"),
            anticipated_resources=d.get("anticipated_resources") or [],
            notes=d.get("notes"),
        )


# -----------------------------------------------------------------------------
# Resource normalization & counting (ESI counts TYPES, complex_procedure=2)
# -----------------------------------------------------------------------------
RESOURCE_ALIASES: Dict[str, Optional[str]] = {
    "labs": "labs",
    "ekg": "ekg",
    "radiograph": "radiograph",
    "x-ray": "radiograph",
    "xray": "radiograph",
    "ct": "ct",
    "mri": "mri",
    "ultrasound": "ultrasound",
    "angiography": "angiography",
    "iv_fluids": "iv_fluids",
    "iv fluids": "iv_fluids",
    "iv_im_or_neb_meds": "iv_im_or_neb_meds",
    "iv_meds": "iv_im_or_neb_meds",
    "im_meds": "iv_im_or_neb_meds",
    "nebulized_meds": "iv_im_or_neb_meds",
    "specialty_consult": "specialty_consult",
    "consult": "specialty_consult",
    "simple_procedure": "simple_procedure",
    "laceration_repair": "simple_procedure",
    "complex_procedure": "complex_procedure",
    "procedural_sedation": "complex_procedure",
    # NOT resources (ignore)
    "oral_meds": None,
    "tetanus": None,
    "prescription_refill": None,
    "crutches": None,
    "splint": None,
    "sling": None,
    "wound_care": None,
    "history_and_physical": None,
    "point_of_care_testing": None,
    "saline_lock": None,
    "fluorescein_stain": None,
    "wood_lamp": None,
}

def _normalize_resource(name: str) -> Optional[str]:
    key = (name or "").strip().lower().replace(" ", "_")
    return RESOURCE_ALIASES.get(key, key if key in RESOURCE_ALIASES.values() else None)

def _count_resources(resources: Iterable[str]) -> int:
    normalized: List[str] = []
    for r in resources or []:
        nr = _normalize_resource(r)
        if nr:
            normalized.append(nr)
            if nr == "complex_procedure":
                normalized.append(nr)  # counts twice
    unique_non_complex = set([r for r in normalized if r != "complex_procedure"])
    complex_twos = sum(1 for r in normalized if r == "complex_procedure")
    return len(unique_non_complex) + complex_twos


# -----------------------------------------------------------------------------
# Danger-zone vitals by age (ESI v5 Appendix B)
# -----------------------------------------------------------------------------
DANGER_ZONE = [
    (0, 1/12, 190, 60),   # <1 month
    (1/12, 1, 180, 55),   # 1–12 months
    (1, 3, 140, 40),      # 1–3 y
    (3, 5, 120, 35),      # 3–5 y
    (5, 12, 120, 30),     # 5–12 y
    (12, 18, 100, 20),    # 12–18 y
    (18, 150, 100, 20),   # adults
]

def _age_thresholds(age_years: Optional[float]) -> Tuple[float, float]:
    age = float(age_years) if age_years is not None else 30.0
    for a_min, a_max, hr_gt, rr_gt in DANGER_ZONE:
        if a_min <= age < a_max:
            return hr_gt, rr_gt
    return 100.0, 20.0

def _danger_zone_uptriage_needed(ex: Extracted) -> bool:
    hr_thr, rr_thr = _age_thresholds(ex.age_years)
    hr_val = ex.hr if (ex.hr is not None and ex.hr <= 220) else None
    rr_val = ex.rr if (ex.rr is not None and ex.rr <= 80) else None
    spo2_bad = ex.spo2 is not None and ex.spo2 < 92
    hr_bad = hr_val is not None and hr_val > hr_thr
    rr_bad = rr_val is not None and rr_val > rr_thr
    if spo2_bad:
        return True
    if hr_bad and rr_bad:
        return True
    if hr_val is not None and hr_val >= 130:
        return True
    if rr_val is not None and rr_val >= 30:
        return True
    return False


# -----------------------------------------------------------------------------
# Semantic concept classifiers (embedding-based)
# -----------------------------------------------------------------------------
def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    import math
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

@dataclass(frozen=True)
class Concept:
    key: str
    title: str
    description: str
    prototypes: Tuple[str, ...]

# High-risk concepts for Decision B
RISK_CONCEPTS: Tuple[Concept, ...] = (
    Concept(
        key="chest_pain_acs",
        title="ACS-type chest pain",
        description="Acute chest discomfort concerning for acute coronary syndrome.",
        prototypes=(
            "pressure or tightness in the chest possibly radiating to jaw or left arm with diaphoresis or nausea",
            "sudden chest pain at rest, worse with exertion, associated with shortness of breath or sweating",
        ),
    ),
    Concept(
        key="sob_compromise",
        title="Shortness of breath with respiratory compromise",
        description="Shortness of breath or difficulty breathing with increased work of breathing or hypoxia.",
        prototypes=(
            "trouble breathing, speaking in short sentences, low oxygen saturation, tachypnea",
            "respiratory distress with retractions or accessory muscle use, fast breathing",
        ),
    ),
    Concept(
        key="stroke_like",
        title="Stroke-like neurological deficit",
        description="Acute focal neurological deficit.",
        prototypes=(
            "facial droop, slurred speech, arm weakness, aphasia, one-sided weakness or numbness",
        ),
    ),
    Concept(
        key="head_injury_loc",
        title="Head injury with loss of consciousness",
        description="Head trauma with reported loss of consciousness.",
        prototypes=(
            "fall with head strike and brief loss of consciousness",
            "hit head, passed out for a few minutes, now awake but different",
        ),
    ),
    Concept(
        key="psychiatric_danger",
        title="Psychiatric danger to self or others",
        description="Suicidal or homicidal ideation or command hallucinations indicating danger.",
        prototypes=(
            "wants to harm self or others, suicidal thoughts with plan or intent",
            "voices telling the patient to kill self or someone else",
        ),
    ),
    Concept(
        key="dialysis_uremia",
        title="Dialysis/uremia concern",
        description="Missed dialysis or uremic symptoms in a dialysis patient.",
        prototypes=(
            "patient on hemodialysis who missed sessions and feels increasingly unwell or fluid overloaded",
        ),
    ),
    Concept(
        key="peritonsillar_airway",
        title="Peritonsillar/epiglottitis airway threat",
        description="Oropharyngeal infection with airway compromise risk.",
        prototypes=(
            "hot potato muffled voice with drooling and trismus, cannot swallow saliva",
        ),
    ),
    Concept(
        key="cspine_diving_neuro",
        title="C-spine risk with neurologic symptoms after diving",
        description="High-risk mechanism with neck injury and neurologic complaints.",
        prototypes=(
            "diving into shallow water and now has neck pain with tingling or weakness in arms or legs",
        ),
    ),
    Concept(
        key="gi_bleed",
        title="Gastrointestinal bleeding",
        description="Upper or lower GI bleed symptoms.",
        prototypes=(
            "vomiting blood or coffee ground emesis",
            "black tarry stools or bright red blood per rectum",
        ),
    ),
    Concept(
        key="pregnancy_complication",
        title="Pregnancy/postpartum complication",
        description="Pregnancy or postpartum with concerning bleeding, severe pain, or other complication.",
        prototypes=(
            "pregnant with heavy vaginal bleeding or severe abdominal pain",
            "postpartum less than six weeks with fever, heavy bleeding, or severe headache",
        ),
    ),
    Concept(
        key="sepsis_systemic",
        title="Systemic infection / sepsis risk",
        description="Fever with abnormal vitals or high-risk host raising concern for sepsis.",
        prototypes=(
            "fever with fast heart rate or fast breathing or low blood pressure; patient may be elderly or immunocompromised",
        ),
    ),
)

# Complaint categories (for resource floors/caps)
COMPLAINT_CONCEPTS: Tuple[Concept, ...] = (
    # 0-resource caps
    Concept("rash_simple", "Simple rash/dermatitis", "Localized pruritic rash or dermatitis without systemic illness.",
            ("poison ivy or contact dermatitis; itch without fever", "localized urticaria or insect bite")),
    Concept("conjunctivitis", "Conjunctivitis", "Conjunctivitis or pink eye with discharge but no severe pain or vision loss.",
            ("pink eye with discharge; no vision change",)),
    Concept("otitis_externa", "Otitis externa", "Ear pain after swimming with tender ear canal.",
            ("swimmer's ear with canal tenderness; no systemic symptoms",)),
    Concept("dental_simple", "Dental pain", "Toothache without swelling, fever, or airway concerns.",
            ("molar pain, uncomplicated dental caries; no swelling",)),
    Concept("rx_refill", "Prescription refill only", "Medication refill request with no acute complaint.",
            ("needs blood pressure medication refill; feels well",)),
    Concept("corneal_abrasion", "Corneal abrasion (contact lens)", "Eye pain and photophobia after contact lens removal; no vision loss.",
            ("eye scratch from contact lens; photophobia; normal vitals",)),

    # 1-resource
    Concept("simple_laceration", "Simple laceration", "Superficial laceration likely requiring suturing; no sedation.",
            ("small skin cut needing stitches; otherwise well",)),
    Concept("msk_wrist_ankle", "Wrist/ankle injury", "Isolated wrist or ankle injury without deformity.",
            ("slipped and hurt wrist; needs x-ray; vitals normal",)),
    Concept("uti_uncomplicated", "Uncomplicated UTI", "Dysuria/frequency in otherwise well adult, afebrile or low-grade fever.",
            ("burning urination, frequency; not pregnant; stable vitals",)),
    Concept("detox_no_si", "Detox request without SI/HI", "Desires help to stop alcohol/drugs; no suicidal ideation.",
            ("wants detox; denies suicidal thoughts; stable",)),

    # 2+ resource floors (typical)
    Concept("first_time_seizure", "First-time seizure", "New onset seizure now resolved; well appearing.",
            ("first ever seizure at home; now back to baseline",)),
    Concept("seizure_recent", "Recent seizure", "Seizure activity in last hours; post-ictal resolved.",
            ("had a seizure earlier today; now fine",)),
    Concept("headache_migraine", "Headache/migraine", "Headache or migraine without neuro deficits.",
            ("throbbing headache with photophobia; similar to prior",)),
    Concept("gastroenteritis", "Gastroenteritis/dehydration", "Vomiting and/or diarrhea with dehydration risk.",
            ("vomiting all day with little intake; may need IV fluids",)),
    Concept("abdominal_pain", "Abdominal pain needing workup", "Abdominal pain that likely needs labs + imaging.",
            ("child with RLQ pain and fever; needs ultrasound or CT",)),
    Concept("pneumonia", "Pneumonia workup", "Cough/fever with concern for pneumonia.",
            ("fever with productive cough and crackles; needs chest x-ray",)),
    Concept("cellulitis", "Cellulitis needing IV meds", "Skin infection with spreading erythema or systemic signs.",
            ("warm red swollen leg with streaking; may need IV antibiotics",)),
    Concept("kidney_stone", "Kidney stone", "Flank pain radiating to groin; hematuria; needs imaging + IV meds.",
            ("sudden flank pain to groin; nausea; stone suspected",)),
    Concept("dvt_suspect", "Suspected DVT", "Unilateral leg swelling/pain after immobility; needs ultrasound.",
            ("calf pain/swelling after long flight; DVT rule-out",)),
    Concept("peg_tube_issue", "PEG/G-tube issue", "PEG or G-tube dislodgement or malfunction.",
            ("G-tube came out; needs replacement/consult",)),
    Concept("hyperemesis_gravidarum", "Hyperemesis gravidarum", "Pregnancy with persistent vomiting; dehydration risk.",
            ("pregnant, cannot keep fluids down; needs IV fluids",)),
)

# Floors and caps for resource counting
CATEGORY_RESOURCE_POLICY: Dict[str, Tuple[int, Optional[int]]] = {
    # 0-resource caps
    "rash_simple": (0, 0),
    "conjunctivitis": (0, 0),
    "otitis_externa": (0, 0),
    "dental_simple": (0, 0),
    "rx_refill": (0, 0),
    "corneal_abrasion": (0, 0),

    # 1-resource exact
    "simple_laceration": (1, 1),
    "msk_wrist_ankle": (1, 1),
    "uti_uncomplicated": (1, 1),
    "detox_no_si": (1, 1),

    # 2+ resource floors
    "first_time_seizure": (2, None),
    "seizure_recent": (2, None),
    "headache_migraine": (0, None),  # may be 0–2+; cap handled below
    "gastroenteritis": (2, None),
    "abdominal_pain": (2, None),
    "pneumonia": (2, None),
    "cellulitis": (2, None),
    "kidney_stone": (2, None),
    "dvt_suspect": (2, None),
    "peg_tube_issue": (2, None),
    "hyperemesis_gravidarum": (2, None),

    "other": (0, None),
}

class _SemanticMatcher:
    """Embedding-based concept similarity using local Ollama embeddings."""
    def __init__(self) -> None:
        self._svc = get_ollama_embedding_service()
        # Precompute and cache concept embeddings
        self._risk_docs = [(c.key, self._combine(c)) for c in RISK_CONCEPTS]
        self._risk_embs = [self._svc.embeddings.embed_query(doc) for _, doc in self._risk_docs]

        self._complaint_docs = [(c.key, self._combine(c)) for c in COMPLAINT_CONCEPTS]
        self._complaint_embs = [self._svc.embeddings.embed_query(doc) for _, doc in self._complaint_docs]

    @staticmethod
    def _combine(c: Concept) -> str:
        return f"{c.title}. {c.description}. Examples: " + " | ".join(c.prototypes)

    def score_risks(self, text: str) -> Dict[str, float]:
        q = text or ""
        q_emb = self._svc.embeddings.embed_query(q)
        scores = {}
        for (key, _doc), doc_emb in zip(self._risk_docs, self._risk_embs):
            scores[key] = _cosine(q_emb, doc_emb)
        return scores

    def best_complaint(self, text: str) -> Tuple[str, float]:
        q = text or ""
        q_emb = self._svc.embeddings.embed_query(q)
        best_key, best_score = "other", 0.0
        for (key, _doc), doc_emb in zip(self._complaint_docs, self._complaint_embs):
            score = _cosine(q_emb, doc_emb)
            if score > best_score:
                best_key, best_score = key, score
        return best_key, best_score


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------
class SchemaFirstOllamaAgent(BaseTriageAgent):
    """
    Triage agent that uses gpt-oss-20b for structured extraction and
    an embedding-based semantic layer for robust high-risk detection
    and complaint categorization. Final ESI is computed deterministically.
    """

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config or {})
        settings = get_settings()
        self._gateway = get_ollama_gateway(
            host=(self.config.get("ollama_host") if self.config else None),
            inference_mode=(self.config.get("inference_mode") if self.config else None),
        )
        self._red_flags = RedFlagDetector(settings.red_flag_lexicon_path)
        self.model_override = self.config.get("model_override", self.config.get("model", "gpt-oss-20b"))
        self.temperature = float(self.config.get("temperature", 0.05))  # low for extraction stability
        self.max_questions = int(self.config.get("max_questions", 3))

        # Semantic matcher (embeddings)
        try:
            self._matcher = _SemanticMatcher()
            logger.info("semantic_matcher_initialized", embedding_model=self._matcher._svc.model)
        except Exception as e:
            logger.warning("semantic_matcher_init_failed", error=str(e))
            self._matcher = None  # Fallback: will rely solely on LLM flags/resources

        logger.info("schema_first_agent_v4_initialized", model=self.model_override, temperature=self.temperature)

    # Public API
    def triage(self, conversation: MedicalConversation) -> ESIAssessment:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._run(conversation))
            finally:
                loop.close()
        except Exception as e:
            logger.error("schema_first_triage_error", error=str(e))
            return ESIAssessment(
                esi_level=3,
                confidence=0.1,
                rationale=f"Safe fallback due to error in SchemaFirstOllamaAgent v4: {e}",
                agent_name=self.name,
                follow_up_questions=[],
            )

    # Internal async
    async def _run(self, conversation: MedicalConversation) -> ESIAssessment:
        text = conversation.get_full_text()

        # Red-flag follow-ups
        matches = self._red_flags.scan(text or "")
        followups = self._red_flags.suggested_questions(matches)[: self.max_questions]

        # LLM extraction
        prompt = SYSTEM_EXTRACTION_INSTRUCTIONS + "\n\n--- TRANSCRIPT START ---\n" + (text or "") + "\n--- TRANSCRIPT END ---\n"
        llm = await self._gateway.stream_completion(
            prompt=prompt,
            json_schema=EXTRACTION_SCHEMA,
            temperature=self.temperature,
            model_override=self.model_override,
        )

        extracted: Optional[Extracted] = None
        if isinstance(llm, dict) and llm:
            try:
                extracted = Extracted.from_json(llm)
            except Exception as e:
                logger.warning("extraction_parse_failed", error=str(e), preview=json.dumps(llm)[:200])

        if not extracted:
            # Minimal fallback: vitals are essential for danger-zone; keep it simple.
            extracted = Extracted()

        level, rationale, used_resources, path = self._decide_esi(extracted, text)

        conf = 0.92 if level in (1, 2) else 0.80 if used_resources >= 2 else 0.70 if used_resources == 1 else 0.66

        return ESIAssessment(
            esi_level=level,
            confidence=conf,
            rationale=f"{rationale} | Decision path: {path}. Resources counted: {used_resources}.",
            follow_up_questions=list(dict.fromkeys(followups)),
            agent_name=self.name,
        )

    # ---------------- Deterministic ESI logic (A/B/C/D) ----------------
    def _decide_esi(self, ex: Extracted, raw_text: str) -> Tuple[int, str, int, str]:
        # A: lifesaving?
        a = self._is_level1(ex)
        if a:
            return 1, a, 0, "A→Level 1"

        # B: high risk? (LLM flags OR semantic concepts + vitals context)
        b = self._is_high_risk_B(ex, raw_text)
        if b:
            return 2, b, 0, "B→Level 2"

        # C: resource counting with complaint calibration
        category, score = self._categorize_complaint(raw_text, ex)
        resources = set(_normalize_resource(r) for r in ex.anticipated_resources if _normalize_resource(r))

        # If extraction missed resources, add minimal heuristics based on category
        if not resources:
            if category == "msk_wrist_ankle":
                resources.add("radiograph")
            elif category == "simple_laceration":
                resources.add("simple_procedure")
            elif category == "uti_uncomplicated":
                resources.add("labs")
            elif category in ("gastroenteritis", "hyperemesis_gravidarum"):
                resources.update({"labs", "iv_fluids", "iv_im_or_neb_meds"})
            elif category in ("abdominal_pain", "dvt_suspect"):
                resources.update({"labs", "ultrasound"})
            elif category == "kidney_stone":
                resources.update({"ct", "iv_im_or_neb_meds"})
            elif category == "pneumonia":
                resources.update({"labs", "radiograph"})
            elif category == "cellulitis":
                resources.update({"labs", "iv_im_or_neb_meds"})
            elif category in ("first_time_seizure", "seizure_recent"):
                resources.update({"labs", "ct"})

        raw_count = _count_resources(resources)
        floor, cap = CATEGORY_RESOURCE_POLICY.get(category, (0, None))

        # Headache special case: cap benign at <=1 if afebrile and no neuro flag
        if category == "headache_migraine":
            if not ex.flags.get("stroke_symptoms") and not (ex.temp_f and ex.temp_f >= 100.4):
                cap = 1 if (cap is None or cap > 1) else cap

        resource_count = max(raw_count, floor)
        if cap is not None:
            resource_count = min(resource_count, cap)

        # D: vital sign up-triage
        if resource_count >= 2:
            if _danger_zone_uptriage_needed(ex):
                return 2, "Danger-zone vital signs for age after predicting ≥2 resources", resource_count, "C→D→Level 2"
            return 3, "Two or more resources anticipated", resource_count, "C→Level 3"
        elif resource_count == 1:
            return 4, "One resource anticipated", resource_count, "C→Level 4"
        else:
            return 5, "No resources anticipated", resource_count, "C→Level 5"

    # ---------------- Decision helpers ----------------
    def _is_level1(self, ex: Extracted) -> Optional[str]:
        # Unresponsive or active seizure now
        if (ex.mental_status and ex.mental_status == "unresponsive") or ex.flags.get("active_seizure"):
            return "Immediate lifesaving intervention: unresponsive or active seizure"

        # Airway/breathing compromise with severe hypoxia or very abnormal RR + AMS
        if ex.flags.get("airway_compromise") or ex.flags.get("respiratory_distress"):
            if ex.spo2 is not None and ex.spo2 < 90:
                return "Severe respiratory failure (SpO2 < 90%)"
            if (ex.rr is not None and (ex.rr <= 10 or ex.rr >= 40)) and (ex.mental_status in {"lethargic", "confused"}):
                return "Severe respiratory compromise with altered mentation"

        # Explicit shock/hypotension
        if ex.flags.get("shock_or_hypotension"):
            return "Signs of shock with hypotension"
        if ex.bp_systolic is not None and ex.bp_systolic < 90:
            return "Profound hypotension requiring immediate support"

        # Penetrating trauma to critical regions
        if ex.flags.get("penetrating_head_neck_chest_abd"):
            return "Penetrating trauma to critical region—immediate interventions likely"

        # Overdose with respiratory depression
        if ex.flags.get("intox_overdose"):
            if (ex.rr is not None and ex.rr <= 10) or (ex.spo2 is not None and ex.spo2 < 90):
                return "Toxic ingestion with respiratory depression (needs airway)"
        return None

    def _is_high_risk_B(self, ex: Extracted, raw_text: str) -> Optional[str]:
        # Direct flags from extraction
        if ex.flags.get("altered_mental_status") or (ex.mental_status in {"confused", "lethargic"}):
            return "Altered mental status—high risk"
        if ex.flags.get("neonate_fever_under_28d"):
            return "Fever in infant <28 days—high risk"
        if (ex.age_years is not None and ex.age_years < (60/365) and ex.temp_f and ex.temp_f >= 100.4):
            return "Fever in infant <60 days—high risk"
        if ex.flags.get("high_risk_trauma"):
            return "High-risk trauma mechanism—high risk"
        if (ex.pregnant or (ex.postpartum_weeks is not None and ex.postpartum_weeks <= 6)) and ex.flags.get("pregnancy_complication"):
            return "Pregnancy/postpartum complication—high risk"

        # Semantic risk scoring if matcher available
        if not self._matcher:
            return None

        scores = self._matcher.score_risks(raw_text or "")
        strong = 0.72
        moderate = 0.62

        # SOB with compromise
        sob_score = scores.get("sob_compromise", 0.0)
        if sob_score >= moderate and (ex.flags.get("respiratory_distress") or (ex.rr and ex.rr >= 30) or (ex.spo2 is not None and ex.spo2 <= 92)):
            return "Shortness of breath with respiratory compromise—high risk"

        # Chest pain suggestive of ACS
        if scores.get("chest_pain_acs", 0.0) >= strong:
            return "Concerning chest pain (ACS risk)—high risk"

        # Stroke
        if scores.get("stroke_like", 0.0) >= strong:
            return "Stroke-like neurological deficit—time-sensitive"

        # Head injury with LOC
        if scores.get("head_injury_loc", 0.0) >= strong:
            return "Head injury with LOC—high risk"

        # Psychiatric danger
        if scores.get("psychiatric_danger", 0.0) >= strong or (ex.severe_psych_distress and scores.get("psychiatric_danger", 0.0) >= moderate):
            return "Psychiatric emergency with danger to self/others—high risk"

        # Dialysis/uremia
        if scores.get("dialysis_uremia", 0.0) >= strong:
            return "Dialysis-related risk—high risk"

        # Peritonsillar/airway
        if scores.get("peritonsillar_airway", 0.0) >= strong:
            return "Potential airway threat (peritonsillar/epiglottitis)—high risk"

        # C-spine after diving with neuro symptoms
        if scores.get("cspine_diving_neuro", 0.0) >= strong:
            return "Possible C-spine injury with neuro symptoms—high risk"

        # GI bleed with concerning features
        gi = scores.get("gi_bleed", 0.0)
        elderly = ex.age_years is not None and ex.age_years >= 65
        hypotn = ex.bp_systolic is not None and ex.bp_systolic < 100
        tachy = ex.hr is not None and ex.hr > 100
        if gi >= moderate and (elderly or hypotn or tachy):
            return "Gastrointestinal bleeding with concerning features—high risk"

        # Sepsis/systemic infection
        sepsis = scores.get("sepsis_systemic", 0.0)
        fever = ex.temp_f is not None and ex.temp_f >= 100.4
        tachyp = ex.rr is not None and ex.rr > 22
        if sepsis >= moderate and (fever and (tachy or tachyp or hypotn)):
            return "Systemic infection risk (fever + abnormal vitals)—high risk"

        return None

    def _categorize_complaint(self, raw_text: str, ex: Extracted) -> Tuple[str, float]:
        if not self._matcher:
            return "other", 0.0
        key, score = self._matcher.best_complaint(" ".join([ex.chief_complaint or "", " ".join(ex.symptoms or []), ex.notes or "", raw_text]))
        # low-confidence guard
        if score < 0.55:
            return "other", score
        return key, score


# Optional smoke test
if __name__ == "__main__":
    conv = MedicalConversation(turns=[])
    agent = SchemaFirstOllamaAgent("Schema-First (Ollama) v4")
    print(agent.triage(conv))

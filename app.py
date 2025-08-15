# app.py
import os
import json
import re
from typing import Optional, List, Dict
from collections import Counter
import difflib
import numpy as np
import streamlit as st
from pathlib import Path

# Internal modules
from logic.rag_lite import ensure_built, search_docx, search_guides
from logic.prompt_templates import build_messages_two_source
from helper_functions.llm import get_completion_by_messages, get_embedding
from utility import check_password

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Guide Copilot", layout="wide")

# Canonical phase order (for inference & context)
PHASE_ORDER: List[str] = [
    "Registration",
    "Prioritisation",
    "Research/Ideation",
    "Estimation",
    "Alignment Checkpoint",
    "Approval of Requirements",
    "Development",   # canonical (no "Phase")
    "Go-live",
]

# Phase keyword hints for heuristic inference
PHASE_KEYWORDS: Dict[str, List[str]] = {
    "Registration": ["registration", "register", "intake", "submission"],
    "Prioritisation": ["prioritisation", "prioritization", "prioritis", "prioritiz", "ranking", "triage", "backlog order", "roadmap"],
    "Research/Ideation": ["discovery", "hypothesis", "user research", "problem", "ideation", "prototype", "spike"],
    "Estimation": ["estimate", "estimation", "sizing", "t-shirt", "story points", "effort"],
    "Alignment Checkpoint": ["alignment", "checkpoint", "steerco", "steering", "governance review", "architecture review", "design review"],
    "Approval of Requirements": ["requirements", "requirement", "brd", "sign-off", "sign off", "approval", "approve"],
    "Development": ["develop", "development", "build", "implement", "code", "unit test", "integration test", "qa", "uat", "regression", "bugfix", "deploy", "ci/cd"],
    "Go-live": ["go-live", "go live", "launch", "release", "rollout", "production deploy", "ship"],
}

# Roles (optional input)
ROLES = [
    "Group Product Owner",
    "Product Owner",
    "Product Manager",
    "Tech Squad Lead",
    "Product Designer",
    "Developer",
]

# -----------------------------------------------------------------------------
# Simple domain relevance gate (keeps the assistant on-purpose)
# -----------------------------------------------------------------------------
DOMAIN_VEC = None
DOMAIN_THRESHOLD = 0.23  # adjust 0.20–0.30 as needed


def _load_domain_vector() -> Optional[np.ndarray]:
    """Load the centroid embedding of the DOCX chunks (built by ensure_built())."""
    global DOMAIN_VEC
    if DOMAIN_VEC is not None:
        return DOMAIN_VEC

    idx_path = Path("index/docx_index.json")
    if not idx_path.exists():
        return None

    try:
        data = json.loads(idx_path.read_text(encoding="utf-8"))
        vecs = np.array(data.get("vecs") or [], dtype="float32")
        if vecs.size == 0:
            return None
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        vecs = vecs / norms
        centroid = vecs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        DOMAIN_VEC = centroid.astype("float32")
        return DOMAIN_VEC
    except Exception:
        return None


def relevance_score(text: str) -> float:
    dv = _load_domain_vector()
    if dv is None or not text:
        return 1.0  # skip gating if unavailable
    q = np.array(get_embedding([text])[0], dtype="float32")
    q = q / (np.linalg.norm(q) + 1e-10)
    return float(np.dot(q, dv))


def is_product_relevant(text: str) -> bool:
    try:
        return relevance_score(text) >= DOMAIN_THRESHOLD
    except Exception:
        return True  # fail-open


# -----------------------------------------------------------------------------
# Phase canonicalisation & inference (avoid "testing phase" hallucinations)
# -----------------------------------------------------------------------------
def canonicalize_phase_name(raw: str, phases_list: List[str]) -> Optional[str]:
    """Map any free-text phase to the closest allowed phase."""
    if not raw:
        return None
    raw_l = raw.strip().lower()
    variant_map = {
        "development phase": "Development",
        "dev": "Development",
        "testing": "Development",
        "test": "Development",
        "qa": "Development",
        "uat": "Development",
        "golive": "Go-live",
        "go live": "Go-live",
    }
    if raw_l in variant_map:
        return variant_map[raw_l]
    for p in phases_list:
        if raw_l == p.lower():
            return p
    match = difflib.get_close_matches(raw, phases_list, n=1, cutoff=0.3)
    return match[0] if match else None


def infer_phase_from_hits_and_text(intent_text: str, guide_hits: list, phases_list: List[str]) -> str:
    """
    Order of evidence:
    (0) Explicit mentions in user text (phase names & keywords) — strongest
    (1) Guide phases from retrieved items — weighted by retrieval score
    (2) Keyword heuristic across phases
    (3) Fuzzy name fallback
    """
    scores = {p: 0.0 for p in phases_list}
    tl = (intent_text or "").lower()

    # (0) Boost explicit mentions of phase names (strong signal)
    for p in phases_list:
        name_tokens = [p.lower()]
        if p.lower() == "go-live":
            name_tokens += ["go live", "golive"]
        for tok in name_tokens:
            if tok in tl:
                scores[p] += 5.0

    # (0b) Boost phase keyword hits
    for p, kws in PHASE_KEYWORDS.items():
        for kw in kws:
            if kw in tl:
                scores[p] += 1.25

    # (1) Use phases found in retrieved guides (canonicalised), weighted by score
    for g in (guide_hits or []):
        it = g.get("item", {})
        s = float(g.get("score", 0.0))
        p_raw = (it.get("phase") or "").strip()
        p = canonicalize_phase_name(p_raw, phases_list)
        if p:
            scores[p] += 2.0 * (1.0 + s)

    # Best score if positive
    best_phase = max(scores.items(), key=lambda kv: kv[1])[0]
    if scores[best_phase] > 0:
        return best_phase

    # Fallbacks
    match = difflib.get_close_matches((intent_text or ""), phases_list, n=1, cutoff=0.1)
    return match[0] if match else "Development"


# -----------------------------------------------------------------------------
# Link allowlisting & output sanitation
# -----------------------------------------------------------------------------
def _allowed_links_from_guides(guide_hits):
    """Collect (title, url, id) triples from retrieved guides; dedupe by URL."""
    seen = set()
    allowed = []
    for g in guide_hits:
        it = g.get("item", {})
        title = (it.get("title") or "").strip()
        url = (it.get("url") or "").strip()
        gid = it.get("id") or ""
        if url and url not in seen:
            allowed.append((title, url, gid))
            seen.add(url)
    return allowed


def allowed_domains_from_guides(guide_hits):
    """Derive a domain allowlist from the URLs that exist in guide hits."""
    hosts = set()
    from urllib.parse import urlparse

    for g in guide_hits:
        url = (g.get("item", {}).get("url") or "").strip()
        if not url:
            continue
        try:
            h = (urlparse(url).hostname or "").lower()
            if h:
                hosts.add(h)
        except Exception:
            pass
    return tuple(sorted(hosts))


def filter_allowed_links_by_domain(allowed_links, allowed_domains):
    """Keep only links whose host matches one of the allowed domains."""
    if not allowed_domains:
        return allowed_links
    out = []
    from urllib.parse import urlparse

    for (title, url, gid) in allowed_links:
        try:
            h = (urlparse(url).hostname or "").lower()
            if any(h == d or h.endswith("." + d) for d in allowed_domains):
                out.append((title, url, gid))
        except Exception:
            continue
    return out


def strip_disallowed_html(md: str) -> str:
    """Neutralize risky HTML tags even though Streamlit is safe-ish."""
    return re.sub(
        r"<(script|iframe|object|embed|link|style)[^>]*>.*?</\1>",
        "",
        md,
        flags=re.I | re.S,
    )


def render_recommended_resources(guide_hits, allowed_links):
    """
    Render '### Recommended Resources' with single-line bullets:
    - "<Title>: <Purpose> (<High/Medium/Low> relevance)"
    Title is hyperlinked if URL is allowed; otherwise 'Contact administrator…' precedes the colon.
    """
    allowed_url_set = {u for _, u, _ in (allowed_links or [])}
    lines = []
    if not guide_hits:
        return "\n- No relevant resources found."

    for idx, g in enumerate(guide_hits):
        it = g.get("item", {})
        title = (it.get("title") or "Untitled").strip()
        url = (it.get("url") or "").strip()
        purpose = (it.get("purpose") or "").strip()

        # Relevance band by rank
        rel = "High" if idx < 2 else ("Medium" if idx < 5 else "Low")

        # Title piece
        if url and url in allowed_url_set:
            title_piece = f"[{title}]({url})"
        elif url and url not in allowed_url_set:
            title_piece = f"{title} — Contact administrator to access the guide."
        else:
            title_piece = f"{title} — Contact administrator to access the guide."

        lines.append(f"- {title_piece}: {purpose or '-'} ({rel} relevance)")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigate", ["Concierge", "Repository", "About Us", "Methodology"], index=0
)

# -----------------------------------------------------------------------------
# GUIDE PAGE
# -----------------------------------------------------------------------------
if page == "Concierge":
    # Do not continue if check_password is not True.  
    if not check_password():  
        st.stop()
    st.title("🧭 Product Delivery Concierge")
    st.caption("Your friendly guide for digital product delivery.")
    with st.expander("Disclaimer"):
        st.markdown("""
                    **IMPORTANT NOTICE:** This web application is developed as a proof-of-concept prototype.  
                    The information provided here is **NOT** intended for actual usage and should not be relied upon for making any decisions,  
                    especially those related to financial, legal, or healthcare matters.  

                    Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.  

                    Always consult with qualified professionals for accurate and personalized advice.
                    """)

    # Build / refresh lightweight indices
    with st.spinner("Preparing knowledge…"):
        ensure_built()

    col_left, col_right = st.columns([1, 1.4], gap="large")

    # Inputs (form so Cmd/Ctrl+Enter submits)
    with col_left:
        with st.form("guide_form", clear_on_submit=False):
            intent_text = st.text_area(
                "I want to…",
                height=140,
                placeholder=(
                    "e.g., know what I need to do during the Registration phase"
                ),
                help="Enter a concise task you want to complete.”",
            )
            role_options = ["(Not specified)"] + ROLES
            role_sel = st.selectbox("Role (optional)", role_options, index=0)
            role = "" if role_sel == "(Not specified)" else role_sel
            get_steps = st.form_submit_button("🔎 Get Guidance")

    with col_right:
        st.markdown("## Result")
        result_box = st.container()

        if get_steps:
            if not intent_text or not intent_text.strip():
                st.warning("Please describe what you want to do in the **I want to…** field.")
            else:
                scenario_obj = {
                    "scenario": intent_text.strip(),
                    "role": role.strip(),
                    "tags": [],  # kept for compatibility with helper if referenced; no UI field anymore
                }
                query = " ".join([scenario_obj["scenario"], scenario_obj["role"]]).strip()

                # Relevance gate
                if not is_product_relevant(scenario_obj["scenario"]):
                    with result_box:
                        st.error(
                            "Sorry, I can’t help with that. This assistant focuses on delivering digital "
                            "products—e.g., planning, approvals, development, testing, and go-live. "
                            "Please rephrase your request in that context."
                        )
                else:
                    with result_box:
                        with st.spinner("Generating your guidance…"):
                            # Retrieval
                            docx_hits = search_docx(query=query, phase=None, top_k=6)
                            guide_hits = search_guides(
                                query=query,
                                role=scenario_obj["role"],
                                phase=None,
                                keywords=None,  # no keywords field anymore
                                top_k=8,
                            )

                            # Allowed links (by guide, then domain)
                            allowed = _allowed_links_from_guides(guide_hits)
                            allowed_domains = allowed_domains_from_guides(guide_hits)
                            allowed = filter_allowed_links_by_domain(allowed, allowed_domains)

                            # Build LLM messages & get completion
                            messages = build_messages_two_source(
                                scenario=scenario_obj,
                                lifecycle_ctx=docx_hits,
                                guide_hits=guide_hits,
                                phase_hints=None,
                                mode="steps",
                                allowed_links=allowed,
                                phases_list=PHASE_ORDER,
                            )
                            answer = get_completion_by_messages(messages)

                            # Replace any non-whitelisted links with admin message
                            allowed_urls = {u for _, u, _ in allowed}

                            def _link_whitelist_repl(m):
                                text = m.group(1)
                                url = m.group(2)
                                return (
                                    m.group(0)
                                    if url in allowed_urls
                                    else f"{text} — Contact administrator to access the guide."
                                )

                            answer = re.sub(
                                r"\[([^\]]+)\]\((https?://[^)]+)\)", _link_whitelist_repl, answer
                            )

                            # ---- Build final markdown with canonical phase and H3 headers
                            # Extract model's "A" body to reuse helpful prose
                            m_a_sum = re.search(r"(A\)\s*Summary[\s\S]*?)(?=\nB\))", answer)
                            m_a_what = re.search(r"(A\)\s*What matters now[\s\S]*?)(?=\nB\))", answer)
                            a_body = ""
                            if m_a_sum:
                                a_body = m_a_sum.group(1)
                            elif m_a_what:
                                a_body = m_a_what.group(1).replace("A) What matters now", "A) Summary")
                            else:
                                a_body = "A) Summary\n" + answer.strip().split("\n\n")[0]

                            # Remove any free-text phase the model wrote
                            a_body = re.sub(r"You are likely in the .*?phase\.\s*", "", a_body, flags=re.I)

                            # Canonical phase (honours explicit mentions first)
                            canon_phase = infer_phase_from_hits_and_text(
                                scenario_obj["scenario"], guide_hits, PHASE_ORDER
                            )

                            summary_md = (
                                f"### Summary\n"
                                f"You are likely in the {canon_phase} phase.\n\n"
                                + a_body.split("\n", 1)[-1].strip()
                            )

                            # Deterministic recommended resources (single-line bullets)
                            recs_md = "### Recommended Resources\n" + render_recommended_resources(
                                guide_hits, allowed
                            )

                            final_md = summary_md + "\n\n" + recs_md
                            final_md = strip_disallowed_html(final_md)

                        # Render after spinner closes
                        st.markdown(final_md, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RESOURCES PAGE
# -----------------------------------------------------------------------------
elif page == "Repository":
    # Do not continue if check_password is not True.  
    if not check_password():  
        st.stop()
    st.title("Resources Repository")
    st.caption("Browse all guides. Search by title and filter by phase.")

    def _to_text(v):
        if v is None:
            return ""
        if isinstance(v, (int, float, bool)):
            return str(v)
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return ", ".join([_to_text(x) for x in v])
        if isinstance(v, dict):
            for k in ("name", "title", "text", "value", "label"):
                if k in v and isinstance(v[k], (str, int, float, bool)):
                    return str(v[k])
            try:
                return ", ".join([_to_text(x) for x in v.values()])
            except Exception:
                return str(v)
        return str(v)

    data_path = Path("data/guides.json")
    if not data_path.exists():
        st.warning("No guides.json found in data/")
    else:
        try:
            raw = json.loads(data_path.read_text(encoding="utf-8"))
            guides = raw.get("guides", raw if isinstance(raw, list) else [])
        except Exception as e:
            st.error(f"Failed to load guides: {e}")
            guides = []

        items = []
        phases_set = set()
        for g in guides:
            title = _to_text(g.get("title") or g.get("guide_name") or "Untitled")
            purpose = _to_text(g.get("purpose") or "")
            phase = _to_text(g.get("phase") or "")
            url = _to_text(g.get("url") or "")
            items.append(
                {
                    "title": title,
                    "purpose": purpose,
                    "phase": phase,
                    "url": url,
                }
            )
            if phase:
                phases_set.add(phase)

        # Search + optional phase filter
        search_q = st.text_input("Search by title", value="", placeholder="Type to filter by title…")
        f_phase = st.selectbox("Filter by Phase", ["(All)"] + sorted(list(phases_set)))

        def visible(it):
            if f_phase != "(All)" and (it["phase"] or "") != f_phase:
                return False
            if search_q.strip() and search_q.strip().lower() not in (it["title"] or "").lower():
                return False
            return True

        filtered = [it for it in items if visible(it)]
        filtered.sort(key=lambda x: x["title"].lower())

        st.write(f"Showing {len(filtered)} of {len(items)} resources.")
        for it in filtered:
            with st.container():
                st.markdown(f"### {it['title']}")
                st.markdown(f"**Purpose:** {it['purpose'] or '-'}")
                st.markdown(f"**Phase:** {it['phase'] or '-'}")
                if it["url"]:
                    st.markdown(f"[Open guide]({it['url']})")
                else:
                    st.markdown("Contact administrator to access the guide.")
                st.markdown("---")

# -----------------------------------------------------------------------------
# ABOUT / METHODOLOGY
# -----------------------------------------------------------------------------
elif page == "About Us":
    st.title("About Product Delivery Concierge")
    st.markdown(
        """
**Problem statement**  
In a large organisations, product delivery spans many phases, teams, processes, and guidelines.
Guidance sits across wikis, PDFs, and tribal knowledge. Users must translate their specific scenario into the right steps and resources.

**Objective**  
Ship a minimal, self-serve guide that captures the user's intent, infers the likely phase, assesses role responsibility,
and returns a **Summary** plus **Recommended Resources**.

**Scope (MVP)**  
- Single Streamlit app; two-source RAG (DOCX for process context, JSON for resources).
- Auto-build lightweight indices when inputs change.
- Outputs: **### Summary**, **### Recommended Resources**.
"""
    )

elif page == "Methodology":
    st.title("Methodology")
    st.markdown(
        """
**Two-source RAG**
- **Lifecycle Context (DOCX):** chunk + embed once; used to explain phase goals and safety checks (for **Summary**).
- **Recommended Guides (JSON):** one compact embedding per guide; used to rank resources (for **Recommended Resources**).

**Retrieval**
- Query = Intent + optional Role.
- Parallel search across DOCX and JSON.
- Links are whitelisted from JSON only; if missing, we show “Contact administrator to access the guide.”
- No citations shown.

**Output**
- **### Summary:** infer likely Phase (snapped to your canonical list) and recap goals; include role-responsibility note.
- **### Recommended Resources:** single-line bullets — **Title** (hyperlinked) + `Purpose: …` + `(High/Medium/Low relevance)`.
"""
    )


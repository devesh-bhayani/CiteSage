"""CiteSage Streamlit UI.

Three tabs backed by the CiteSage FastAPI service:
    Query  — ask a question, view cited answer + source chunks
    Ingest — upload a document (.pdf / .md / .html / .txt)
    Stats  — in-process usage / cost counters

Run locally
-----------
    uvicorn citesage.api.main:app --reload         # terminal 1
    streamlit run src/citesage/ui/app.py           # terminal 2

Environment
-----------
    CITESAGE_API_URL   default "http://localhost:8000"
    CITESAGE_API_KEY   sent as X-API-Key if set
"""

from __future__ import annotations

import os

import requests
import streamlit as st

from ..ingestion.loaders import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES


DEFAULT_API_URL = os.environ.get("CITESAGE_API_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.environ.get("CITESAGE_API_KEY", "")


def _headers(api_key: str) -> dict:
    return {"X-API-Key": api_key} if api_key else {}


def _api_get(url: str, api_key: str, path: str) -> tuple[int, dict | str]:
    try:
        r = requests.get(f"{url}{path}", headers=_headers(api_key), timeout=30)
    except requests.RequestException as exc:
        return 0, f"network error: {exc}"
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


def _api_post_json(
    url: str, api_key: str, path: str, payload: dict, timeout: int = 120
) -> tuple[int, dict | str]:
    try:
        r = requests.post(
            f"{url}{path}",
            json=payload,
            headers=_headers(api_key),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return 0, f"network error: {exc}"
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


def _api_post_file(
    url: str, api_key: str, path: str, name: str, data: bytes
) -> tuple[int, dict | str]:
    try:
        r = requests.post(
            f"{url}{path}",
            files={"file": (name, data)},
            headers=_headers(api_key),
            timeout=300,
        )
    except requests.RequestException as exc:
        return 0, f"network error: {exc}"
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


# ---------------------------------------------------------------------------
# Page setup + sidebar (connection settings)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CiteSage", page_icon=":books:", layout="wide")

with st.sidebar:
    st.title("CiteSage")
    st.caption("Document QA with verified citations.")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    api_key = st.text_input(
        "API Key (X-API-Key)",
        value=DEFAULT_API_KEY,
        type="password",
        help="Leave empty if the server is in dev/open mode.",
    )

    if st.button("Check /health", use_container_width=True):
        code, body = _api_get(api_url, api_key, "/health")
        if code == 200 and isinstance(body, dict):
            st.success(
                f"Status: {body.get('status')} — provider {body.get('provider')}"
            )
            st.json(body.get("checks", {}))
        else:
            st.error(f"Unhealthy ({code}): {body}")


tab_query, tab_ingest, tab_stats = st.tabs(["Query", "Ingest", "Stats"])


# ---------------------------------------------------------------------------
# Query tab
# ---------------------------------------------------------------------------

with tab_query:
    st.header("Ask a question")
    question = st.text_area(
        "Question",
        placeholder="e.g. What is self-attention in the Transformer architecture?",
        height=100,
    )
    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Running pipeline..."):
            code, body = _api_post_json(
                api_url, api_key, "/query", {"question": question}, timeout=300
            )
        if code != 200 or not isinstance(body, dict):
            st.error(f"Query failed ({code}): {body}")
        else:
            declined = body.get("declined", False)
            path = body.get("path_taken", "")
            confidence = body.get("confidence", "")

            if declined:
                st.warning(
                    f"Declined (path={path}, confidence={confidence}). "
                    "No relevant sources found."
                )
            st.subheader("Answer")
            st.write(body.get("answer", ""))

            badge_cols = st.columns(4)
            badge_cols[0].metric("Path", path or "—")
            badge_cols[1].metric("Confidence", confidence or "—")
            cost = body.get("cost_usd", 0.0)
            badge_cols[2].metric("Cost (USD)", f"${cost:.6f}")
            usage = body.get("token_usage", {}) or {}
            tokens = int(usage.get("input_tokens", 0) or 0) + int(
                usage.get("output_tokens", 0) or 0
            )
            badge_cols[3].metric("Tokens", tokens)

            citations = body.get("citations", []) or []
            if citations:
                st.subheader(f"Sources ({len(citations)})")
                for i, cite in enumerate(citations, start=1):
                    src = cite.get("source_file", "?")
                    page = cite.get("page_number")
                    score = cite.get("score", 0.0)
                    header = f"[{i}] {src}"
                    if page is not None:
                        header += f" · p.{page}"
                    header += f" · score {score:.3f}"
                    with st.expander(header):
                        heading = cite.get("section_heading")
                        if heading:
                            st.caption(heading)
                        st.write(cite.get("content_preview", ""))
                        st.caption(f"chunk_id: `{cite.get('chunk_id', '')}`")
            elif not declined:
                st.info("Answered with no citations.")

            with st.expander("Raw response"):
                st.json(body)


# ---------------------------------------------------------------------------
# Ingest tab
# ---------------------------------------------------------------------------

with tab_ingest:
    st.header("Ingest a document")
    st.caption(
        f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))} — "
        f"max {MAX_FILE_SIZE_BYTES // 1_048_576} MB."
    )

    allowed_exts = [ext.lstrip(".") for ext in sorted(ALLOWED_EXTENSIONS)]
    upload = st.file_uploader("Choose a file", type=allowed_exts)

    if upload is not None:
        size_mb = upload.size / 1_048_576 if upload.size else 0
        st.caption(f"{upload.name} — {size_mb:.2f} MB")
        if st.button("Ingest", type="primary"):
            with st.spinner("Chunking and indexing..."):
                code, body = _api_post_file(
                    api_url, api_key, "/ingest", upload.name, upload.getvalue()
                )
            if code == 200 and isinstance(body, dict):
                st.success(
                    f"Ingested {body.get('chunks_ingested', 0)} chunk(s) from "
                    f"{body.get('filename', upload.name)}"
                )
                st.json(body)
            else:
                st.error(f"Ingest failed ({code}): {body}")


# ---------------------------------------------------------------------------
# Stats tab
# ---------------------------------------------------------------------------

with tab_stats:
    st.header("Usage stats (this API process)")
    if st.button("Refresh"):
        st.rerun()

    code, body = _api_get(api_url, api_key, "/stats")
    if code == 200 and isinstance(body, dict):
        cols = st.columns(4)
        cols[0].metric("Queries", body.get("query_count", 0))
        cols[1].metric("Declined", body.get("declined_count", 0))
        cols[2].metric("Fast path", body.get("fast_path_count", 0))
        cols[3].metric("Thorough path", body.get("thorough_path_count", 0))

        cols = st.columns(3)
        cols[0].metric(
            "Total tokens",
            (body.get("total_input_tokens", 0) + body.get("total_output_tokens", 0)),
        )
        cols[1].metric("Total cost (USD)", f"${body.get('total_cost_usd', 0.0):.6f}")
        cols[2].metric(
            "Avg cost/query (USD)",
            f"${body.get('average_cost_per_query_usd', 0.0):.6f}",
        )

        st.caption(f"Uptime: {body.get('uptime_seconds', 0.0):.1f} s")
        with st.expander("Raw"):
            st.json(body)
    else:
        st.error(f"Stats unavailable ({code}): {body}")

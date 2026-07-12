"""
NBKR Institute AI Chatbot — RAG + NLP + ML v7.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ML Layer   (Supervised — scikit-learn):
  • TF-IDF vectoriser  (1-2 grams, 3000 features)
  • Multinomial Naive Bayes intent classifier
  • Trained on labelled intent examples at startup
  • Predicts: greeting / farewell / help / timetable / faculty / services / general / student

NLP Layer  (spaCy en_core_web_sm):
  • Lemmatisation, POS tagging, NER, query expansion

RAG Layer:
  • Embeddings : sentence-transformers/all-MiniLM-L6-v2
  • Retrieval  : FAISS IndexFlatIP (cosine similarity)
  • Confidence : threshold 0.30

Student Data Warehouse:
  • Source     : student_data_warehouse.json (extracted from PDF)
  • CRUD       : GET /students, POST /students, PUT /students/{roll}, DELETE /students/{roll}
  • Chat       : Roll-number lookup, name search, section/branch/CGPA mining

UI:
  • All responses rendered as structured HTML (tables / cards)
  • Frontend correctly renders innerHTML for ALL bot messages
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json, os, re, numpy as np, uvicorn
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────
chat_history: List[Dict] = []
active_connections: List[WebSocket] = []

# ─────────────────────────────────────────────────────────────────────────────
# Student Data Warehouse — CRUD
# ─────────────────────────────────────────────────────────────────────────────
_STUDENT_DW_PATH = "student_data_warehouse.json"
_students: List[Dict] = []
_roll_index: Dict[str, Dict] = {}

def _load_student_dw():
    """Load student data warehouse from JSON file into memory."""
    global _students, _roll_index
    if not os.path.exists(_STUDENT_DW_PATH):
        _students, _roll_index = [], {}
        return
    with open(_STUDENT_DW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for s in data:
        s["roll_no"] = s.get("roll_no", "").strip().upper()
    _students = data
    _roll_index = {s["roll_no"]: s for s in _students}
    print(f"✓ Student Data Warehouse loaded — {len(_students)} records")

def _save_student_dw():
    """Persist in-memory student list back to JSON."""
    with open(_STUDENT_DW_PATH, "w", encoding="utf-8") as f:
        json.dump(_students, f, indent=2, ensure_ascii=False)

def _sv(v):
    """Return clean string value or 'N/A'."""
    return str(v).strip() if v and str(v).strip() not in ("", "None") else "N/A"

# ── CRUD helpers ──────────────────────────────────────────────────────────────
def student_get(roll: str) -> Optional[Dict]:
    return _roll_index.get(roll.strip().upper())

def student_create(data: Dict) -> Tuple[bool, str]:
    roll = data.get("roll_no", "").strip().upper()
    if not roll:
        return False, "roll_no is required"
    if roll in _roll_index:
        return False, f"Student {roll} already exists. Use update to modify."
    data["roll_no"] = roll
    _students.append(data)
    _roll_index[roll] = data
    _save_student_dw()
    return True, f"Student {roll} added successfully."

def student_update(roll: str, updates: Dict) -> Tuple[bool, str]:
    roll = roll.strip().upper()
    if roll not in _roll_index:
        return False, f"Student {roll} not found."
    updates.pop("roll_no", None)          # roll_no is immutable
    _roll_index[roll].update(updates)
    _save_student_dw()
    return True, f"Student {roll} updated successfully."

def student_delete(roll: str) -> Tuple[bool, str]:
    roll = roll.strip().upper()
    if roll not in _roll_index:
        return False, f"Student {roll} not found."
    _students.remove(_roll_index[roll])
    del _roll_index[roll]
    _save_student_dw()
    return True, f"Student {roll} deleted."

# ── Data mining helpers ───────────────────────────────────────────────────────
def _mine_by_name(q: str) -> List[Dict]:
    q = q.lower().strip()
    return [s for s in _students if q in s.get("name","").lower()]

def _mine_by_section(sec: str) -> List[Dict]:
    return [s for s in _students if s.get("section","").upper() == sec.upper()]

def _mine_by_branch(branch: str) -> List[Dict]:
    b = branch.upper()
    return [s for s in _students if b in s.get("branch","").upper()]

def _mine_cgpa(low: float, high: float) -> List[Dict]:
    result = []
    for s in _students:
        try:
            if low <= float(s.get("cgpa", "")) <= high:
                result.append(s)
        except (ValueError, TypeError):
            pass
    return sorted(result, key=lambda x: float(x.get("cgpa", 0)), reverse=True)

def _mine_toppers(n: int = 5) -> List[Dict]:
    valid = [s for s in _students if s.get("cgpa") not in ("", "N/A", None)]
    try:
        return sorted(valid, key=lambda x: float(x.get("cgpa", 0)), reverse=True)[:n]
    except Exception:
        return []

def _mine_stats() -> Dict:
    total = len(_students)
    branch_counts: Dict[str, int] = {}
    section_counts: Dict[str, int] = {}
    cgpa_vals = []
    for s in _students:
        b = s.get("branch") or "Unknown"
        branch_counts[b] = branch_counts.get(b, 0) + 1
        sec = s.get("section") or "N/A"
        section_counts[sec] = section_counts.get(sec, 0) + 1
        try:
            cgpa_vals.append(float(s.get("cgpa", "")))
        except (ValueError, TypeError):
            pass
    avg_cgpa = round(sum(cgpa_vals) / len(cgpa_vals), 2) if cgpa_vals else 0
    return {"total": total, "avg_cgpa": avg_cgpa,
            "branch_counts": branch_counts, "section_counts": section_counts}

# ── HTML builders ─────────────────────────────────────────────────────────────
_STH = 'style="border:1px solid #c5cae9;padding:9px 12px;background:#e8eaf6;font-size:12px;font-weight:700;color:#1a237e;text-align:left"'
_STD = 'style="border:1px solid #e8eaf6;padding:8px 12px;font-size:12px;color:#1a1a2e"'

def _sbadge(text, fg="#1a237e", bg="#e8eaf6"):
    return (f'<span style="background:{bg};color:{fg};padding:2px 9px;'
            f'border-radius:10px;font-size:11px;font-weight:600">{text}</span>')

def build_student_card(s: Dict) -> str:
    branch = _sv(s.get("branch"))
    sec    = _sv(s.get("section"))
    cgpa   = _sv(s.get("cgpa"))
    phone  = _sv(s.get("phone"))
    email  = _sv(s.get("email"))
    school = _sv(s.get("school"))
    tenth  = _sv(s.get("tenth_percentage"))
    inter  = _sv(s.get("inter_diploma"))
    inter_p= _sv(s.get("inter_percentage"))

    try:
        cval = float(cgpa)
        cbg = "#e8f5e9" if cval >= 8 else ("#fff3e0" if cval >= 7 else "#fce4ec")
        cfg = "#2e7d32" if cval >= 8 else ("#e65100" if cval >= 7 else "#c62828")
    except ValueError:
        cbg, cfg = "#e8eaf6", "#1a237e"

    def row(key, val, hi=False):
        bg = "background:#f0f4ff;" if hi else ""
        return (f'<div style="display:flex;border-bottom:1px solid #e8eaf6;'
                f'padding:10px 16px;font-size:13px;align-items:center;{bg}">'
                f'<span style="min-width:170px;font-weight:600;color:#555;font-size:12px;'
                f'text-transform:uppercase;letter-spacing:.03em">{key}</span>'
                f'<span style="color:#1a1a2e;font-size:13px">{val}</span></div>')

    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif;max-width:620px">
  <div style="background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
              padding:12px 18px;border-radius:10px 10px 0 0">
    <div style="font-size:15px;font-weight:700">🎓 {s.get('name','—')}</div>
    <div style="font-size:11px;opacity:.8;margin-top:2px">{s.get('roll_no','')}</div>
  </div>
  <div style="background:#fff;border:1px solid #c5cae9;border-top:none;
              border-radius:0 0 10px 10px;overflow:hidden">
    {row("📋 Roll Number", f"<b>{s.get('roll_no','—')}</b>", True)}
    {row("👤 Name", s.get('name','—'))}
    {row("📱 Phone", f'<a href="tel:{phone}" style="color:#1a237e;text-decoration:none">{phone}</a>')}
    {row("📧 Email", f'<a href="mailto:{email}" style="color:#1a237e;text-decoration:none">{email}</a>', True)}
    {row("🏫 Branch", _sbadge(branch, "#1b5e20", "#e8f5e9") if "AI" in branch.upper() else _sbadge(branch))}
    {row("🔠 Section", _sbadge(f"Section {sec}", "#4a148c", "#f3e5f5") if sec not in ("N/A","-","") else sec)}
    {row("🏆 CGPA", _sbadge(f"CGPA: {cgpa}", cfg, cbg), True)}
    {row("🏛️ School", school)}
    {row("📊 10th %", tenth)}
    {row("📚 Inter/Diploma", inter)}
    {row("📈 Inter/Dip %", inter_p, True)}
  </div>
</div>"""

def build_students_table(rows_data: List[Dict], title: str) -> str:
    if not rows_data:
        return '<p style="font-family:Segoe UI,sans-serif;color:#c62828">No records found.</p>'
    body = ""
    for i, s in enumerate(rows_data, 1):
        bg = "background:#f8f9ff;" if i % 2 == 0 else ""
        body += (f'<tr style="{bg}">'
                 f'<td {_STD}>{i}</td>'
                 f'<td {_STD}><b>{s.get("roll_no","")}</b></td>'
                 f'<td {_STD}>{s.get("name","")}</td>'
                 f'<td {_STD}>{_sv(s.get("phone"))}</td>'
                 f'<td {_STD}>{_sv(s.get("email"))}</td>'
                 f'<td {_STD}>{_sv(s.get("branch"))}</td>'
                 f'<td {_STD}>{_sv(s.get("section"))}</td>'
                 f'<td {_STD}>{_sv(s.get("cgpa"))}</td>'
                 f'</tr>')
    return f"""
<div style="font-family:'Segoe UI',sans-serif;margin:8px 0">
  <div style="background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
              padding:10px 16px;border-radius:8px 8px 0 0;
              display:flex;justify-content:space-between;align-items:center">
    <b>{title}</b>
    <span style="font-size:11px;opacity:.8">{len(rows_data)} record(s)</span>
  </div>
  <div style="overflow-x:auto;border:1px solid #c5cae9;border-top:none;border-radius:0 0 8px 8px">
    <table style="width:100%;border-collapse:collapse;min-width:700px">
      <thead><tr>
        <th {_STH}>#</th><th {_STH}>Roll No</th><th {_STH}>Name</th>
        <th {_STH}>Phone</th><th {_STH}>Email</th>
        <th {_STH}>Branch</th><th {_STH}>Section</th><th {_STH}>CGPA</th>
      </tr></thead>
      <tbody>{body}</tbody>
    </table>
  </div>
</div>"""

def build_student_stats_card() -> str:
    st = _mine_stats()
    bc = "".join(
        f'<tr><td style="padding:5px 10px;border:1px solid #e8eaf6;font-size:12px">{b}</td>'
        f'<td style="padding:5px 10px;border:1px solid #e8eaf6;font-size:12px;font-weight:700">{c}</td></tr>'
        for b, c in sorted(st["branch_counts"].items(), key=lambda x: -x[1])
    )
    sc = "".join(
        f'<tr><td style="padding:5px 10px;border:1px solid #e8eaf6;font-size:12px">Section {s}</td>'
        f'<td style="padding:5px 10px;border:1px solid #e8eaf6;font-size:12px;font-weight:700">{c}</td></tr>'
        for s, c in sorted(st["section_counts"].items(), key=lambda x: -x[1])
    )
    return f"""
<div style="font-family:'Segoe UI',sans-serif;margin:8px 0">
  <div style="background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
              padding:10px 16px;border-radius:8px 8px 0 0">
    <b>📊 Student Analytics — Department Report</b>
  </div>
  <div style="background:#fff;border:1px solid #c5cae9;border-top:none;
              border-radius:0 0 8px 8px;padding:14px 18px">
    <div style="display:flex;gap:18px;flex-wrap:wrap;margin-bottom:14px">
      <div style="background:#e8eaf6;border-radius:8px;padding:12px 20px;text-align:center;min-width:90px">
        <div style="font-size:24px;font-weight:700;color:#1a237e">{st['total']}</div>
        <div style="font-size:11px;color:#555">Total Students</div>
      </div>
      <div style="background:#e8f5e9;border-radius:8px;padding:12px 20px;text-align:center;min-width:90px">
        <div style="font-size:24px;font-weight:700;color:#2e7d32">{st['avg_cgpa']}</div>
        <div style="font-size:11px;color:#555">Avg CGPA</div>
      </div>
    </div>
    <div style="display:flex;gap:24px;flex-wrap:wrap">
      <div>
        <div style="font-weight:700;color:#1a237e;font-size:12px;margin-bottom:6px">Branch Distribution</div>
        <table style="border-collapse:collapse"><tbody>{bc}</tbody></table>
      </div>
      <div>
        <div style="font-weight:700;color:#1a237e;font-size:12px;margin-bottom:6px">Section Distribution</div>
        <table style="border-collapse:collapse"><tbody>{sc}</tbody></table>
      </div>
    </div>
  </div>
</div>"""

# ── Chat-level student query handler ─────────────────────────────────────────
_ROLL_PAT = re.compile(r'\b(2[34]KB[15]A\w{4,6})\b', re.IGNORECASE)

def handle_student_query(query: str) -> Optional[str]:
    """
    Returns an HTML response if the query is about student info, else None.
    Handles: roll-number lookup, name search, section/branch filter,
             CGPA mining, top-N, stats, all-students list.
    """
    ml = query.lower().strip()

    # ── TIMETABLE GUARD — never intercept timetable queries ──────────────
    timetable_kw = {"timetable","time table","schedule","class","period","timing",
                    "lecture","lab","1st yr","2nd yr","3rd yr","1st year","2nd year",
                    "3rd year","first year","second year","third year","monday",
                    "tuesday","wednesday","thursday","friday","saturday","weekly",
                    "slot","semester","sem "}
    if any(kw in ml for kw in timetable_kw):
        return None  # Let timetable handler deal with it

    # Roll number — highest priority
    rm = _ROLL_PAT.search(query)
    if rm:
        s = student_get(rm.group(1))
        if s:
            return build_student_card(s)
        return (f'<p style="font-family:Segoe UI,sans-serif;color:#c62828">'
                f'❌ Roll number <b>{rm.group(1).upper()}</b> not found in the student database.</p>')

    # Only proceed if there's a student-specific keyword
    # NOTE: "section" alone is NOT here — it's too ambiguous with timetable sections.
    # Only "student section" combos are handled below.
    student_kw = {"student","roll","rollno","roll no","roll number","admission",
                  "topper","cgpa","all students","student list","student info",
                  "student detail","student stats","student report","student analytics"}
    if not any(kw in ml for kw in student_kw):
        return None  # Not a student query — fall through to existing RAG

    # Top-N students
    if any(w in ml for w in ["topper","top student","highest cgpa","best student","rank"]):
        n = 10
        nm = re.search(r'\d+', ml)
        if nm:
            n = int(nm.group())
        return build_students_table(_mine_toppers(n), f"🏆 Top {n} Students by CGPA")

    # CGPA range mining
    cgpa_m = re.search(
        r'cgpa\s*(above|below|between|>|<|>=|<=)?\s*(\d+\.?\d*)\s*(?:and|to)?\s*(\d+\.?\d*)?', ml)
    if cgpa_m:
        grp1, grp2, grp3 = cgpa_m.group(1), cgpa_m.group(2), cgpa_m.group(3)
        if grp3:
            lo, hi = float(grp2), float(grp3)
        elif grp1 in ("above", ">", ">="):
            lo, hi = float(grp2), 10.0
        elif grp1 in ("below", "<", "<="):
            lo, hi = 0.0, float(grp2)
        else:
            lo, hi = float(grp2), 10.0
        return build_students_table(_mine_cgpa(lo, hi), f"📈 Students with CGPA {lo}–{hi}")

    # Section filter — only when "student" is also present
    if "student" in ml:
        sec_m = re.search(r'\bsection\s*([abcdABCD])\b', ml)
        if sec_m:
            sec = sec_m.group(1).upper()
            return build_students_table(_mine_by_section(sec), f"🔠 Section {sec} Students")

    # Branch filter — only when "student" is also present
    if "student" in ml:
        if any(w in ml for w in ["ai&ds","aids","ai ds"]):
            return build_students_table(_mine_by_branch("AI"), "🤖 AI&DS Branch Students")
        if "it branch" in ml or ("it" in ml and "student" in ml):
            return build_students_table(_mine_by_branch("IT"), "💻 IT Branch Students")

    # Stats
    if any(w in ml for w in ["stats","statistics","analytics","report","summary",
                              "count","total student","student report","student analytics"]):
        return build_student_stats_card()

    # All students
    if any(w in ml for w in ["all students","list all","show all students","student list"]):
        return build_students_table(_students, f"📋 All Students ({len(_students)} records)")

    # Name search
    name_m = re.search(
        r'(?:find|search|who is|info(?:rmation)?\s+(?:of|about)?|student\s+(?:named|called)?)\s+(.+)', ml)
    if name_m:
        q_name = name_m.group(1).strip()
        hits = _mine_by_name(q_name)
        if len(hits) == 1:
            return build_student_card(hits[0])
        if hits:
            return build_students_table(hits, f"🔍 Search: '{q_name}'")
        return (f'<p style="font-family:Segoe UI,sans-serif;color:#c62828">'
                f'No student found matching "<b>{q_name}</b>".</p>')

    return None  # let RAG handle it


def handle_student_timetable_query(query: str) -> str:
    """
    Called when the user asks about a student's timetable.
    Strategy:
      1. If a roll number is present → look up the student's section → show that section's timetable.
      2. If a name is present → look up their section → show timetable.
      3. If only "student timetable" (no roll/name) → show year-wise overview.
    """
    q = query.lower().strip()

    # ── 1. Roll number lookup ────────────────────────────────────────────
    rm = _ROLL_PAT.search(query)
    if rm:
        s = student_get(rm.group(1))
        if s:
            sec_raw = s.get("section", "").strip().upper()   # e.g. "A", "B"
            branch  = s.get("branch", "").upper()
            name    = s.get("name", rm.group(1))

            # Determine year from roll number (3rd char = year digit for some schemas)
            # Also accept explicit year from query
            year = None
            ym = re.search(r'\b([1-4])(?:st|nd|rd|th)?\s*year\b', q)
            if ym:
                year = int(ym.group(1))

            # Map section letter → timetable key (try all years, prefer year match)
            candidates = []
            if sec_raw:
                for yr_num, prefix in [(1,"1st_Year"), (2,"2nd_Year"),
                                       (3,"3rd_Year"), (4,"4th_Year")]:
                    key = f"{prefix}_Section_{sec_raw}"
                    if key in (_TT_DATA.get("timetable") or {}):
                        candidates.append((yr_num, key))

            if candidates:
                # Pick the year that was asked, else show overview card
                if year:
                    match = next((k for y, k in candidates if y == year), None)
                    if match:
                        header = _student_tt_header(name, s, match)
                        return header + build_section_week_table(match)
                # More than one year available → show student card + overview
                header = _student_tt_header(name, s, None)
                return header + build_yearwise_timetable_overview()
            else:
                # No timetable data for their section
                header = _student_tt_header(name, s, None)
                return (header +
                        f'<p style="font-family:Segoe UI,sans-serif;color:#e65100;padding:10px">'
                        f'⚠️ No timetable data found for Section <b>{sec_raw or "?"}</b>. '
                        f'Try asking for a specific year, e.g. "1st year Section {sec_raw or "A"} timetable".</p>')

        # Roll not found in DB
        return (f'<p style="font-family:Segoe UI,sans-serif;color:#c62828">'
                f'❌ Roll number <b>{rm.group(1).upper()}</b> not found in the student database.</p>')

    # ── 2. Name-based lookup ─────────────────────────────────────────────
    name_m = re.search(
        r'(?:student|of|for|named?|called?)\s+([a-z][a-z\s]{2,30}?)(?:\s+timetable|\s+schedule|$)', q)
    if name_m:
        hits = _mine_by_name(name_m.group(1).strip())
        if len(hits) == 1:
            s = hits[0]
            sec_raw = s.get("section", "").strip().upper()
            name    = s.get("name", "Student")
            candidates = []
            if sec_raw:
                for yr_num, prefix in [(1,"1st_Year"), (2,"2nd_Year"),
                                       (3,"3rd_Year"), (4,"4th_Year")]:
                    key = f"{prefix}_Section_{sec_raw}"
                    if key in (_TT_DATA.get("timetable") or {}):
                        candidates.append((yr_num, key))
            if candidates:
                header = _student_tt_header(name, s, candidates[0][1])
                return header + build_section_week_table(candidates[0][1])
            header = _student_tt_header(name, s, None)
            return header + build_yearwise_timetable_overview()
        if len(hits) > 1:
            return build_students_table(hits, f"🔍 Multiple matches — which student?")

    # ── 3. Generic "student timetable" → year-wise overview ──────────────
    return build_yearwise_timetable_overview()


def _student_tt_header(name: str, student: Dict, tt_key) -> str:
    """Small info banner shown above the timetable for a specific student."""
    sec  = student.get("section", "—")
    roll = student.get("roll_no", "—")
    yr   = tt_key.replace("_Section_"," · Section ").replace("_Year"," Year").replace("_"," ") if tt_key else "—"
    return f"""
<div style="font-family:'Segoe UI',sans-serif;margin-bottom:4px;
            background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
            padding:10px 16px;border-radius:8px;
            display:flex;gap:18px;flex-wrap:wrap;align-items:center">
  <span style="font-size:14px;font-weight:700">🎓 {name}</span>
  <span style="font-size:11px;opacity:.85">Roll: <b>{roll}</b></span>
  <span style="font-size:11px;opacity:.85">Section: <b>{sec}</b></span>
  {f'<span style="font-size:11px;opacity:.85">Timetable: <b>{yr}</b></span>' if tt_key else ''}
</div>"""

embeddings_model  = None
faiss_index       = None
doc_embeddings    = None   # precomputed document embeddings matrix
knowledge_docs: List[Dict] = []
bm25_index        = None   # BM25Okapi index over knowledge_docs
bm25_tokenized_corpus: List[List[str]] = []

_BM25_STOPWORDS = {
    "the","is","are","a","an","of","to","in","on","for","and","or",
    "tell","me","about","what","how","can","you","do","does","did",
    "please","i","my","it","this","that","was","were","be","been",
    "with","at","by","from","as","if","will","would","should","could"
}

def _bm25_tokenize(text: str) -> List[str]:
    """Tokenize for BM25, stripping common filler words that would
    otherwise inflate matches on off-topic queries."""
    tokens = re.findall(r'\w+', text.lower())
    return [t for t in tokens if t not in _BM25_STOPWORDS]
reranker_model    = None   # CrossEncoder model for reranking top candidates
nlp               = None   # spaCy
ml_classifier     = None   # TF-IDF + Naive Bayes pipeline
ml_vectorizer     = None
audit_file        = "admin_audit.json"

CONFIDENCE_THRESHOLD = 0.70
TOP_K = 1
# Slightly more lenient threshold for the "unknown" fallback, since it's
# searching across ALL document types (not a single filtered type), and
# hybrid scores blend two different signals (vector + BM25).
UNKNOWN_FALLBACK_THRESHOLD = 0.55


# ─────────────────────────────────────────────────────────────────────────────
# Supervised ML — Training data & classifier
# ─────────────────────────────────────────────────────────────────────────────
_TRAIN_DATA = [
    # greeting
    ("hello", "greeting"), ("hi", "greeting"), ("hey", "greeting"),
    ("good morning", "greeting"), ("good afternoon", "greeting"),
    ("good evening", "greeting"), ("hi there", "greeting"),
    ("howdy", "greeting"), ("what's up", "greeting"), ("greetings", "greeting"),
    # farewell
    ("bye", "farewell"), ("goodbye", "farewell"), ("thank you", "farewell"),
    ("thanks", "farewell"), ("thank you so much", "farewell"),
    ("ok thanks", "farewell"), ("that's all", "farewell"),
    ("see you", "farewell"), ("appreciate it", "farewell"),
    # help
    ("help", "help"), ("what can you do", "help"), ("what do you know", "help"),
    ("how can you help", "help"), ("capabilities", "help"),
    ("what topics", "help"), ("what can i ask", "help"),
    ("show me options", "help"), ("menu", "help"),
    # timetable — year-wise overview
    ("all timetables", "timetable"), ("year wise timetable", "timetable"),
    ("timetable overview", "timetable"), ("all years timetable", "timetable"),
    ("show all timetables", "timetable"), ("timetable index", "timetable"),
    ("all sections timetable", "timetable"), ("yearwise timetable", "timetable"),
    # timetable — section specific
    ("show section a timetable", "timetable"), ("section b schedule", "timetable"),
    ("section c timetable", "timetable"), ("section d timetable", "timetable"),
    ("timetable for section a", "timetable"), ("what is the timetable", "timetable"),
    ("class schedule", "timetable"), ("show timetable", "timetable"),
    ("monday schedule section a", "timetable"), ("section a monday", "timetable"),
    ("cp lab schedule", "timetable"), ("engineering physics timetable", "timetable"),
    ("beee schedule", "timetable"), ("all sections timetable", "timetable"),
    ("section time table", "timetable"), ("table format timetable", "timetable"),
    ("weekly schedule", "timetable"), ("class timing", "timetable"),
    ("lecture schedule", "timetable"), ("period schedule", "timetable"),
    # 2nd year timetable
    ("2nd year section a timetable", "timetable"), ("second year section b", "timetable"),
    ("2nd year timetable", "timetable"), ("second year schedule", "timetable"),
    ("2nd year section a", "timetable"), ("2nd year section b", "timetable"),
    ("ai lab schedule", "timetable"), ("ids lab timetable", "timetable"),
    ("full stack lab schedule", "timetable"), ("dti lab timetable", "timetable"),
    ("artificial intelligence schedule", "timetable"), ("smds timetable", "timetable"),
    # 3rd year timetable
    ("3rd year section a timetable", "timetable"), ("third year section b", "timetable"),
    ("3rd year timetable", "timetable"), ("third year schedule", "timetable"),
    ("3rd year section a", "timetable"), ("3rd year section b", "timetable"),
    ("deep learning lab schedule", "timetable"), ("nlp timetable", "timetable"),
    ("big data analytics schedule", "timetable"), ("social network analysis timetable", "timetable"),
    ("soft skills lab schedule", "timetable"), ("workshop timetable", "timetable"),
    # faculty
    ("who is the hod", "faculty"), ("head of department", "faculty"),
    ("list all faculty", "faculty"), ("faculty members", "faculty"),
    ("who teaches machine learning", "faculty"), ("professor list", "faculty"),
    ("show faculty", "faculty"), ("who is dr", "faculty"),
    ("assistant professor", "faculty"), ("associate professor", "faculty"),
    ("faculty specialization", "faculty"), ("how many faculty", "faculty"),
    ("staff members", "faculty"), ("teachers list", "faculty"),
    ("who are the lecturers", "faculty"), ("faculty details", "faculty"),
    # services
    ("how to check attendance", "services"), ("attendance system", "services"),
    ("e-journal", "services"), ("ejournal", "services"),
    ("online portal", "services"), ("intranet", "services"),
    ("assessment tool", "services"), ("exam duties", "services"),
    ("how to login", "services"), ("portal access", "services"),
    ("library timings", "services"), ("hostel", "services"),
    ("fee structure", "services"), ("admission process", "services"),
    ("placement record", "services"), ("courses offered", "services"),
    ("exam schedule", "services"), ("results", "services"),
    # general
    ("tell me about nbkr", "general"), ("about the college", "general"),
    ("nbkr institute", "general"), ("what is nbkr", "general"),
    ("college information", "general"), ("department info", "general"),
    # college info
    ("about nbkr", "college"), ("nbkr college", "college"),
    # labs R23
    ("what labs are in 2nd year", "labs"), ("labs in iii-i", "labs"),
    ("ai&ds labs", "labs"), ("all labs r23", "labs"),
    ("which labs in sem 1", "labs"), ("labs in 1st year", "labs"),
    ("cp lab semester", "labs"), ("ai lab which year", "labs"),
    ("ml lab semester", "labs"), ("dl lab year", "labs"),
    ("fsd lab", "labs"), ("oop java lab", "labs"),
    ("data structure lab", "labs"), ("ds lab", "labs"),
    ("bda lab", "labs"), ("tinkering lab", "labs"),
    ("laboratory list", "labs"), ("practical subjects", "labs"),
    ("workshops in 1st year", "labs"), ("skill courses", "labs"),
    ("labs in 3-1", "labs"), ("labs in 2-2", "labs"),
    ("4th year labs", "labs"), ("iv year labs", "labs"),
    ("nbkrist", "college"), ("naac grade", "college"),
    ("nba accreditation", "college"), ("college established", "college"),
    ("programs offered", "college"), ("campus facilities", "college"),
    ("placement companies", "college"), ("major recruiters", "college"),
    ("career opportunities", "college"), ("recommended skills", "college"),
    ("department overview", "college"), ("ai&ds overview", "college"),
    ("college events", "college"), ("student chapters", "college"),
    ("what clubs", "college"), ("techvyuha", "college"),
    ("inspiron", "college"), ("college labs", "college"),
    ("intake", "college"), ("college vision", "college"),
    ("affiliation", "college"), ("autonomous college", "college"),
    # student
    ("student info", "student"), ("student details", "student"),
    ("roll number", "student"), ("roll no", "student"),
    ("student roll", "student"), ("find student", "student"),
    ("search student", "student"), ("student record", "student"),
    ("show student", "student"), ("student list", "student"),
    ("all students", "student"), ("student cgpa", "student"),
    ("topper", "student"), ("top students", "student"),
    ("section a students", "student"), ("section b students", "student"),
    ("ai&ds students", "student"), ("it branch students", "student"),
    ("student analytics", "student"), ("student stats", "student"),
    ("student report", "student"), ("admission number", "student"),
    # curriculum
    ("curriculum", "curriculum"), ("syllabus", "curriculum"),
    ("what subjects", "curriculum"), ("subjects in 1st year", "curriculum"),
    ("subjects in 2nd year", "curriculum"), ("subjects in 3rd year", "curriculum"),
    ("courses offered", "curriculum"), ("what do we study", "curriculum"),
    ("1st year subjects", "curriculum"), ("2nd year subjects", "curriculum"),
    ("3rd year subjects", "curriculum"), ("4th year subjects", "curriculum"),
    ("first year syllabus", "curriculum"), ("second year syllabus", "curriculum"),
    ("professional electives", "curriculum"), ("skill courses", "curriculum"),
    ("lab subjects", "curriculum"), ("elective subjects", "curriculum"),
    ("semester 1 subjects", "curriculum"), ("semester 2 subjects", "curriculum"),
    # circulars
    ("circular", "circulars"), ("notice", "circulars"),
    ("announcement", "circulars"), ("show circulars", "circulars"),
    ("latest notice", "circulars"), ("recent circular", "circulars"),
    ("fee notice", "circulars"), ("class commencement notice", "circulars"),
    ("tuition fee circular", "circulars"), ("early bird offer", "circulars"),
    ("btech classes start", "circulars"), ("when do classes start", "circulars"),
    ("fee payment deadline", "circulars"), ("enrolment notice", "circulars"),
    ("show announcements", "circulars"), ("any new notice", "circulars"),
    ("college notice", "circulars"), ("notice board", "circulars"),
    ("b-10/2026/01", "circulars"), ("iii iv btech notice", "circulars"),
]

def train_ml_classifier():
    """Train TF-IDF + Multinomial Naive Bayes intent classifier."""
    global ml_classifier, ml_vectorizer
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.preprocessing import LabelEncoder
        import pickle

        texts  = [t for t, _ in _TRAIN_DATA]
        labels = [l for _, l in _TRAIN_DATA]

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000,
                                      sublinear_tf=True)),
            ("clf",   MultinomialNB(alpha=0.3)),
        ])
        pipeline.fit(texts, labels)
        ml_classifier = pipeline
        print(f"✓ ML classifier trained  (TF-IDF + Naive Bayes, {len(texts)} samples, "
              f"{len(set(labels))} classes)")
        return True
    except Exception as e:
        print(f"⚠ ML classifier failed: {e}")
        return False


def ml_predict_intent(query: str) -> Tuple[str, float]:
    """Return (intent_label, confidence) from the trained ML classifier."""
    if ml_classifier is None:
        return "general", 0.0
    proba = ml_classifier.predict_proba([query.lower()])[0]
    classes = ml_classifier.classes_
    idx = int(np.argmax(proba))
    return classes[idx], float(proba[idx])


# ─────────────────────────────────────────────────────────────────────────────
# NLP — spaCy query analysis
# ─────────────────────────────────────────────────────────────────────────────
class QueryAnalysis:
    def __init__(self):
        self.original: str = ""
        self.lemmatized: str = ""
        self.expanded: str = ""
        self.tokens: List[str] = []
        self.entities: List[Tuple[str, str]] = []
        self.person_names: List[str] = []
        self.intent_signals: List[str] = []
        self.question_type: str = "unknown"


def analyse_query(query: str) -> QueryAnalysis:
    qa = QueryAnalysis()
    qa.original = query
    if nlp is None:
        qa.lemmatized = query.lower()
        qa.expanded   = query.lower()
        qa.tokens     = query.lower().split()
        return qa

    doc = nlp(query)
    q_lower = query.lower()
    if q_lower.startswith(("who", "whose")):       qa.question_type = "who"
    elif q_lower.startswith(("what", "which")):    qa.question_type = "what"
    elif q_lower.startswith(("when",)):            qa.question_type = "when"
    elif q_lower.startswith(("how many","how much","list","show all")): qa.question_type = "list"
    elif q_lower.startswith(("how",)):             qa.question_type = "how"
    elif q_lower.startswith(("where",)):           qa.question_type = "where"
    elif q_lower.startswith(("show","display","give")): qa.question_type = "show"

    keep_pos = {"NOUN", "PROPN", "VERB", "ADJ"}
    meaningful = []
    for token in doc:
        if (not token.is_stop and not token.is_punct and not token.is_space
                and token.pos_ in keep_pos and len(token.lemma_) > 1):
            meaningful.append(token.lemma_.lower())

    qa.tokens     = meaningful
    qa.lemmatized = " ".join(meaningful)

    for ent in doc.ents:
        qa.entities.append((ent.text, ent.label_))
        if ent.label_ == "PERSON":
            qa.person_names.append(ent.text.lower())

    qa.intent_signals = [t.lemma_.lower() for t in doc
                         if t.pos_ in {"NOUN", "PROPN"} and not t.is_stop]
    extra = " ".join(qa.tokens + [e[0] for e in qa.entities])
    qa.expanded = f"{query} {extra}".strip()
    return qa


def initialize_nlp() -> bool:
    global nlp
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy NLP model loaded  (en_core_web_sm)")
        return True
    except Exception as e:
        print(f"⚠ spaCy not available: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection — ML first, NLP rules as fallback
# ─────────────────────────────────────────────────────────────────────────────
def detect_intent(query: str, qa: QueryAnalysis) -> str:
    """
    Two-stage intent detection:
      Stage 0 — Static match greetings, farewells, help (short-circuit)
      Stage 1 — Supervised ML (TF-IDF + Naive Bayes) with confidence ≥ 0.45
      Stage 2 — spaCy lemma/entity rule fallback
    """
    q = query.lower().strip().strip("?!.,")
    
    # Common static matches
    greetings = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "hi there", "howdy", "greetings"}
    farewells = {"bye", "goodbye", "thanks", "thank you", "thank you so much", "ok thanks", "that's all", "see you", "appreciate it"}
    helps = {"help", "what can you do", "capabilities", "menu", "what can i ask"}
    
    if q in greetings or any(q.startswith(w) for w in ["hello ", "hi ", "hey "]):
        return "greeting"
    if q in farewells or any(q.startswith(w) for w in ["bye ", "thank you ", "thanks "]):
        return "farewell"
    if q in helps:
        return "help"

    ml_intent, ml_conf = ml_predict_intent(query)
    if ml_conf >= 0.45:
        return ml_intent

    # Rule-based fallback using NLP signals
    circular_signals = {"circular","notice","announcement","fee","enrolment",
                        "commencement","early bird","btech","b.tech"}
    timetable_lemmas = {"timetable","schedule","class","period","timing","slot","lecture"}
    faculty_lemmas   = {"faculty","professor","hod","head","teacher","lecturer",
                        "instructor","staff","doctor","dr","mr","mrs","ms","prof"}
    service_lemmas   = {"attendance","journal","portal","intranet","assessment",
                        "exam","hostel","admission","placement","library",
                        "result","mark","grade"}

    signals = set(qa.intent_signals + qa.tokens)
    
    # Rule fallback for greeting/farewell/help
    if q in greetings or any(w in q for w in ["hello", "hi there", "good morning"]):
        return "greeting"
    if q in farewells or any(w in q for w in ["goodbye", "thank you"]):
        return "farewell"
    if q in helps or "help" in q or "capabilities" in q:
        return "help"
        
    if (signals & circular_signals
            or any(w in q for w in ["circular","notice","announcement","fee notice",
                                    "early bird","class start","commencement","enrolment"])):
        return "circulars"
    if signals & timetable_lemmas or any(w in q for w in ["timetable","schedule","time table"]):
        return "timetable"
    if (signals & faculty_lemmas or qa.question_type == "who"
            or qa.person_names
            or any(w in q for w in ["who is","who teaches","hod","head of"])):
        return "faculty"
    if signals & service_lemmas:
        return "services"
    # student signals
    student_signals = {"student","roll","rollno","topper","cgpa","admission"}
    if signals & student_signals or any(w in q for w in ["roll no","roll number","student info","student list"]):
        return "student"
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# Faculty data
# ─────────────────────────────────────────────────────────────────────────────
_FACULTY_DATA: List[Dict] = []
_CIRCULARS: List[Dict] = []

def load_faculty_data():
    global _FACULTY_DATA
    if os.path.exists("aids_faculty_data.json"):
        with open("aids_faculty_data.json", "r", encoding="utf-8") as f:
            _FACULTY_DATA = json.load(f)

    # Merge specializations from aids_faculty_kb.json
    for f in _FACULTY_DATA:
        f["specialization"] = "—"

    if os.path.exists("aids_faculty_kb.json"):
        try:
            with open("aids_faculty_kb.json", "r", encoding="utf-8") as f_kb:
                kb_data = json.load(f_kb)
            for kb_key, kb_val in kb_data.items():
                if kb_key in ["aids faculty", "ai ds department", "hod aids", "machine learning faculty", "ai faculty"]:
                    continue
                kb_norm = _normalise_name(kb_key)
                kb_tokens = {t for t in kb_norm.split() if len(t) >= 4}
                if not kb_tokens:
                    continue

                matched_fac = None
                for f in _FACULTY_DATA:
                    fn_norm = _normalise_name(f.get("name", ""))
                    fn_tokens = {t for t in fn_norm.split() if len(t) >= 4}
                    if kb_norm == fn_norm or (kb_tokens & fn_tokens):
                        matched_fac = f
                        break

                if matched_fac:
                    match = re.search(r'Specialization:\s*([^.]+)', kb_val)
                    if match:
                        matched_fac["specialization"] = match.group(1).strip()
        except Exception as e:
            print(f"Error merging specialization: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Curriculum data
# ─────────────────────────────────────────────────────────────────────────────
_CURRICULUM: Dict = {}

def load_curriculum():
    global _CURRICULUM
    if os.path.exists("aids_curriculum.json"):
        with open("aids_curriculum.json", "r", encoding="utf-8") as f:
            _CURRICULUM = json.load(f)
        print(f"✓ Curriculum loaded: {len(_CURRICULUM)} semesters")

# ─────────────────────────────────────────────────────────────────────────────
# Bus Fee Module
# ─────────────────────────────────────────────────────────────────────────────
_BUS_FEES: Dict = {}
_BUS_YEAR: str = "2026-27"

def load_bus_fees():
    global _BUS_FEES, _BUS_YEAR
    if os.path.exists("nbkr_bus_fees.json"):
        with open("nbkr_bus_fees.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        _BUS_FEES = data.get("bus_fees", {})
        _BUS_YEAR = data.get("academic_year", "2026-27")
        total = sum(len(v) for v in _BUS_FEES.values())
        print(f"✓ Bus fees loaded: {len(_BUS_FEES)} routes, {total} stops")

def _search_bus_fee(location: str) -> List[Dict]:
    """Search all stops case-insensitively. Returns list of matches with route info."""
    loc = location.strip().lower()
    results = []
    for route, stops in _BUS_FEES.items():
        for entry in stops:
            stop_parts = [s.strip().lower() for s in entry["stop"].replace("(","").replace(")","").split(",")]
            if any(loc == part or loc in part or part in loc for part in stop_parts):
                results.append({
                    "location": entry["stop"],
                    "fee": entry["fee"],
                    "route": route.replace("_", " ").title(),
                    "year": _BUS_YEAR
                })
    return results

def build_bus_fee_card(matches: List[Dict], query: str) -> str:
    """Build a structured HTML card for bus fee results."""
    if not matches:
        return f"""
<div style="font-family:'Segoe UI',sans-serif;max-width:500px;margin:8px 0">
  <div style="background:linear-gradient(135deg,#e65100,#bf360c);color:#fff;
              padding:12px 18px;border-radius:10px 10px 0 0;font-size:14px;font-weight:700">
    🚌 Bus Fee Information
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;
              border-radius:0 0 10px 10px;padding:14px 18px">
    <p style="font-size:13px;color:#c62828">Sorry, bus fee information for
    <b>'{query}'</b> is not available in the current dataset.</p>
  </div>
</div>"""

    # Build rows for each match
    rows = ""
    for i, m in enumerate(matches):
        bg = "background:#f8f9ff;" if i % 2 == 0 else ""
        rows += f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:10px 18px;border-bottom:1px solid #eee;{bg}">
  <div>
    <div style="font-size:13px;font-weight:600;color:#1a1a2e">📍 {m['location']}</div>
    <div style="font-size:11px;color:#888;margin-top:2px">Route: {m['route']}</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:16px;font-weight:700;color:#1b5e20">₹{m['fee']:,}</div>
    <div style="font-size:10px;color:#888">Academic Year: {m['year']}</div>
  </div>
</div>"""

    title = f"Bus Fee — {matches[0]['location']}" if len(matches) == 1 else \
            f"Bus Fee Results for '{query}' ({len(matches)} match{'es' if len(matches)>1 else ''})"

    return f"""
<div style="font-family:'Segoe UI',sans-serif;max-width:540px;margin:8px 0">
  <div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);color:#fff;
              padding:12px 18px;border-radius:10px 10px 0 0;font-size:14px;font-weight:700">
    🚌 {title}
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;
              border-radius:0 0 10px 10px;overflow:hidden">
    {rows}
    <div style="padding:9px 18px;background:#f0f4ff;font-size:11px;color:#555">
      📌 N.B.K.R. Institute of Science &amp; Technology, Vidyanagar
    </div>
  </div>
</div>"""

def build_all_routes_card() -> str:
    """Show all routes and their stops with fees."""
    if not _BUS_FEES:
        return '<p style="font-family:Segoe UI,sans-serif;color:#c62828">Bus fee data not available.</p>'

    th = 'style="border:1px solid #c5cae9;padding:8px 12px;background:#e8eaf6;font-size:11px;font-weight:700;color:#1a237e;text-align:left"'
    sections = ""
    for route, stops in _BUS_FEES.items():
        route_label = route.replace("_", " ").title()
        rows = ""
        for i, s in enumerate(stops):
            bg = "background:#f8f9ff;" if i % 2 == 0 else ""
            rows += (f'<tr style="{bg}">'
                     f'<td style="border:1px solid #e8eaf6;padding:7px 12px;font-size:12.5px;color:#333">{s["stop"]}</td>'
                     f'<td style="border:1px solid #e8eaf6;padding:7px 12px;font-size:13px;font-weight:700;color:#1b5e20;text-align:right">₹{s["fee"]:,}</td>'
                     f'</tr>')
        sections += f"""
<div style="margin-bottom:14px">
  <div style="padding:8px 14px;background:linear-gradient(135deg,#1b5e20,#2e7d32);
              color:#fff;font-size:12px;font-weight:700;border-radius:6px;margin-bottom:4px">
    🚌 {route_label}
  </div>
  <div style="overflow-x:auto;border:1px solid #c5cae9;border-radius:0 0 6px 6px">
    <table style="width:100%;border-collapse:collapse">
      <thead><tr><th {th}>Boarding Point</th><th {th} style="text-align:right">Fee (Annual)</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""

    return f"""
<div style="font-family:'Segoe UI',sans-serif;margin:8px 0">
  <div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);color:#fff;
              padding:11px 16px;border-radius:8px 8px 0 0;display:flex;
              justify-content:space-between;align-items:center">
    <b>🚌 NBKR Bus Fee Structure — {_BUS_YEAR}</b>
    <span style="font-size:11px;opacity:.8">{len(_BUS_FEES)} Routes</span>
  </div>
  <div style="background:#fff;border:1px solid #c5cae9;border-top:none;
              border-radius:0 0 8px 8px;padding:14px">
    {sections}
  </div>
</div>"""

def handle_bus_fee_query(query: str) -> Optional[str]:
    """Return bus fee HTML if query is bus-fee related, else None."""
    ml = query.lower().strip()
    bus_kw = {"bus fee","bus fare","bus charge","transportation fee","transport fee",
              "bus fees","bus cost","boarding fee","bus route","bus from",
              "fee for bus","how much bus","bus amount"}
    if not any(kw in ml for kw in bus_kw) and not re.search(r'\bbus\b', ml):
        return None
    if not _BUS_FEES:
        return None

    # Show all routes
    if any(w in ml for w in ["all","routes","list","show all","all routes","all fees"]):
        return build_all_routes_card()

    # Extract location from query — strip bus-related words
    location = re.sub(
        r'\b(bus|fee|fees|fare|charge|charges|transport|transportation|'
        r'from|for|what|is|the|how|much|cost|amount|boarding|route|routes)\b',
        '', ml, flags=re.IGNORECASE
    ).strip().strip('?').strip()

    if not location:
        return build_all_routes_card()

    matches = _search_bus_fee(location)
    return build_bus_fee_card(matches, location)

def build_curriculum_card(sem_key: str, data: Dict) -> str:
    """Build a structured HTML card — subjects and labs shown in separate clear sections."""
    grad_map = {
        "I Year":   "linear-gradient(135deg,#1a237e,#283593)",
        "II Year":  "linear-gradient(135deg,#1b5e20,#2e7d32)",
        "III Year": "linear-gradient(135deg,#e65100,#bf360c)",
        "IV Year":  "linear-gradient(135deg,#4a148c,#6a1b9a)",
    }
    grad = next((v for k, v in grad_map.items() if k in sem_key),
                "linear-gradient(135deg,#667eea,#764ba2)")

    th  = ('style="border:1px solid #c5cae9;padding:8px 12px;background:#e8eaf6;'
           'font-size:11px;font-weight:700;color:#1a237e;text-align:left;'
           'text-transform:uppercase;letter-spacing:.04em"')
    td_no = 'style="border:1px solid #e8eaf6;padding:7px 12px;font-size:12px;color:#1a237e;font-weight:700;text-align:center;width:36px"'
    td_c  = 'style="border:1px solid #e8eaf6;padding:8px 12px;font-size:12.5px;color:#1a1a2e;font-weight:600"'
    td_l  = 'style="border:1px solid #e8eaf6;padding:8px 12px;font-size:12px;color:#2e7d32;font-weight:500"'
    td_nl = 'style="border:1px solid #e8eaf6;padding:8px 12px;font-size:12px;color:#aaa"'

    sections_html = ""

    for section, items in data.items():
        if not items:
            continue

        if section in ("Electives", "Skill Courses", "Non-Electives"):
            # Split into subjects (no lab) and subjects with labs
            subj_rows = ""
            lab_rows  = ""
            for i, item in enumerate(items):
                course = item.get("Course","—")
                lab    = item.get("Lab","")
                act    = ", ".join(item.get("Activity",[]))
                # strip course code from name for cleaner display
                code_m = re.search(r'\(([^)]+)\)', course)
                code   = code_m.group(1) if code_m else ""
                name   = re.sub(r'\s*\([^)]+\)', '', course).strip()
                bg_e   = "background:#f8f9ff;" if i % 2 == 0 else ""
                bg_o   = "background:#f1f8e9;" if i % 2 == 0 else "background:#f9fbe7;"

                if act:
                    subj_rows += (f'<tr style="{bg_e}">'
                                  f'<td {td_no}>{i+1}</td>'
                                  f'<td {td_c}>{name}</td>'
                                  f'<td style="border:1px solid #e8eaf6;padding:8px 12px;'
                                  f'font-size:11px;color:#888">{code}</td>'
                                  f'<td {td_l}>{act}</td></tr>')
                else:
                    has_lab = lab and lab != "None"
                    subj_rows += (f'<tr style="{bg_e}">'
                                  f'<td {td_no}>{i+1}</td>'
                                  f'<td {td_c}>{name}</td>'
                                  f'<td style="border:1px solid #e8eaf6;padding:8px 12px;'
                                  f'font-size:11px;color:#888">{code}</td>'
                                  f'<td {td_l if has_lab else td_nl}>'
                                  f'{"✓" if has_lab else "—"}</td></tr>')
                    if has_lab:
                        j = len([x for x in items[:i+1] if x.get("Lab") and x.get("Lab") != "None"])
                        bg_l = "background:#f1f8e9;" if j % 2 == 0 else ""
                        lab_rows += (f'<tr style="{bg_l}">'
                                     f'<td {td_no}>{j}</td>'
                                     f'<td {td_c}>{name}</td>'
                                     f'<td {td_l}>🔬 {lab}</td></tr>')

            # Subjects table
            col3 = "Activity" if section == "Non-Electives" else "Has Lab"
            sections_html += f"""
<div style="border-top:1px solid #eee">
  <div style="padding:8px 14px;background:#e8eaf6;font-size:11px;font-weight:700;
              color:#1a237e;text-transform:uppercase;letter-spacing:.04em">
    📚 {section}
  </div>
  <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;min-width:360px">
      <thead><tr>
        <th {th} style="width:36px;text-align:center">#</th>
        <th {th}>Subject / Course</th>
        <th {th} style="width:110px">Code</th>
        <th {th} style="width:80px">{col3}</th>
      </tr></thead>
      <tbody>{subj_rows}</tbody>
    </table>
  </div>
</div>"""

            # Labs table — only if any labs exist
            if lab_rows:
                sections_html += f"""
<div style="border-top:1px solid #eee">
  <div style="padding:8px 14px;background:#e8f5e9;font-size:11px;font-weight:700;
              color:#2e7d32;text-transform:uppercase;letter-spacing:.04em">
    🔬 Labs / Practicals
  </div>
  <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;min-width:360px">
      <thead><tr>
        <th {th} style="width:36px;text-align:center">#</th>
        <th {th}>Subject</th>
        <th {th}>Lab Name</th>
      </tr></thead>
      <tbody>{lab_rows}</tbody>
    </table>
  </div>
</div>"""

        else:
            # Professional Electives / Management options
            opts = "".join(
                f'<div style="padding:8px 14px;border-bottom:1px solid #f0f0f0;'
                f'font-size:12.5px;color:#333">🔹 {item.get("Option","")}</div>'
                for item in items if item.get("Option")
            )
            sections_html += f"""
<div style="border-top:1px solid #eee">
  <div style="padding:8px 14px;background:#fff8e1;font-size:11px;font-weight:700;
              color:#e65100;text-transform:uppercase;letter-spacing:.04em">
    🎯 {section}
  </div>
  {opts}
</div>"""

    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif;max-width:100%;box-sizing:border-box">
  <div style="background:{grad};color:#fff;padding:13px 18px;border-radius:10px 10px 0 0">
    <div style="font-size:15px;font-weight:700">📖 {sem_key}</div>
    <div style="font-size:11px;opacity:.8;margin-top:3px">AI &amp; DS Department — NBKR Institute</div>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;
              border-radius:0 0 10px 10px;overflow:hidden">
    {sections_html}
  </div>
</div>"""

def build_all_curriculum_table() -> str:
    """Summary table of all semesters."""
    if not _CURRICULUM:
        return '<p style="font-family:Segoe UI,sans-serif;color:#c62828">Curriculum data not available.</p>'

    rows = ""
    for i, (sem, data) in enumerate(_CURRICULUM.items()):
        total = sum(len(v) for v in data.values())
        electives = data.get("Electives", [])
        course_names = ", ".join(
            e.get("Course","").split("(")[0].strip() for e in electives[:3]
        )
        if len(electives) > 3:
            course_names += f" +{len(electives)-3} more"
        bg = "background:#f8f9ff;" if i % 2 == 0 else ""
        rows += (f'<tr style="{bg}">'
                 f'<td style="border:1px solid #e8eaf6;padding:9px 14px;font-size:13px;font-weight:700;color:#1a237e">{sem}</td>'
                 f'<td style="border:1px solid #e8eaf6;padding:9px 14px;font-size:12px;color:#333">{course_names or "—"}</td>'
                 f'<td style="border:1px solid #e8eaf6;padding:9px 14px;font-size:12px;color:#555;text-align:center">{total}</td>'
                 f'</tr>')

    th = 'style="border:1px solid #c5cae9;padding:9px 14px;background:#e8eaf6;font-size:12px;font-weight:700;color:#1a237e;text-align:left"'
    return f"""
<div style="font-family:'Segoe UI',sans-serif;margin:8px 0">
  <div style="background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
              padding:11px 16px;border-radius:8px 8px 0 0;display:flex;
              justify-content:space-between;align-items:center">
    <b>📖 AI &amp; DS Curriculum — All Semesters</b>
    <span style="font-size:11px;opacity:.8">{len(_CURRICULUM)} Semesters</span>
  </div>
  <div style="overflow-x:auto;border:1px solid #c5cae9;border-top:none;border-radius:0 0 8px 8px">
    <table style="width:100%;border-collapse:collapse;min-width:500px">
      <thead><tr>
        <th {th}>Semester</th>
        <th {th}>Key Courses</th>
        <th {th} style="text-align:center">Total Items</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <p style="font-size:11px;color:#888;margin-top:5px;font-family:Segoe UI,sans-serif">
    Ask "1st year curriculum", "3rd year sem 2 subjects" for detailed view.
  </p>
</div>"""

def handle_curriculum_query(query: str) -> Optional[str]:
    """Return curriculum HTML if query is about subjects/courses, else None."""
    ml = query.lower().strip()

    curric_kw = {"curriculum","syllabus","subject","course","sem","semester",
                 "what subjects","what courses","which subjects","1st year subjects",
                 "2nd year subjects","3rd year subjects","4th year subjects",
                 "first year subjects","second year subjects","third year subjects",
                 "fourth year subjects","elective","professional elective",
                 "skill course","lab subjects","what do we study","what is taught"}
    if not any(kw in ml for kw in curric_kw):
        return None

    if not _CURRICULUM:
        return '<p style="font-family:Segoe UI,sans-serif;color:#c62828">Curriculum data not loaded.</p>'

    # All curriculum overview
    if any(w in ml for w in ["all","overview","full","complete","entire","list all"]):
        return build_all_curriculum_table()

    # ── Precise year detection ─────────────────────────────────────────────
    # Only match the exact year word — avoid "1" matching "11" etc.
    year = None
    if re.search(r'\b(1st|first|one|i)\b.*\b(year|yr)\b|\b(year|yr)\b.*\b(1st|first|one|i)\b', ml):
        year = 1
    elif re.search(r'\b(2nd|second|two|ii)\b.*\b(year|yr)\b|\b(year|yr)\b.*\b(2nd|second|two|ii)\b', ml):
        year = 2
    elif re.search(r'\b(3rd|third|three|iii)\b.*\b(year|yr)\b|\b(year|yr)\b.*\b(3rd|third|three|iii)\b', ml):
        year = 3
    elif re.search(r'\b(4th|fourth|four|iv)\b.*\b(year|yr)\b|\b(year|yr)\b.*\b(4th|fourth|four|iv)\b', ml):
        year = 4
    # Also catch plain "1st year", "2nd year" etc. without "year" mentioned twice
    if year is None:
        if re.search(r'\b1st\b|\bfirst year\b', ml): year = 1
        elif re.search(r'\b2nd\b|\bsecond year\b', ml): year = 2
        elif re.search(r'\b3rd\b|\bthird year\b', ml): year = 3
        elif re.search(r'\b4th\b|\bfourth year\b', ml): year = 4

    # ── Precise semester detection ─────────────────────────────────────────
    sem = None
    if re.search(r'\b(sem\s*1|sem\s*i|semester\s*1|semester\s*i|first\s*sem)\b', ml):
        sem = 1
    elif re.search(r'\b(sem\s*2|sem\s*ii|semester\s*2|semester\s*ii|second\s*sem)\b', ml):
        sem = 2

    # ── Key matching — exact year prefix check ────────────────────────────
    year_prefix = {1: "I Year", 2: "II Year", 3: "III Year", 4: "IV Year"}
    sem_suffix  = {1: "Sem I", 2: "Sem II"}

    matched_keys = []
    for key in _CURRICULUM:
        # Year check — must match exact roman numeral prefix
        if year is not None:
            prefix = year_prefix[year]
            if not key.startswith(prefix):
                continue  # Skip keys that don't start with the right year
        # Semester check
        if sem is not None:
            suffix = sem_suffix[sem]
            if suffix not in key:
                continue
        matched_keys.append(key)

    if len(matched_keys) == 1:
        key = matched_keys[0]
        return build_curriculum_card(key, _CURRICULUM[key])

    if len(matched_keys) > 1:
        return "".join(build_curriculum_card(k, _CURRICULUM[k]) for k in matched_keys)

    # Nothing matched — show overview
    return build_all_curriculum_table()

def load_circulars():
    global _CIRCULARS
    if os.path.exists("nbkr_circulars.json"):
        with open("nbkr_circulars.json", "r", encoding="utf-8") as f:
            _CIRCULARS = json.load(f)
        print(f"✓ Circulars loaded: {len(_CIRCULARS)} notices")

_DESIG_ORDER = {"head of the department":0,"professor":1,
                "associate professor":2,"assistant professor":3}
def _desig_rank(d): return _DESIG_ORDER.get(d.lower().strip(), 9)

_DESIG_COLOR = {
    "Head of the Department": ("#1a237e","#e8eaf6"),
    "Professor":              ("#1b5e20","#e8f5e9"),
    "Associate Professor":    ("#e65100","#fff3e0"),
    "Assistant Professor":    ("#4a148c","#f3e5f5"),
}
def _badge(designation):
    fg, bg = _DESIG_COLOR.get(designation, ("#333","#f5f5f5"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:10px;font-size:11px;font-weight:600">{designation}</span>')

_FTH = 'style="border:1px solid #ddd;padding:10px 14px;text-align:left;font-size:13px;background:#f0f4ff;font-weight:700;color:#1a1a2e"'
_FTD = 'style="border:1px solid #ddd;padding:9px 13px;font-size:13px;vertical-align:top;color:#1a1a2e"'
_FTD_C = 'style="border:1px solid #ddd;padding:9px 13px;font-size:13px;text-align:center;vertical-align:middle;color:#1a1a2e"'


def build_faculty_list_table(faculty_list=None):
    data = sorted(faculty_list if faculty_list else _FACULTY_DATA,
                  key=lambda x: _desig_rank(x.get("designation","")))
    rows = ""
    for i, f in enumerate(data, 1):
        bg = "#fafafa" if i % 2 == 0 else "#fff"
        phone = f.get("phone", "—") or "—"
        email = f.get("email", "—") or "—"
        # show only first email for table brevity
        first_email = email.split(",")[0].strip()
        rows += (f'<tr style="background:{bg}">'
                 f'<td {_FTD_C}>{i}</td>'
                 f'<td {_FTD}><b>{f.get("name","—")}</b></td>'
                 f'<td {_FTD}>{_badge(f.get("designation","—"))}</td>'
                 f'<td {_FTD} style="color:#555">{f.get("specialization","—")}</td>'
                 f'<td {_FTD}>{phone}</td>'
                 f'<td {_FTD}><a href="mailto:{first_email}" style="color:#1a237e;font-size:12px">{first_email}</a></td>'
                 f'</tr>')
    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif">
  <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;
              padding:10px 16px;border-radius:8px 8px 0 0;display:flex;
              justify-content:space-between;align-items:center">
    <b>👥 AI &amp; DS Department — Faculty List</b>
    <span style="font-size:11px;opacity:.85">{len(data)} Members</span>
  </div>
  <div style="overflow-x:auto;border:1px solid #ddd;border-top:none;border-radius:0 0 8px 8px">
    <table style="width:100%;border-collapse:collapse;min-width:700px">
      <thead><tr>
        <th {_FTH} style="text-align:center;width:45px">S.No</th>
        <th {_FTH}>Name</th>
        <th {_FTH}>Designation</th>
        <th {_FTH}>Specialization</th>
        <th {_FTH}>Phone</th>
        <th {_FTH}>Email</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""


def build_faculty_timetable_html_embed(timetable) -> str:
    """Render faculty weekly timetable.
    Supports:
      - Slot format: {"Mon": {"9:00-10:00": "SUBJECT (CLASS)", ...}, ...}
      - List format: {"Monday": [{time, subject, class, type}, ...], ...}
      - Old flat list: [{day, time, subject, class}, ...]
    """
    if not timetable:
        return ""

    LUNCH_VALS = {"L","U","N","C","H","BREAK","LUNCH",""}
    SLOTS = ["9:00-10:00","10:00-11:00","11:00-12:00","12:00-1:00",
             "1:00-2:00","2:00-3:00","3:00-4:00","4:00-5:00"]
    DAY_ORDER = ["Mon","Tue","Wed","Thu","Fri","Sat"]

    # ── Detect slot format: {"Mon": {"9:00-10:00": "...", ...}, ...} ──────────
    first_val = next(iter(timetable.values()), None)
    is_slot_format = isinstance(first_val, dict)

    if isinstance(timetable, dict) and is_slot_format:
        # Build a grid table: rows = days, cols = time slots
        # Detect which days/slots have content
        active_days = [d for d in DAY_ORDER if d in timetable and
                       any(v for v in timetable[d].values() if v not in LUNCH_VALS)]
        if not active_days:
            return ""

        th_style = ('style="border:1px solid #c5cae9;padding:7px 10px;background:#1a237e;'
                    'color:#fff;font-size:11px;font-weight:700;text-align:center;white-space:nowrap"')
        th_day   = ('style="border:1px solid #c5cae9;padding:7px 10px;background:#283593;'
                    'color:#fff;font-size:11px;font-weight:700;text-align:center"')

        header = f'<tr><th {th_day}>Day</th>' + \
                 ''.join(f'<th {th_style}>{s}</th>' for s in SLOTS) + '</tr>'

        rows_html = ""
        for day in active_days:
            day_slots = timetable.get(day, {})
            row = f'<td style="border:1px solid #e0e0e0;padding:7px 10px;font-size:12px;' \
                  f'font-weight:700;color:#880e4f;background:#fce4ec;text-align:center;' \
                  f'white-space:nowrap">{day}</td>'
            for slot in SLOTS:
                val = day_slots.get(slot, "")
                if val in LUNCH_VALS:
                    if val in ("L","U","N","C","H"):
                        row += (f'<td style="border:1px solid #e0e0e0;padding:4px;'
                                f'background:#f5f5f5;text-align:center;font-size:13px;'
                                f'font-weight:900;color:#999">{val}</td>')
                    elif val == "BREAK":
                        row += (f'<td style="border:1px solid #e0e0e0;padding:4px;'
                                f'background:#f5f5f5;text-align:center;font-size:10px;'
                                f'color:#999">BREAK</td>')
                    else:
                        row += f'<td style="border:1px solid #e0e0e0;background:#fafafa"></td>'
                else:
                    # Parse "SUBJECT (CLASS)" or "SUBJECT CLASS"
                    m = re.match(r'^(.+?)\s*\((.+)\)$', val.strip())
                    subj = m.group(1).strip() if m else val.strip()
                    cls  = m.group(2).strip() if m else ""
                    is_lab = any(w in subj.upper() for w in
                                 ("LAB","WORKSHOP","PROJECT","TUTORIAL"))
                    bg  = "#fffde7" if is_lab else "#f0f4ff"
                    col = "#5a4000" if is_lab else "#1a237e"
                    cls_html = (f'<div style="font-size:9.5px;color:{col};opacity:.75;'
                                f'margin-top:2px">({cls})</div>') if cls else ""
                    row += (f'<td style="border:1px solid #e0e0e0;padding:5px 7px;'
                            f'background:{bg};text-align:center;vertical-align:middle">'
                            f'<div style="font-size:11px;font-weight:700;color:{col}">{subj}</div>'
                            f'{cls_html}</td>')
            rows_html += f'<tr>{row}</tr>'

        return f"""
<div style="margin-top:0;border-top:1px solid #eee">
  <div style="padding:9px 16px;background:#f8f9ff;font-size:11px;font-weight:700;
              color:#888;text-transform:uppercase;letter-spacing:.04em">
    📅 Weekly Timetable
  </div>
  <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;min-width:700px">
      <thead>{header}</thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>"""

    # ── List-of-slots format: {"Monday": [{time, subject, class, type}...]} ──
    if isinstance(timetable, dict) and not is_slot_format:
        day_order_long = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
        rows_html = ""
        for day in day_order_long:
            slots = timetable.get(day, [])
            classes = [s for s in slots if s.get("type") not in ("break",)]
            if not classes:
                continue
            for i, s in enumerate(classes):
                subj    = s.get("subject","—")
                time    = s.get("time","—")
                cls     = s.get("class","—") or "—"
                is_lab  = any(w in subj.upper() for w in ("LAB","WORKSHOP","PROJECT","TUTORIAL"))
                subj_bg  = "#fffde7" if is_lab else "#f0f4ff"
                subj_col = "#5a4000" if is_lab else "#1a237e"
                day_cell = (f'<td rowspan="{len(classes)}" style="border:1px solid #e0e0e0;'
                            f'padding:8px 12px;font-size:12px;font-weight:700;color:#880e4f;'
                            f'background:#fce4ec;text-align:center;vertical-align:middle;'
                            f'white-space:nowrap">{day[:3]}</td>') if i == 0 else ""
                rows_html += (f'<tr>{day_cell}'
                              f'<td style="border:1px solid #e0e0e0;padding:7px 12px;font-size:12px;'
                              f'white-space:nowrap;color:#333">{time}</td>'
                              f'<td style="border:1px solid #e0e0e0;padding:7px 12px;font-size:12px;'
                              f'font-weight:600;background:{subj_bg};color:{subj_col}">{subj}</td>'
                              f'<td style="border:1px solid #e0e0e0;padding:7px 12px;font-size:11px;'
                              f'color:#555">{cls}</td>'
                              f'</tr>')
        if not rows_html:
            return ""
        th = ('style="border:1px solid #c5cae9;padding:8px 12px;background:#e8eaf6;'
              'font-size:11px;font-weight:700;color:#1a237e;text-align:left;'
              'text-transform:uppercase;letter-spacing:.04em"')
        return f"""
<div style="margin-top:0;border-top:1px solid #eee">
  <div style="padding:9px 16px;background:#f8f9ff;font-size:11px;font-weight:700;
              color:#888;text-transform:uppercase;letter-spacing:.04em">📅 Weekly Timetable</div>
  <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;min-width:380px">
      <thead><tr>
        <th {th}>Day</th><th {th}>Time</th><th {th}>Subject</th><th {th}>Class</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>"""

    return ""


def extract_subjects_handled(f: Dict) -> List[str]:
    subjects = set()
    timetable = f.get("timetable", {})
    if isinstance(timetable, dict):
        for day, slots in timetable.items():
            if isinstance(slots, dict):
                for slot, val in slots.items():
                    if not val:
                        continue
                    val_clean = val.strip()
                    if val_clean in {"L", "U", "N", "C", "H", "BREAK", "LUNCH", "l", "u", "n", "c", "h", "break", "lunch"}:
                        continue
                    match = re.match(r'^(.+?)\s*\((.+)\)$', val_clean)
                    subj = match.group(1).strip() if match else val_clean
                    if subj and subj not in ["-", "LUNCH", "BREAK"]:
                        subjects.add(subj)
    full_subjects = []
    subj_full = globals().get("SUBJECT_FULL", {})
    for s in sorted(subjects):
        full_name = subj_full.get(s, s)
        if full_name != s:
            full_subjects.append(f"{full_name} ({s})")
        else:
            full_subjects.append(s)
    return full_subjects


def build_faculty_card(f, include_timetable=False):
    """Rich faculty card showing all fields: name, designation, department,
    qualification, phone, email, date of joining, cabin/room, subjects handled,
    and optional timetable. No timetable by default."""
    name  = f.get("name", "—")
    desig = f.get("designation", "—")
    qual  = f.get("Qualification", "")
    phone = f.get("phone", "")
    email = f.get("email", "")
    doj   = f.get("date_of_joining", "")
    room  = f.get("room") or f.get("cabin") or f.get("cabin_no")

    def _row(icon, label, value, hi=False):
        bg = "background:#f8f9ff;" if hi else ""
        return (f'<div style="display:flex;align-items:flex-start;gap:10px;'
                f'padding:10px 16px;border-top:1px solid #eee;{bg}">'
                f'<span style="min-width:130px;font-size:11px;font-weight:700;'
                f'color:#888;text-transform:uppercase;letter-spacing:.04em;padding-top:1px;flex-shrink:0">'
                f'{icon} {label}</span>'
                f'<span style="font-size:13px;color:#222;line-height:1.6;word-break:break-word">{value}</span></div>')

    # Department
    dept_html = _row("🏛️", "Department", "Artificial Intelligence & Data Science")

    # Qualification pills
    qual_html = ""
    if qual:
        pills = "".join(
            f'<span style="display:inline-block;background:#ede9fe;color:#5b21b6;'
            f'padding:2px 9px;border-radius:8px;font-size:12px;font-weight:600;margin:2px 3px 2px 0">'
            f'{e.strip()}</span>'
            for e in qual.split(",") if e.strip()
        )
        qual_html = _row("🎓", "Qualification", pills, hi=True)

    # Phone — make each number a tel link
    phone_html = ""
    if phone:
        # Split by / or , to handle multi-phones
        nums = [p.strip() for p in re.split(r'[,/]', phone) if p.strip()]
        phone_html = _row("📱", "Phone",
            "".join(f'<a href="tel:{n}" style="color:#1a237e;text-decoration:none;'
                    f'margin-right:10px;display:inline-block">{n}</a>' for n in nums))

    # Email links
    email_html = ""
    if email:
        mails = [m.strip() for m in email.split(",") if m.strip()]
        email_html = _row("📧", "Email",
            "".join(f'<a href="mailto:{m}" style="color:#1a237e;text-decoration:none;'
                    f'display:block;word-break:break-all">{m}</a>' for m in mails), hi=True)

    # Date of joining
    doj_html = _row("📅", "Date of Joining", doj) if doj else ""

    # Cabin / Room
    room_html = ""
    if room:
        room_html = _row("🚪", "Cabin/Room", room)

    # Subjects Handled
    subj_list = extract_subjects_handled(f)
    subj_html = ""
    if subj_list:
        pills = "".join(
            f'<span style="display:inline-block;background:#e0f2fe;color:#0369a1;'
            f'padding:2px 9px;border-radius:8px;font-size:12px;font-weight:600;margin:2px 3px 2px 0">'
            f'{s}</span>'
            for s in subj_list
        )
        subj_html = _row("📚", "Subjects Handled", pills, hi=True)

    # Timetable
    tt_html = ""
    if include_timetable:
        tt_html = build_faculty_timetable_html_embed(f.get("timetable", {}))

    # Header gradient by designation
    grad = {
        "Head of the Department": "linear-gradient(135deg,#1a237e,#283593)",
        "Professor":               "linear-gradient(135deg,#1b5e20,#2e7d32)",
        "Associate Professor":     "linear-gradient(135deg,#e65100,#bf360c)",
        "Assistant Professor":     "linear-gradient(135deg,#4a148c,#6a1b9a)",
    }.get(desig, "linear-gradient(135deg,#667eea,#764ba2)")

    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif;max-width:100%;box-sizing:border-box">
  <div style="background:{grad};color:#fff;padding:14px 18px;border-radius:10px 10px 0 0">
    <div style="font-size:15px;font-weight:700;word-break:break-word">👤 {name}</div>
    <div style="margin-top:5px">{_badge(desig)}</div>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;
              border-radius:0 0 10px 10px;overflow:hidden">
    {dept_html}
    {qual_html}
    {phone_html}
    {email_html}
    {doj_html}
    {room_html}
    {subj_html}
    {tt_html}
  </div>
</div>
<div style="font-family:'Segoe UI',sans-serif;margin-top:10px;border-radius:10px;overflow:hidden;border:1px solid #c5cae9">
  <div style="background:linear-gradient(135deg,#2d3a6b,#4a5568);color:#fff;padding:10px 16px;font-size:13px;font-weight:700;display:flex;align-items:center;gap:8px">
    📅 Faculty Schedule Overview
  </div>
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr>
        <th style="padding:9px 14px;background:#f4f6fb;color:#2d3a6b;font-size:11px;font-weight:700;text-align:left;border:1px solid #e0e4f0">FACULTY NAME</th>
        <th style="padding:9px 14px;background:#f4f6fb;color:#2d3a6b;font-size:11px;font-weight:700;text-align:center;border:1px solid #e0e4f0">TIME TABLE VIEW</th>
        <th style="padding:9px 14px;background:#f4f6fb;color:#2d3a6b;font-size:11px;font-weight:700;text-align:center;border:1px solid #e0e4f0">LEISURE PERIODS VIEW</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px 14px;border:1px solid #e8eaf6;font-size:13px;color:#1a1a2e">
          <div style="font-weight:600">{name}</div>
          <div style="font-size:11px;color:#7c3aed;margin-top:2px">{desig}</div>
        </td>
        <td style="padding:10px 14px;border:1px solid #e8eaf6;text-align:center">
          <button onclick="quickSend('timetable of {name}')"
            onmouseover="this.style.opacity=0.85" onmouseout="this.style.opacity=1.0"
            style="background:#2d3a6b;color:#fff;border:none;padding:7px 13px;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer">
            📅 View Timetable
          </button>
        </td>
        <td style="padding:10px 14px;border:1px solid #e8eaf6;text-align:center">
          <button onclick="quickSend('leisure hours of {name}')"
            onmouseover="this.style.opacity=0.85" onmouseout="this.style.opacity=1.0"
            style="background:#7c3aed;color:#fff;border:none;padding:7px 13px;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer">
            🕐 View Leisure
          </button>
        </td>
      </tr>
    </tbody>
  </table>
</div>"""


def build_specialization_table(spec_label, faculty_list):
    if not faculty_list:
        return f'<p style="font-family:Segoe UI,sans-serif">No faculty found for <b>{spec_label}</b>.</p>'
    rows = ""
    for i, f in enumerate(faculty_list, 1):
        bg = "#fafafa" if i % 2 == 0 else "#fff"
        rows += (f'<tr style="background:{bg}"><td {_FTD_C}>{i}</td>'
                 f'<td {_FTD}><b>{f.get("name","—")}</b></td>'
                 f'<td {_FTD}>{_badge(f.get("designation","—"))}</td>'
                 f'<td {_FTD} style="color:#555">{f.get("specialization","—")}</td></tr>')
    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif">
  <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:10px 16px;border-radius:8px 8px 0 0">
    <b>🔬 Faculty — {spec_label}</b>
  </div>
  <div style="overflow-x:auto;border:1px solid #ddd;border-top:none;border-radius:0 0 8px 8px">
    <table style="width:100%;border-collapse:collapse;min-width:420px">
      <thead><tr>
        <th {_FTH} style="text-align:center;width:55px">S.No</th>
        <th {_FTH}>Name</th><th {_FTH}>Designation</th><th {_FTH}>Specialization</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Circular HTML builder
# ─────────────────────────────────────────────────────────────────────────────
def build_circular_card(c: Dict) -> str:
    """Render a single circular as a modern styled HTML card."""
    cid    = c.get("id", "—")
    date   = c.get("date", "—")
    title  = c.get("title", "Notice")
    content= c.get("content", "")
    portal = c.get("payment_portal", "")
    kd     = c.get("key_dates", {})
    applic = c.get("applicable_to", "")

    # Key date blocks
    kd_html = ""
    if kd.get("class_commencement"):
        kd_html += f'<div style="display:flex;align-items:center;gap:10px;padding:10px 16px;background:#fff8e1;border-top:1px solid #eee"><span style="font-size:11px;font-weight:700;color:#e65100;text-transform:uppercase;letter-spacing:.04em;min-width:150px">📅 Class Commencement</span><span style="font-size:13px;font-weight:600;color:#e65100">{kd["class_commencement"]}</span></div>'
    if kd.get("early_bird_deadline"):
        kd_html += f'<div style="display:flex;align-items:center;gap:10px;padding:10px 16px;background:#e8f5e9;border-top:1px solid #eee"><span style="font-size:11px;font-weight:700;color:#2e7d32;text-transform:uppercase;letter-spacing:.04em;min-width:150px">🎯 Early Bird Deadline</span><span style="font-size:13px;font-weight:600;color:#2e7d32">{kd["early_bird_deadline"]} — {kd.get("early_bird_discount","10%")} discount</span></div>'

    portal_html = ""
    if portal:
        portal_html = f'<div style="display:flex;align-items:center;gap:10px;padding:10px 16px;border-top:1px solid #eee"><span style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:.04em;min-width:150px">🌐 Payment Portal</span><a href="{portal}" style="color:#667eea;font-size:13px;word-break:break-all">{portal}</a></div>'

    applic_html = ""
    if applic:
        applic_html = f'<div style="display:flex;align-items:center;gap:10px;padding:10px 16px;background:#f8f9ff;border-top:1px solid #eee"><span style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:.04em;min-width:150px">👥 Applicable To</span><span style="font-size:13px;color:#333">{applic}</span></div>'

    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif">
  <div style="background:linear-gradient(135deg,#e65100,#bf360c);color:#fff;padding:12px 16px;border-radius:10px 10px 0 0;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px">
    <b style="font-size:14px">📢 {title}</b>
    <span style="font-size:11px;opacity:.85">No. {cid} &nbsp;|&nbsp; {date}</span>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;border-radius:0 0 10px 10px;overflow:hidden">
    <div style="padding:12px 16px;background:#fff3e0;font-size:13px;color:#333;line-height:1.7">{content}</div>
    {kd_html}
    {applic_html}
    {portal_html}
    <div style="display:flex;align-items:center;gap:10px;padding:10px 16px;background:#f0f4ff;border-top:1px solid #eee">
      <span style="font-size:11px;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:.04em;min-width:150px">📝 Action Required</span>
      <span style="font-size:13px;color:#333">Pay fee via portal &amp; submit fund transfer details at college office to collect receipt.</span>
    </div>
  </div>
</div>"""


def build_all_circulars_table() -> str:
    """Render all circulars as a summary table."""
    if not _CIRCULARS:
        return _info_card("📢 Circulars & Announcements", [
            ("Status", "No circulars available at this time."),
        ])

    rows = ""
    for i, c in enumerate(_CIRCULARS):
        bg = "#f8f9ff" if i % 2 == 0 else "#fff"
        rows += (f'<tr style="background:{bg}">'
                 f'<td style="border:1px solid #ddd;padding:9px 14px;font-size:13px;text-align:center">{i+1}</td>'
                 f'<td style="border:1px solid #ddd;padding:9px 14px;font-size:13px;font-weight:700;color:#e65100">{c.get("id","—")}</td>'
                 f'<td style="border:1px solid #ddd;padding:9px 14px;font-size:13px">{c.get("date","—")}</td>'
                 f'<td style="border:1px solid #ddd;padding:9px 14px;font-size:13px">{c.get("title","—")}</td>'
                 f'</tr>')

    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif">
  <div style="background:linear-gradient(135deg,#e65100,#bf360c);color:#fff;padding:10px 16px;border-radius:8px 8px 0 0;display:flex;justify-content:space-between;align-items:center">
    <b>📢 NBKR Circulars &amp; Announcements</b>
    <span style="font-size:11px;opacity:.85">{len(_CIRCULARS)} Notice(s)</span>
  </div>
  <div style="overflow-x:auto;border:1px solid #ddd;border-top:none;border-radius:0 0 8px 8px">
    <table style="width:100%;border-collapse:collapse;min-width:500px">
      <thead><tr style="background:#fff3e0">
        <th style="border:1px solid #ddd;padding:9px 14px;font-size:13px;font-weight:700;text-align:center;width:50px">S.No</th>
        <th style="border:1px solid #ddd;padding:9px 14px;font-size:13px;font-weight:700;text-align:left">Notice No.</th>
        <th style="border:1px solid #ddd;padding:9px 14px;font-size:13px;font-weight:700;text-align:left">Date</th>
        <th style="border:1px solid #ddd;padding:9px 14px;font-size:13px;font-weight:700;text-align:left">Subject</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <p style="font-size:11px;color:#888;margin-top:5px;font-family:Segoe UI,sans-serif">
    Ask "show circular B-10/2026/01" or "fee notice" for full details.
  </p>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge base loading
# ─────────────────────────────────────────────────────────────────────────────
def load_knowledge_base() -> List[Dict]:
    docs = []
    
    # Load department info
    if os.path.exists("nbkr_department_info.json"):
        try:
            with open("nbkr_department_info.json", "r", encoding="utf-8") as f:
                dept = json.load(f)
            if "vision" in dept:
                docs.append({
                    "text": "department vision. What is the vision of the department? Tell me about the vision of the department. Department Vision Statement.",
                    "display_text": f"Department Vision:\n{dept['vision']}",
                    "type": "department",
                    "key": "vision"
                })
            if "mission" in dept:
                docs.append({
                    "text": "department mission. What is the mission of the department? Tell me about the mission of the department. Department Mission Statement.",
                    "display_text": f"Department Mission:\n{dept['mission']}",
                    "type": "department",
                    "key": "mission"
                })
            if "hod_message" in dept:
                docs.append({
                    "text": "HOD message. What is the message from the HOD H.O.D.? HOD's message. Tell me about the HOD message.",
                    "display_text": f"Message from HOD:\n{dept['hod_message']}",
                    "type": "department",
                    "key": "hod_message"
                })
            if "about" in dept:
                docs.append({
                    "text": "about the department. Tell me about the AI & DS department. General department info and information.",
                    "display_text": f"About the Department:\n{dept['about']}",
                    "type": "department",
                    "key": "about"
                })
            if "peos" in dept:
                peo_text = "Department Program Educational Objectives (PEOs):\n" + "\n".join(dept["peos"])
                docs.append({
                    "text": "department PEOs. What are the Program Educational Objectives PEOs of the department? Tell me about the PEOs.",
                    "display_text": peo_text,
                    "type": "department",
                    "key": "peos"
                })
            if "psos" in dept:
                pso_text = "Department Program Specific Outcomes (PSOs):\n" + "\n".join(dept["psos"])
                docs.append({
                    "text": "department PSOs. What are the Program Specific Outcomes PSOs of the department? Tell me about the PSOs.",
                    "display_text": pso_text,
                    "type": "department",
                    "key": "psos"
                })
            print("✓ Loaded department info into knowledge base")
        except Exception as e:
            print(f"⚠ Error loading department info: {e}")

    # Load college/facilities info into the searchable knowledge base
    # (previously only reachable via the hardcoded college_kw keyword list —
    # this makes it findable by hybrid search too, as a safety net)
    if os.path.exists("nbkr_college_info.json"):
        try:
            with open("nbkr_college_info.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            # 1. "college"
            coll = data.get("college", {})
            if "name" in coll:
                docs.append({
                    "text": f"college name institute: {coll['name']}. What is the name of the college?",
                    "display_text": f"College Name:\n{coll['name']}",
                    "type": "college",
                    "key": "name"
                })
            if "short_name" in coll:
                docs.append({
                    "text": f"college short name abbreviation: {coll['short_name']}. What is the short name of the college?",
                    "display_text": f"College Short Name:\n{coll['short_name']}",
                    "type": "college",
                    "key": "short_name"
                })
            if "established" in coll:
                docs.append({
                    "text": f"college establishment year established: {coll['established']}. When was the college established?",
                    "display_text": f"Established:\n{coll['established']}",
                    "type": "college",
                    "key": "established"
                })
            if "location" in coll:
                loc = coll["location"]
                loc_str = f"{loc.get('village', '')}, {loc.get('district', '')}, {loc.get('state', '')}, {loc.get('country', '')}"
                docs.append({
                    "text": f"college location village district state country: {loc_str}. Where is the college located?",
                    "display_text": f"Location:\n{loc_str}",
                    "type": "college",
                    "key": "location"
                })
            if "affiliation" in coll:
                docs.append({
                    "text": f"college affiliation university: {coll['affiliation']}. What is the college affiliation?",
                    "display_text": f"Affiliation:\n{coll['affiliation']}",
                    "type": "college",
                    "key": "affiliation"
                })
            if "autonomous_status" in coll:
                docs.append({
                    "text": f"college autonomous status: {coll['autonomous_status']}. Is the college autonomous?",
                    "display_text": f"Autonomous Status:\n{'Yes' if coll['autonomous_status'] else 'No'}",
                    "type": "college",
                    "key": "autonomous_status"
                })
            if "approvals" in coll:
                appr = coll["approvals"]
                appr_list = [f"{k.upper()}: {'Yes' if v is True else v}" for k, v in appr.items()]
                appr_str = ", ".join(appr_list)
                docs.append({
                    "text": f"college approvals recognition aicte ugc: {appr_str}. What are the college approvals?",
                    "display_text": f"Approvals & Recognitions:\n{appr_str}",
                    "type": "college",
                    "key": "approvals"
                })
            if "accreditation" in coll:
                acc = coll["accreditation"]
                naac = acc.get("naac", {})
                nba = acc.get("nba", {})
                acc_str = f"NAAC Grade: {naac.get('grade', '')}, NBA Status: {nba.get('status', '')} ({nba.get('type', '')} for {nba.get('programs', '')})"
                docs.append({
                    "text": f"college accreditation naac grade nba status: {acc_str}. What are the accreditations of the college?",
                    "display_text": f"Accreditation:\n{acc_str}",
                    "type": "college",
                    "key": "accreditation"
                })
            if "vision" in coll:
                docs.append({
                    "text": f"college vision statement: {coll['vision']}. What is the vision of the college?",
                    "display_text": f"College Vision:\n{coll['vision']}",
                    "type": "college",
                    "key": "vision"
                })

            # 2. "programs_offered"
            prog = data.get("programs_offered", {})
            for btech in prog.get("undergraduate", []):
                docs.append({
                    "text": f"college program course undergraduate btech: {btech}. Is there {btech} offered?",
                    "display_text": f"Undergraduate Program:\n{btech}",
                    "type": "college",
                    "key": "undergraduate_program"
                })
            for pg in prog.get("postgraduate", []):
                docs.append({
                    "text": f"college program course postgraduate pg: {pg}. Is there {pg} offered?",
                    "display_text": f"Postgraduate Program:\n{pg}",
                    "type": "college",
                    "key": "postgraduate_program"
                })

            # 3. "campus"
            camp = data.get("campus", {})
            if "campus_area" in camp:
                docs.append({
                    "text": f"college campus area size: {camp['campus_area']}. How big is the campus?",
                    "display_text": f"Campus Area:\n{camp['campus_area']}",
                    "type": "college",
                    "key": "campus_area"
                })
            for facility in camp.get("facilities", []):
                docs.append({
                    "text": f"college campus facility: {facility}. Is there a {facility} on campus? Do you have a {facility}?",
                    "display_text": f"{facility}",
                    "type": "college",
                    "key": "facility"
                })

            # 4. "student_chapters"
            for chap in data.get("student_chapters", []):
                docs.append({
                    "text": f"college student chapter club activity: {chap}. Is there a {chap} chapter on campus?",
                    "display_text": f"Student Chapter / Club:\n{chap}",
                    "type": "college",
                    "key": "student_chapter"
                })

            # 5. "placements"
            plac = data.get("placements", {})
            for rec in plac.get("major_recruiters", []):
                docs.append({
                    "text": f"college placement major recruiter company hiring: {rec}. Does {rec} recruit from college?",
                    "display_text": f"Major Recruiter:\n{rec}",
                    "type": "college",
                    "key": "recruiter"
                })
            for focus in plac.get("placement_focus", []):
                docs.append({
                    "text": f"college placement training focus skill: {focus}. What is the placement training focus?",
                    "display_text": f"Placement Focus:\n{focus}",
                    "type": "college",
                    "key": "placement_focus"
                })

            # 6. "department"
            dept = data.get("department", {})
            if "name" in dept:
                docs.append({
                    "text": f"department name: {dept['name']}. What is the department name?",
                    "display_text": f"Department Name:\n{dept['name']}",
                    "type": "college",
                    "key": "dept_name"
                })
            if "short_name" in dept:
                docs.append({
                    "text": f"department short name: {dept['short_name']}. What is the department short name?",
                    "display_text": f"Department Short Name:\n{dept['short_name']}",
                    "type": "college",
                    "key": "dept_short_name"
                })
            if "overview" in dept:
                docs.append({
                    "text": f"department overview info description: {dept['overview']}. What is the department about?",
                    "display_text": f"Department Overview:\n{dept['overview']}",
                    "type": "college",
                    "key": "dept_overview"
                })
            if "approved_intake" in dept:
                docs.append({
                    "text": f"department approved intake seats: {dept['approved_intake']}. What is the intake of the department?",
                    "display_text": f"Approved Intake:\n{dept['approved_intake']} seats",
                    "type": "college",
                    "key": "dept_approved_intake"
                })
            for obj in dept.get("objectives", []):
                docs.append({
                    "text": f"department objective goal: {obj}. What is the objective of the department?",
                    "display_text": f"Department Objective:\n{obj}",
                    "type": "college",
                    "key": "dept_objective"
                })
            for sub in dept.get("subjects", []):
                docs.append({
                    "text": f"department subject course: {sub}. Do we study {sub} in the department?",
                    "display_text": f"Department Subject:\n{sub}",
                    "type": "college",
                    "key": "dept_subject"
                })
            for lab in dept.get("laboratories", []):
                docs.append({
                    "text": f"department laboratory lab: {lab}. Is there a {lab} in the department?",
                    "display_text": f"Department Laboratory:\n{lab}",
                    "type": "college",
                    "key": "dept_laboratory"
                })
            for opp in dept.get("career_opportunities", []):
                docs.append({
                    "text": f"department career opportunity job role: {opp}. What career opportunities are there?",
                    "display_text": f"Career Opportunity:\n{opp}",
                    "type": "college",
                    "key": "dept_career_opportunity"
                })
            for skill in dept.get("recommended_skills", []):
                docs.append({
                    "text": f"department recommended skill technology: {skill}. What skills are recommended?",
                    "display_text": f"Recommended Skill:\n{skill}",
                    "type": "college",
                    "key": "dept_recommended_skill"
                })

            # 7. "events_and_activities"
            for ev in data.get("events_and_activities", []):
                docs.append({
                    "text": f"college event activity technical symposium workshop: {ev}. What events are held?",
                    "display_text": f"Event / Activity:\n{ev}",
                    "type": "college",
                    "key": "event_activity"
                })

            print("✓ Loaded college info into knowledge base")
        except Exception as e:
            print(f"⚠ Error loading college info: {e}")

    # Load bus fees
    if os.path.exists("nbkr_bus_fees.json"):
        try:
            with open("nbkr_bus_fees.json", "r", encoding="utf-8") as f:
                bus_data = json.load(f)
            for route_name, stops in bus_data.get("bus_fees", {}).items():
                route_label = route_name.replace("_", " ").title()
                for stop_entry in stops:
                    stop = stop_entry.get("stop", "")
                    fee = stop_entry.get("fee", 0)
                    text = f"The bus fee for {stop} on the {route_label} is Rs {fee} per year."
                    docs.append({
                        "text": text,
                        "type": "bus_fee",
                        "stop": stop,
                        "fee": fee,
                        "route": route_label
                    })
            print("✓ Loaded bus fees into knowledge base")
        except Exception as e:
            print(f"⚠ Error loading bus fees: {e}")

    if os.path.exists("aids_faculty_data.json"):
        with open("aids_faculty_data.json","r",encoding="utf-8") as f:
            faculty_list = json.load(f)
        for fac in faculty_list:
            name  = fac.get("name","")
            desig = fac.get("designation","")
            spec  = fac.get("specialization","")
            qual  = fac.get("Qualification","")
            phone = fac.get("phone","")
            email = fac.get("email","")
            doj   = fac.get("date_of_joining","")
            text  = (f"{name} is {desig} in AI & DS Department at NBKR Institute. "
                     f"Specialization: {spec}.")
            if qual:  text += f" Qualification: {qual}."
            if phone: text += f" Phone: {phone}."
            if email: text += f" Email: {email}."
            if doj:   text += f" Date of Joining: {doj}."
            docs.append({"text": text, "type": "faculty",
                         "name": name, "designation": desig, "data": fac})

    if os.path.exists("aids_timetable_data.json"):
        with open("aids_timetable_data.json","r",encoding="utf-8") as f:
            tt = json.load(f)
        subjects_map = tt.get("subjects", {})
        faculty_map  = tt.get("faculty", {})
        for section, days in tt.get("timetable",{}).items():
            if "4th_Year" in section:
                year_label = "4th Year 1st Semester"
            elif "3rd_Year" in section:
                year_label = "3rd Year 1st Semester"
            elif "2nd_Year" in section:
                year_label = "2nd Year 2nd Semester"
            else:
                year_label = "1st Year 1st Semester"
            sec_label = section.replace("_"," ")
            for day, periods in days.items():
                lines = [f"{sec_label} {year_label} {day} timetable:"]
                for slot, subj in periods.items():
                    full_subj = subjects_map.get(subj.split("(")[0].strip(), subj)
                    lines.append(f"  {slot} → {subj} ({full_subj})")
                docs.append({"text":"\n".join(lines),"type":"timetable",
                             "section":section,"day":day,"year":year_label})

    for kb_file in ["nbkr_knowledge_base.json","aids_timetable_kb.json"]:
        if os.path.exists(kb_file):
            with open(kb_file,"r",encoding="utf-8") as f:
                for key, val in json.load(f).items():
                    if val and str(val).strip():
                        docs.append({"text":f"{key}: {val}","type":"general","key":key})

    if os.path.exists("nbkr_circulars.json"):
        with open("nbkr_circulars.json","r",encoding="utf-8") as f:
            for c in json.load(f):
                text = (f"Circular {c.get('id','')} dated {c.get('date','')}: "
                        f"{c.get('title','')}. {c.get('content','')}")
                docs.append({"text": text, "type": "circular",
                             "id": c.get("id",""), "date": c.get("date",""),
                             "title": c.get("title",""), "data": c})

    print(f"✓ Knowledge base: {len(docs)} documents loaded")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# RAG — FAISS initialisation
# ─────────────────────────────────────────────────────────────────────────────
def initialize_rag() -> bool:
    global embeddings_model, faiss_index, knowledge_docs, doc_embeddings
    print("🔄 Initialising RAG engine …")
    try:
        from sentence_transformers import SentenceTransformer
        embeddings_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("✓ Sentence-transformer model loaded")
    except Exception as e:
        print(f"⚠ Embedding model failed: {e}"); return False

    knowledge_docs = load_knowledge_base()
    if not knowledge_docs:
        print("⚠ No documents found"); return False

    try:
        import faiss
        texts = [d["text"] for d in knowledge_docs]
        vecs  = embeddings_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        doc_embeddings = vecs.astype("float32")
        dim   = vecs.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(doc_embeddings)
        print(f"✓ FAISS index built  ({len(knowledge_docs)} vectors, dim={dim})")

        # Build BM25 keyword index alongside the vector index
        global bm25_index, bm25_tokenized_corpus
        from rank_bm25 import BM25Okapi
        bm25_tokenized_corpus = [_bm25_tokenize(t) for t in texts]
        bm25_index = BM25Okapi(bm25_tokenized_corpus)
        print(f"✓ BM25 index built ({len(bm25_tokenized_corpus)} documents)")
    except Exception as e:
        print(f"⚠ FAISS failed: {e}"); return False

    # Load cross-encoder reranker (re-scores top candidates using query+doc
    # together, catching relevance nuances pure vector/BM25 scoring misses)
    global reranker_model
    try:
        from sentence_transformers import CrossEncoder
        reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✓ Reranker model loaded")
    except Exception as e:
        print(f"⚠ Reranker failed to load, continuing without it: {e}")
        reranker_model = None

    return True


def retrieve(qa: QueryAnalysis, top_k: int = TOP_K) -> List[Tuple[Dict, float]]:
    if embeddings_model is None or faiss_index is None:
        return []
    search_text = qa.expanded if qa.expanded.strip() else qa.original
    bge_query = "Represent this sentence for searching relevant passages: " + search_text
    q_vec = embeddings_model.encode([bge_query], normalize_embeddings=True).astype("float32")
    scores, indices = faiss_index.search(q_vec, top_k)
    return [(knowledge_docs[idx], float(score))
            for score, idx in zip(scores[0], indices[0]) if idx < len(knowledge_docs)]


def retrieve_filtered(qa: QueryAnalysis, doc_type: str) -> List[Tuple[Dict, float]]:
    if embeddings_model is None or doc_embeddings is None:
        return []
    matching_indices = [i for i, d in enumerate(knowledge_docs) if d["type"] == doc_type]
    if not matching_indices:
        return []
    
    search_text = qa.expanded if qa.expanded.strip() else qa.original
    bge_query = "Represent this sentence for searching relevant passages: " + search_text
    q_vec = embeddings_model.encode([bge_query], normalize_embeddings=True)[0]
    
    filtered_embs = doc_embeddings[matching_indices]
    scores = np.dot(filtered_embs, q_vec)
    
    results = []
    for idx_in_filtered, score in enumerate(scores):
        original_idx = matching_indices[idx_in_filtered]
        results.append((knowledge_docs[original_idx], float(score)))
        
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def hybrid_retrieve(qa: QueryAnalysis, top_k: int = TOP_K, alpha: float = 0.5) -> List[Tuple[Dict, float]]:
    """
    Combine BM25 keyword score + FAISS vector score into one ranked list.
    alpha weights vector score vs keyword score (0.5 = equal weight).
    Falls back to vector-only retrieve() if BM25 isn't available.
    """
    if bm25_index is None or embeddings_model is None or faiss_index is None:
        return retrieve(qa, top_k)

    search_text = qa.expanded if qa.expanded.strip() else qa.original
    query_tokens = _bm25_tokenize(search_text)

    # BM25 scores across ALL docs, normalized to 0-1
    bm25_scores = bm25_index.get_scores(query_tokens)
    max_bm25 = max(bm25_scores) if len(bm25_scores) and max(bm25_scores) > 0 else 1.0
    bm25_scores_norm = [s / max_bm25 for s in bm25_scores]

    # Vector scores across ALL docs (reuse doc_embeddings, already normalized)
    bge_query = "Represent this sentence for searching relevant passages: " + search_text
    q_vec = embeddings_model.encode([bge_query], normalize_embeddings=True)[0]
    vector_scores = doc_embeddings @ q_vec  # cosine similarity, already 0-1 range

    combined = [
        (i, alpha * float(vector_scores[i]) + (1 - alpha) * float(bm25_scores_norm[i]))
        for i in range(len(knowledge_docs))
    ]
    combined.sort(key=lambda x: x[1], reverse=True)

    # No reranker available — return hybrid-scored results as before
    if reranker_model is None:
        return [(knowledge_docs[i], score) for i, score in combined[:top_k]]

    # Rerank only the top candidate pool (cheap: reranking 10 docs, not 256)
    candidate_pool = combined[:10]
    pairs = [(search_text, knowledge_docs[i]["text"]) for i, _ in candidate_pool]
    rerank_scores = reranker_model.predict(pairs)

    # Use rerank_scores ONLY to decide the new ORDER. Keep pairing each
    # candidate with its ORIGINAL hybrid score (candidate_pool[j][1]), not
    # the reranker's raw score, so existing confidence thresholds elsewhere
    # in the codebase keep working unchanged.
    order = sorted(
        range(len(candidate_pool)),
        key=lambda j: float(rerank_scores[j]),
        reverse=True
    )
    reranked = [
        (candidate_pool[j][0], candidate_pool[j][1])  # (doc_index, original_hybrid_score)
        for j in order
    ]

    return [(knowledge_docs[i], score) for i, score in reranked[:top_k]]



# ─────────────────────────────────────────────────────────────────────────────
# Timetable data & HTML builders
# ─────────────────────────────────────────────────────────────────────────────
_TT_DATA: Dict = {}

def load_timetable_data():
    global _TT_DATA
    if os.path.exists("aids_timetable_data.json"):
        with open("aids_timetable_data.json","r",encoding="utf-8") as f:
            _TT_DATA = json.load(f)

SLOT_ORDER = ["9-10","10-11","11-12","9-12","10-12","1-2","2-3","3-4","4-5","1-4","2-4","2-5","3-5"]
DAYS_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
SUBJECT_FULL = {
    # 1st Year
    "LAC":"Linear Algebra & Calculus",
    "EP":"Engineering Physics",
    "BEEE":"Basic Electrical & Electronics Engineering",
    "CP LAB":"Computer Programming Lab",
    "EP-LAB":"Engineering Physics Lab",
    "EEE WS":"EEE Workshop",
    "IT WS":"IT Workshop",
    "NGCS":"NSS/NCC/Community Service",
    "ENGINEERING GRAPHICS":"Engineering Graphics",
    "INTRODUCTION TO PROGRAMMING":"Introduction to Programming",
    # 2nd Year
    "AI":"Artificial Intelligence",
    "IDS":"Introduction to Data Science",
    "DL_CO":"Digital Logic & Computer Organization",
    "BE":"Business Environment",
    "SMDS":"Statistical Methods for Data Science",
    "DTI LAB":"Design Thinking & Innovation Lab",
    "IDS LAB":"Introduction to Data Science Lab",
    "AI LAB":"Artificial Intelligence Lab",
    "FULL STACK DEVELOPMENT LAB":"Full Stack Development Lab",
    # 3rd Year
    "SNA":"Social Network Analysis",
    "BDA":"Big Data Analytics",
    "AIF":"AI for Finance",
    "NLP":"Natural Language Processing",
    "DL":"Deep Learning",
    "AWPS":"Academic Writing and Public Speaking",
    "TPW.IPR":"Technical Paper Writing & IPR",
    "DL LAB":"Deep Learning Lab",
    "BD and DV LAB":"Big Data & Data Visualization Lab",
    "WORKSHOP":"Workshop",
    "SOFT SKILLS LAB":"Soft Skills Lab",
}

# ── Subject colour palette (matches the image style) ─────────────────────────
_SUBJ_COLORS = {
    # 1st Year
    "LAC":"#cce5ff","EP":"#fff3cd","BEEE":"#f8d7da","CP LAB":"#d4edda",
    "EP-LAB":"#ffeeba","EEE WS":"#e2d9f3","IT WS":"#d1ecf1",
    "NGCS":"#e2e3e5","ENGINEERING GRAPHICS":"#d6d8f7",
    "INTRODUCTION TO PROGRAMMING":"#ffd8a8",
    # 2nd Year
    "AI":"#cce5ff","IDS":"#e2d9f3","DL_CO":"#d4edda","BE":"#fff3cd",
    "SMDS":"#f8d7da","DTI LAB":"#ffd8a8","IDS LAB":"#d6d8f7",
    "AI LAB":"#d1ecf1","FULL STACK DEVELOPMENT LAB":"#c3e6cb",
    # 3rd Year
    "SNA":"#d4edda","BDA":"#fff3cd","AIF":"#cce5ff","NLP":"#e2d9f3",
    "DL":"#f8d7da","AWPS":"#ffeeba","TPW.IPR":"#c3e6cb",
    "DL LAB":"#ffd8a8","BD and DV LAB":"#d1ecf1",
    "WORKSHOP":"#ffd8a8","SOFT SKILLS LAB":"#d4edda",
}
# Darker text colour for each background
_SUBJ_TEXT = {
    "LAC":"#004085","EP":"#856404","BEEE":"#721c24","CP LAB":"#155724",
    "EP-LAB":"#856404","EEE WS":"#432874","IT WS":"#0c5460",
    "NGCS":"#383d41","ENGINEERING GRAPHICS":"#2d2f8f",
    "INTRODUCTION TO PROGRAMMING":"#7d3c00",
    "AI":"#004085","IDS":"#432874","DL_CO":"#155724","BE":"#856404",
    "SMDS":"#721c24","DTI LAB":"#7d3c00","IDS LAB":"#2d2f8f",
    "AI LAB":"#0c5460","FULL STACK DEVELOPMENT LAB":"#155724",
    "SNA":"#155724","BDA":"#856404","AIF":"#004085","NLP":"#432874",
    "DL":"#721c24","AWPS":"#856404","TPW.IPR":"#155724",
    "DL LAB":"#7d3c00","BD and DV LAB":"#0c5460",
    "WORKSHOP":"#7d3c00","SOFT SKILLS LAB":"#155724",
}

# ── Fixed atomic 1-hour columns (exactly as in the reference image) ───────────
# Compound slots like "9-12", "1-4", "2-5" expand into these via colspan.
_ATOMIC_COLS = ["9-10", "10-11", "11-12", "12-1", "1-2", "2-3", "3-4", "4-5"]

# Map each compound/atomic slot → list of atomic cols it covers
_SLOT_SPAN: Dict[str, List[str]] = {
    "9-10":  ["9-10"],
    "10-11": ["10-11"],
    "11-12": ["11-12"],
    "9-12":  ["9-10","10-11","11-12"],
    "10-12": ["10-11","11-12"],
    "12-1":  ["12-1"],
    "1-2":   ["1-2"],
    "2-3":   ["2-3"],
    "3-4":   ["3-4"],
    "4-5":   ["4-5"],
    "1-4":   ["1-2","2-3","3-4"],
    "2-4":   ["2-3","3-4"],
    "2-5":   ["2-3","3-4","4-5"],
    "3-5":   ["3-4","4-5"],
}
_ATOMIC_COLS_3RD_4TH = ["9:00-9:50", "9:50-10:40", "10:50-11:40", "11:40-12:30", "12:30-1:30", "1:30-2:20", "2:20-3:10", "3:10-4:00"]
_SLOT_SPAN_3RD_4TH = {
    "9:00-9:50": ["9:00-9:50"],
    "9:50-10:40": ["9:50-10:40"],
    "10:50-11:40": ["10:50-11:40"],
    "11:40-12:30": ["11:40-12:30"],
    "12:30-1:30": ["12:30-1:30"],
    "1:30-2:20": ["1:30-2:20"],
    "2:20-3:10": ["2:20-3:10"],
    "3:10-4:00": ["3:10-4:00"],
    "9:00-10:50": ["9:00-9:50", "9:50-10:40"],
    "9:50-12:30": ["9:50-10:40", "10:50-11:40", "11:40-12:30"],
    "10:50-12:30": ["10:50-11:40", "11:40-12:30"],
    "1:30-4:00": ["1:30-2:20", "2:20-3:10", "3:10-4:00"],
    "2:20-4:00": ["2:20-3:10", "3:10-4:00"],
    "9:00-12:30": ["9:00-9:50", "9:50-10:40", "10:50-11:40", "11:40-12:30"]
}
_LUNCH_VALS = {"L","U","N","C","H","BREAK","LUNCH"}

def _is_lab(subj: str) -> bool:
    """Return True if the subject name looks like a lab/workshop (yellow in reference)."""
    s = subj.upper()
    return any(w in s for w in ("LAB","WORKSHOP","GRAPHICS","WS","PROGRAMMING"))

def _render_cell(val: str, colspan: int) -> str:
    """Return a <td> for one cell, with correct colspan, colour, and style."""
    cs = f' colspan="{colspan}"' if colspan > 1 else ""
    base = ("border:1px solid #bbb;padding:6px 8px;text-align:center;"
            "vertical-align:middle;font-size:11.5px;")

    v = val.strip()

    # Lunch / break letters — special grey column
    if v in _LUNCH_VALS:
        letter_style = ("font-weight:900;font-size:15px;letter-spacing:1px;"
                        "color:#666;font-style:italic")
        if v == "BREAK":
            letter_style = "font-weight:900;font-size:12px;color:#666;letter-spacing:1px"
        return (f'<td{cs} style="{base}background:#f0f0f0;width:52px;min-width:52px">'
                f'<span style="{letter_style}">{v}</span></td>')

    # Empty
    if not v or v == "-":
        return f'<td{cs} style="{base}background:#fff;color:#ccc"></td>'

    # Parse "SUBJECT (FAC1, FAC2)"
    m = re.match(r'^(.+?)\s*\((.+)\)$', v)
    subj = m.group(1).strip() if m else v
    fac  = m.group(2).strip() if m else ""

    # Labs → yellow (exactly like reference image)
    if _is_lab(subj) or colspan >= 3:
        bg, fg = "#ffe066", "#5a4000"
    else:
        bg, fg = "#fff", "#1a1a2e"
        for code, colour in _SUBJ_COLORS.items():
            if code in subj:
                bg = colour
                fg = _SUBJ_TEXT.get(code, "#1a1a2e")
                break

    fac_html = (f'<div style="font-size:10px;color:{fg};opacity:.8;margin-top:2px">'
                f'( {fac} )</div>') if fac else ""
    return (f'<td{cs} style="{base}background:{bg};color:{fg};font-weight:700">'
            f'<div style="font-size:11.5px">{subj}</div>{fac_html}</td>')


def _build_day_row(day_data: Dict, active_cols: List[str], day_label: str, is_new_tt: bool = False) -> str:
    """
    Build one <tr> for a day.
    Uses colspan for multi-hour slots; skips atomic cols already consumed.
    Only renders columns that are in active_cols.
    """
    day_th = ('style="border:1px solid #bbb;padding:8px 10px;font-weight:700;'
              'font-size:12px;background:#fce4ec;color:#880e4f;'
              'white-space:nowrap;text-align:center;min-width:52px"')

    # Build a map: atomic_col → (val, colspan) for this day
    col_map: Dict[str, tuple] = {}
    slot_span = _SLOT_SPAN_3RD_4TH if is_new_tt else _SLOT_SPAN
    lunch_col = "12:30-1:30" if is_new_tt else "12-1"

    for slot_key, val in day_data.items():
        covered = slot_span.get(slot_key, [slot_key])
        # Only count cols that are active
        active_covered = [c for c in covered if c in active_cols]
        if not active_covered:
            continue
        span = len(active_covered)
        col_map[active_covered[0]] = (val, span)
        # Mark subsequent cols as "skip"
        for c in active_covered[1:]:
            col_map[c] = ("__skip__", 0)

    cells = f'<td {day_th}>{day_label}</td>'
    for col in active_cols:
        if col not in col_map:
            if col == lunch_col:
                lunch_letters = {"Mon": "L", "Tue": "U", "Wed": "N", "Thu": "C", "Fri": "H", "Sat": "BREAK"}
                cells += _render_cell(lunch_letters.get(day_label, "LUNCH"), 1)
            else:
                cells += _render_cell("", 1)
        elif col_map[col][0] == "__skip__":
            pass  # consumed by colspan
        else:
            val, span = col_map[col]
            cells += _render_cell(val, span)

    return f'<tr>{cells}</tr>'


def build_section_week_table(section_key: str) -> str:
    """Build the reference-image-style timetable with proper colspan for labs."""
    tt   = _TT_DATA.get("timetable", {})
    data = tt.get(section_key, {})
    if not data:
        return (f'<p style="font-family:Segoe UI,sans-serif;color:#555">'
                f'No timetable found for {section_key.replace("_"," ")}.</p>')

    is_new_tt = "3rd_Year" in section_key or "4th_Year" in section_key
    atomic_cols = _ATOMIC_COLS_3RD_4TH if is_new_tt else _ATOMIC_COLS
    slot_span = _SLOT_SPAN_3RD_4TH if is_new_tt else _SLOT_SPAN
    lunch_col = "12:30-1:30" if is_new_tt else "12-1"

    # Determine which atomic columns actually have data (drop empty trailing cols)
    used_atomic: set = set()
    for day_data in data.values():
        for slot_key in day_data:
            for ac in slot_span.get(slot_key, [slot_key]):
                used_atomic.add(ac)

    # Keep only atomic cols that are used, in fixed order, always keeping lunch_col
    active_cols = [c for c in atomic_cols if (c == lunch_col or c in used_atomic)]

    label      = section_key.replace("_", " ")
    year_label = ("4th Year · 1st Semester" if "4th_Year" in section_key else
                  "3rd Year · 1st Semester" if "3rd_Year" in section_key else
                  "2nd Year · 2nd Semester" if "2nd_Year" in section_key else
                  "1st Year · 1st Semester")

    # Header
    th = ('style="border:1px solid #bbb;padding:8px 10px;text-align:center;'
          'font-size:11.5px;font-weight:700;background:#2d3a6b;color:#fff;white-space:nowrap"')
    th_lunch = ('style="border:1px solid #bbb;padding:6px;text-align:center;'
                'font-size:11px;font-weight:700;background:#555;color:#fff;'
                'width:60px;min-width:60px"')
    col_headers = "".join(
        f'<th {th_lunch}>12:30 – 1:30<br><span style="font-size:9px;font-weight:400;opacity:0.85">Lunch Break</span></th>' if (is_new_tt and c == "12:30-1:30") else
        f'<th {th_lunch}>12 – 1<br><span style="font-size:9px;font-weight:400;opacity:0.85">Lunch Break</span></th>' if (not is_new_tt and c == "12-1") else
        f'<th {th}>{c.replace("-"," – ")}</th>'
        for c in active_cols
    )
    header_row = f'<tr><th {th}>Day \\ Hour</th>{col_headers}</tr>'

    # Day rows
    rows = ""
    for day in DAYS_ORDER:
      day_data = data.get(day, {})
      rows += _build_day_row(day_data, active_cols, day[:3], is_new_tt)

    # Legend
    subj_map = _TT_DATA.get("subjects", {})
    fac_map  = _TT_DATA.get("faculty", {})
    used_subj: set = set()
    used_fac:  set = set()
    for day_data in data.values():
      for val in day_data.values():
        m = re.match(r'^(.+?)\s*\((.+)\)$', val.strip())
        if m:
          used_subj.add(m.group(1).strip())
          for f in m.group(2).split(","):
            used_fac.add(f.strip())

    td_s = 'style="padding:2px 10px;font-size:11px;color:#333;white-space:nowrap"'
    td_v = 'style="padding:2px 10px;font-size:11px;color:#555"'
    subj_rows = "".join(
        f'<tr><td {td_s}>( <b>{k}</b> )</td><td {td_v}>{v}</td></tr>'
        for k, v in subj_map.items() if k in used_subj
    )
    fac_rows = "".join(
        f'<tr><td {td_s}>( <b>{k}</b> )</td><td {td_v}>{v}</td></tr>'
        for k, v in fac_map.items() if k in used_fac
    )
    legend_html = ""
    if subj_rows or fac_rows:
      legend_html = f"""
<div style="display:flex;gap:32px;flex-wrap:wrap;margin-top:8px;padding:10px 14px;
            background:#f8f9fa;border:1px solid #ddd;border-radius:6px">
  <table style="border-collapse:collapse"><tbody>{subj_rows}</tbody></table>
  <table style="border-collapse:collapse"><tbody>{fac_rows}</tbody></table>
</div>"""

    return f"""
<div class="timetable-card-container" style="margin:8px 0;font-family:'Segoe UI',Arial,sans-serif">
  <div style="background:linear-gradient(135deg,#2d3a6b,#4a5568);color:#fff;
              padding:10px 16px;border-radius:8px 8px 0 0;
              display:flex;justify-content:space-between;align-items:center">
    <b>📅 AI &amp; DS — {label} Weekly Timetable</b>
    <span style="font-size:11px;opacity:.85">{year_label}</span>
  </div>
  <div style="overflow-x:auto;border:1px solid #bbb;border-top:none;
              border-radius:0 0 8px 8px;background:#fff">
    <table class="timetable-table" style="width:100%;border-collapse:collapse;min-width:680px">
      <thead>{header_row}</thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  {legend_html}
  <div class="timetable-actions" style="margin-top: 12px; display: flex; gap: 10px; justify-content: flex-end; flex-wrap: wrap;">
    <button class="timetable-action-btn png-btn" onclick="downloadAsPNG(this)" style="padding: 6px 12px; font-size: 11.5px; font-weight: 600; border-radius: 6px; cursor: pointer; display: flex; align-items: center; gap: 6px; background: #6366f1; border: none; color: white;">
      <span>📸</span> Download Image (PNG)
    </button>
    <button class="timetable-action-btn csv-btn" onclick="downloadAsExcel(this)" style="padding: 6px 12px; font-size: 11.5px; font-weight: 600; border-radius: 6px; cursor: pointer; display: flex; align-items: center; gap: 6px; background: #10b981; border: none; color: white;">
      <span>📊</span> Export CSV
    </button>
    <button class="timetable-action-btn pdf-btn" onclick="downloadAsPDF(this)" style="padding: 6px 12px; font-size: 11.5px; font-weight: 600; border-radius: 6px; cursor: pointer; display: flex; align-items: center; gap: 6px; background: #ef4444; border: none; color: white;">
      <span>📄</span> Print / PDF
    </button>
  </div>
</div>"""


def build_day_table(section_key, day):
    tt   = _TT_DATA.get("timetable",{})
    data = tt.get(section_key,{}).get(day,{})
    if not data:
        return (f'<p style="font-family:Segoe UI,sans-serif;color:#555">'
                f'No classes for {section_key.replace("_"," ")} on {day}.</p>')
    is_new_tt = "3rd_Year" in section_key or "4th_Year" in section_key
    atomic_cols = _ATOMIC_COLS_3RD_4TH if is_new_tt else _ATOMIC_COLS
    slot_span = _SLOT_SPAN_3RD_4TH if is_new_tt else _SLOT_SPAN
    
    slot_order_map = {s: i for i, s in enumerate(atomic_cols)}
    slots = sorted(data.keys(), key=lambda s: slot_order_map.get(
        slot_span.get(s, [s])[0], 99))
    label = section_key.replace("_"," ")
    th = ('style="border:1px solid #bbb;padding:8px 12px;text-align:center;'
          'font-size:12px;font-weight:700;background:#2d3a6b;color:#fff"')
    rows = ""
    for slot in slots:
        val = data[slot]
        covered = slot_span.get(slot, [slot])
        active  = [c for c in covered if c in atomic_cols]
        span    = len(active)
        if is_new_tt:
            label_s = f'{active[0].split("-")[0]} – {active[-1].split("-")[1]}'
        else:
            label_s = f'{active[0].replace("-"," – ")} – {active[-1].split("-")[1]}:00' if span > 1 else slot.replace("-"," – ")
        rows += (f'<tr><td style="border:1px solid #bbb;padding:8px 12px;font-weight:700;'
                 f'background:#e8eeff;color:#1a1a2e;font-size:12px;white-space:nowrap">'
                 f'{label_s}</td>{_render_cell(val, 1)}</tr>')
    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',Arial,sans-serif">
  <div style="background:linear-gradient(135deg,#2d3a6b,#4a5568);color:#fff;padding:10px 16px;border-radius:8px 8px 0 0">
    <b>📅 {label} — {day} Schedule</b>
  </div>
  <div style="overflow-x:auto;border:1px solid #bbb;border-top:none;border-radius:0 0 8px 8px;background:#fff">
    <table style="width:100%;border-collapse:collapse">
      <thead><tr><th {th}>Time Slot</th><th {th}>Subject (Faculty)</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""


def build_subject_table(subject_code, section_key=None):
    tt = _TT_DATA.get("timetable",{})
    # Search all sections across all years, or just the specified one
    sections = [section_key] if section_key else list(tt.keys())
    rows = ""
    for sec in sections:
        for day in DAYS_ORDER:
            for slot, val in tt.get(sec,{}).get(day,{}).items():
                if subject_code in val:
                    sec_label = sec.replace("_"," ")
                    rows += (f'<tr><td {_TD_TT}>{sec_label}</td>'
                             f'<td {_TD_TT}>{day}</td>'
                             f'<td {_TD_TIME}>{slot}</td>'
                             f'{_cell(val)}</tr>')
    if not rows:
        return f'<p style="font-family:Segoe UI,sans-serif">No <b>{subject_code}</b> classes found.</p>'
    full_name = SUBJECT_FULL.get(subject_code, subject_code)
    scope = section_key.replace("_"," ") if section_key else "All Sections"
    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif">
  <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:10px 16px;border-radius:8px 8px 0 0">
    <b>📚 {full_name} ({subject_code}) — {scope}</b>
  </div>
  <div style="overflow-x:auto;border:1px solid #ddd;border-top:none;border-radius:0 0 8px 8px">
    <table style="width:100%;border-collapse:collapse">
      <thead><tr style="background:#f0f4ff">
        <th {_TH_TT}>Section</th><th {_TH_TT}>Day</th><th {_TH_TT}>Time</th><th {_TH_TT}>Subject</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""


def build_all_sections_overview(year: int = 1):
    tt = _TT_DATA.get("timetable",{})
    if year == 4:
        sections = ["4th_Year_Section_A","4th_Year_Section_B"]
        title = "4th Year — All Sections Overview"
    elif year == 3:
        sections = ["3rd_Year_Section_A","3rd_Year_Section_B"]
        title = "3rd Year — All Sections Overview"
    elif year == 2:
        sections = ["2nd_Year_Section_A","2nd_Year_Section_B"]
        title = "2nd Year — All Sections Overview"
    else:
        sections = ["1st_Year_Section_A","1st_Year_Section_B","1st_Year_Section_C","1st_Year_Section_D"]
        title = "1st Year — All Sections Overview"
    day_headers = "".join(f'<th {_TH_TT}>{d[:3]}</th>' for d in DAYS_ORDER)
    header = (f'<tr style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff">'
              f'<th {_TH_TT}>Section</th>{day_headers}</tr>')
    rows = ""
    for sec in sections:
        cells = ""
        for day in DAYS_ORDER:
            day_data = tt.get(sec,{}).get(day,{})
            if day_data:
                code = list(day_data.values())[0].split("(")[0].strip()
                cells += f'<td style="border:1px solid #ccc;padding:7px 9px;text-align:center;font-size:11px;color:#1a1a2e">{code}</td>'
            else:
                cells += f'<td {_TD_EMPTY}>—</td>'
        label = sec.replace("_"," ").replace("1st Year ","").replace("2nd Year ","")
        rows += (f'<tr><td style="border:1px solid #ccc;padding:9px;font-weight:700;'
                 f'background:#f0f4ff;font-size:12px">{label}</td>{cells}</tr>')
    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif">
  <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:10px 16px;border-radius:8px 8px 0 0">
    <b>📅 AI &amp; DS — {title}</b>
  </div>
  <div style="overflow-x:auto;border:1px solid #ddd;border-top:none;border-radius:0 0 8px 8px">
    <table style="width:100%;border-collapse:collapse">
      <thead>{header}</thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <p style="font-size:11px;color:#888;margin-top:5px;font-family:Segoe UI,sans-serif">
    Ask "2nd year Section A timetable" or "1st year Section B Monday" for full details.
  </p>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Timetable query parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_timetable_query(qa: QueryAnalysis) -> Dict:
    q = qa.original.lower().strip()
    result = {"section": None, "day": None, "subject": None, "year": None,
              "overview": False}

    # Check for year-wise overview request immediately
    overview_kw = {"all timetable","all timetables","year wise","yearwise",
                   "year-wise","all year","all years","timetable overview",
                   "timetable index","all sections timetable","show timetables"}
    if any(kw in q for kw in overview_kw):
        result["overview"] = True
        return result

    # 1. Advanced Year and Section extraction via regular expression cascades
    year = None
    sec_letter = None

    # Pattern A: aids style, e.g. "4-aids-a", "2_aids_b", "1 aids c"
    m_aids = re.search(r'\b([1-4])[-_\s]*aids[-_\s]*([a-m])\b', q)
    if m_aids:
        year = int(m_aids.group(1))
        sec_letter = m_aids.group(2).upper()
    else:
        # Pattern B: digit style, e.g. "4yr sec a", "4_yr_sec_a", "4yr_sec_a", "4th year section a", "2b", "2-b"
        m_dig = re.search(r'\b([1-4])(?:st|nd|rd|th)?[-_\s]*(?:yr|year)?[-_\s]*(?:sec|section|sect)?[-_\s]*([a-m])\b', q)
        if m_dig:
            year = int(m_dig.group(1))
            sec_letter = m_dig.group(2).upper()
        else:
            # Pattern C: word style, e.g. "fourth year section a", "second yr b", "third aids c"
            m_word = re.search(r'\b(first|second|third|fourth|one|two|three|four)[-_\s]*(?:yr|year|aids)?[-_\s]*(?:sec|section|sect)?[-_\s]*([a-m])\b', q)
            if m_word:
                word_map = {"first": 1, "one": 1, "second": 2, "two": 2, "third": 3, "three": 3, "fourth": 4, "four": 4}
                year = word_map[m_word.group(1)]
                sec_letter = m_word.group(2).upper()
            else:
                # Pattern D: roman style, e.g. "iv yr sec a", "ii year b", "iii aids a"
                m_roman = re.search(r'\b(i|ii|iii|iv)[-_\s]*(?:yr|year|aids)?[-_\s]*(?:sec|section|sect)?[-_\s]*([a-m])\b', q)
                if m_roman:
                    roman_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
                    year = roman_map[m_roman.group(1)]
                    sec_letter = m_roman.group(2).upper()

    # Fallback to separate year/section detection if no unified pattern matched
    if year is None:
        year_map = {
            "1st year": 1, "first year": 1, "1 year": 1, "year 1": 1, "i year": 1,
            "2nd year": 2, "second year": 2, "2 year": 2, "year 2": 2, "ii year": 2,
            "3rd year": 3, "third year": 3, "3 year": 3, "year 3": 3, "iii year": 3,
            "4th year": 4, "fourth year": 4, "4 year": 4, "year 4": 4,
        }
        for phrase, yr in year_map.items():
            if phrase in q:
                year = yr
                break
        if year is None:
            m = re.search(r'\b([1-4])\s*(?:st|nd|rd|th)?\s*year\b', q)
            if m:
                year = int(m.group(1))

    if sec_letter is None:
        m_sec = re.search(r'\b(?:section|sec|sect)[-_\s]*([a-m])\b', q)
        if m_sec:
            sec_letter = m_sec.group(1).upper()

    result["year"] = year

    # Build full section key from year + letter
    if sec_letter:
        if year == 4:
            result["section"] = f"4th_Year_Section_{sec_letter}"
        elif year == 3:
            result["section"] = f"3rd_Year_Section_{sec_letter}"
        elif year == 2:
            result["section"] = f"2nd_Year_Section_{sec_letter}"
        else:
            result["section"] = f"1st_Year_Section_{sec_letter}"

    # ── Day detection ─────────────────────────────────────────────────────
    for day in DAYS_ORDER:
        if day.lower() in q:
            result["day"] = day
            break

    # ── Subject detection (only when no section) ──────────────────────────
    if result["section"] is None:
        subject_aliases = [
            # 1st year
            ("linear algebra","LAC"),("calculus","LAC"),(" lac ","LAC"),
            ("engineering physics","EP"),(" ep ","EP"),("physics lab","EP-LAB"),
            ("ep-lab","EP-LAB"),("ep lab","EP-LAB"),
            ("basic electrical","BEEE"),("beee","BEEE"),("electrical","BEEE"),
            ("cp lab","CP LAB"),("programming lab","CP LAB"),
            ("eee workshop","EEE WS"),("eee ws","EEE WS"),
            ("it workshop","IT WS"),("it ws","IT WS"),
            ("engineering graphics","ENGINEERING GRAPHICS"),("graphics","ENGINEERING GRAPHICS"),
            ("introduction to programming","INTRODUCTION TO PROGRAMMING"),
            ("ngcs","NGCS"),(" nss ","NGCS"),(" ncc ","NGCS"),
            # 2nd year
            ("artificial intelligence"," AI "),(" ai ","AI"),
            ("introduction to data science","IDS"),(" ids ","IDS"),
            ("digital logic","DL_CO"),("dl_co","DL_CO"),("dl co","DL_CO"),
            ("computer organization","DL_CO"),
            ("business environment","BE"),
            ("statistical methods","SMDS"),("smds","SMDS"),
            ("design thinking","DTI LAB"),("dti lab","DTI LAB"),("dti","DTI LAB"),
            ("ids lab","IDS LAB"),("data science lab","IDS LAB"),
            ("ai lab","AI LAB"),
            ("full stack","FULL STACK DEVELOPMENT LAB"),("fullstack","FULL STACK DEVELOPMENT LAB"),
            # 3rd year
            ("social network","SNA"),(" sna ","SNA"),
            ("big data analytics","BDA"),(" bda ","BDA"),
            ("ai for finance","AIF"),(" aif ","AIF"),
            ("natural language processing","NLP"),(" nlp ","NLP"),
            ("deep learning","DL"),(" dl ","DL"),
            ("academic writing","AWPS"),("awps","AWPS"),("public speaking","AWPS"),
            ("technical paper","TPW.IPR"),("tpw","TPW.IPR"),("ipr","TPW.IPR"),
            ("dl lab","DL LAB"),("deep learning lab","DL LAB"),
            ("bd and dv lab","BD and DV LAB"),("data visualization lab","BD and DV LAB"),
            ("soft skills","SOFT SKILLS LAB"),("workshop","WORKSHOP"),
        ]
        padded = f" {q} "
        for alias, code in subject_aliases:
            if alias in padded:
                result["subject"] = code.strip()
                break

    return result


def build_yearwise_timetable_overview() -> str:
    """
    Mobile-first year-wise timetable index.
    Each year = one row with a coloured label + section buttons side by side.
    No horizontal scroll — fits any screen width.
    """
    tt = _TT_DATA.get("timetable", {})

    YEAR_CONFIG = [
        {
            "label": "1st Year",
            "short": "1st",
            "grad":  "linear-gradient(135deg,#1a237e,#3949ab)",
            "light": "#e8eaf6",
            "text":  "#1a237e",
            "sections": ["A","B","C","D"],
            "key_fn": lambda s: f"1st_Year_Section_{s}",
        },
        {
            "label": "2nd Year",
            "short": "2nd",
            "grad":  "linear-gradient(135deg,#1b5e20,#388e3c)",
            "light": "#e8f5e9",
            "text":  "#1b5e20",
            "sections": ["A","B"],
            "key_fn": lambda s: f"2nd_Year_Section_{s}",
        },
        {
            "label": "3rd Year",
            "short": "3rd",
            "grad":  "linear-gradient(135deg,#bf360c,#e64a19)",
            "light": "#fbe9e7",
            "text":  "#bf360c",
            "sections": ["A","B"],
            "key_fn": lambda s: f"3rd_Year_Section_{s}",
        },
        {
            "label": "4th Year",
            "short": "4th",
            "grad":  "linear-gradient(135deg,#4a148c,#7b1fa2)",
            "light": "#f3e5f5",
            "text":  "#4a148c",
            "sections": ["A","B"],
            "key_fn": lambda s: f"4th_Year_Section_{s}",
        },
    ]

    ALL_SECTIONS = ["A","B","C","D"]

    rows_html = ""
    for yr in YEAR_CONFIG:
        avail  = yr["sections"]
        grad   = yr["grad"]
        light  = yr["light"]
        color  = yr["text"]
        key_fn = yr["key_fn"]
        label  = yr["label"]
        short  = yr["short"]

        # Section buttons
        btn_cells = ""
        for sec in ALL_SECTIONS:
            key      = key_fn(sec)
            has_data = sec in avail and key in tt and bool(tt[key])

            if has_data:
                btn_cells += f"""
<div style="flex:1;min-width:0;padding:3px">
  <button onclick="quickSend('{label} Section {sec} timetable')"
    style="width:100%;background:{grad};color:#fff;border:none;
           border-radius:8px;padding:10px 4px;font-size:13px;
           font-weight:700;cursor:pointer;line-height:1.3;
           box-shadow:0 2px 6px rgba(0,0,0,.18);
           transition:opacity .15s,transform .1s;
           display:flex;flex-direction:column;align-items:center;gap:2px"
    onmouseover="this.style.opacity='.82'"
    onmouseout="this.style.opacity='1'"
    ontouchstart="this.style.opacity='.75'"
    ontouchend="this.style.opacity='1'">
    <span style="font-size:15px">Sec {sec}</span>
    <span style="font-size:9px;opacity:.85;font-weight:500">View →</span>
  </button>
</div>"""
            else:
                btn_cells += f"""
<div style="flex:1;min-width:0;padding:3px">
  <div style="width:100%;background:#f0f0f0;border-radius:8px;
              padding:10px 4px;text-align:center;
              font-size:13px;color:#ccc;font-weight:600;
              box-sizing:border-box">
    Sec {sec}
  </div>
</div>"""

        rows_html += f"""
<div style="display:flex;align-items:stretch;gap:0;
            border-bottom:2px solid #fff;background:#fff">

  <!-- Year label -->
  <div style="background:{grad};color:#fff;
              min-width:52px;max-width:52px;width:52px;
              display:flex;align-items:center;justify-content:center;
              flex-direction:column;gap:2px;padding:10px 4px;
              border-radius:0;flex-shrink:0">
    <span style="font-size:16px;font-weight:800;line-height:1">{short}</span>
    <span style="font-size:9px;font-weight:500;opacity:.85">Year</span>
  </div>

  <!-- Section buttons -->
  <div style="flex:1;display:flex;align-items:center;
              padding:6px 6px;background:{light};gap:0">
    {btn_cells}
  </div>

</div>"""

    return f"""
<div style="margin:6px 0;font-family:'Segoe UI',Arial,sans-serif;
            max-width:480px;width:100%;box-sizing:border-box">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#2d3a6b,#4a5568);color:#fff;
              padding:10px 14px;border-radius:10px 10px 0 0;
              display:flex;justify-content:space-between;align-items:center">
    <span style="font-size:14px;font-weight:700">📅 Timetable — All Years</span>
    <span style="font-size:10px;opacity:.8">AI &amp; DS Dept</span>
  </div>

  <!-- Column headers -->
  <div style="display:flex;background:#2d3a6b;border-left:none">
    <div style="min-width:52px;max-width:52px;width:52px;
                padding:6px 4px;text-align:center;
                font-size:10px;font-weight:700;color:#aab;flex-shrink:0">
      Year
    </div>
    <div style="flex:1;display:flex">
      {''.join(f'<div style="flex:1;text-align:center;padding:6px 2px;font-size:11px;font-weight:700;color:#fff">Sec {s}</div>' for s in ALL_SECTIONS)}
    </div>
  </div>

  <!-- Year rows -->
  <div style="border:1px solid #ddd;border-top:none;
              border-radius:0 0 10px 10px;overflow:hidden">
    {rows_html}
  </div>

  <!-- Tip -->
  <div style="margin-top:8px;padding:8px 12px;background:#f0f4ff;
              border-radius:8px;font-size:11px;color:#555;text-align:center">
    💡 Tap a section button to view its full weekly timetable
  </div>

</div>"""


def handle_timetable_query(qa: QueryAnalysis) -> str:
    parsed  = parse_timetable_query(qa)
    section = parsed["section"]
    day     = parsed["day"]
    subject = parsed["subject"]
    year    = parsed["year"]
    q       = qa.original.lower().strip()

    # Year-wise overview — triggered when user asks for all timetables / year overview
    overview_triggers = {
        "all timetable", "all timetables", "year wise", "yearwise",
        "year-wise", "all year", "all years", "show all timetable",
        "timetable overview", "timetable index", "all sections timetable",
        "show timetables", "list timetable", "timetable list",
    }
    if parsed.get("overview") or any(t in q for t in overview_triggers):
        return build_yearwise_timetable_overview()

    # If only year given (no section) → show overview filtered to that year
    if year and not section and not subject:
        return build_yearwise_timetable_overview()

    if section and day:   return build_day_table(section, day)
    if section:           return build_section_week_table(section)
    if subject:           return build_subject_table(subject)


# ─────────────────────────────────────────────────────────────────────────────
# Faculty Timetable & Leisure Hour functions
# ─────────────────────────────────────────────────────────────────────────────

# Slot time labels for faculty data (aids_faculty_data.json format)
_FAC_SLOT_ORDER = [
    "9:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-1:00",
    "1:00-2:00", "2:00-3:00", "3:00-4:00", "4:00-5:00"
]
_FAC_LUNCH_VALS = {"L", "U", "N", "C", "H", "BREAK", "LUNCH", "l", "u", "n", "c", "h", "break", "lunch"}
_FAC_DAYS_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
_FAC_DAY_LABELS = {
    "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
    "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday"
}


def build_faculty_timetable_html(faculty: Dict) -> str:
    """Build an HTML weekly timetable card from a faculty record's 'timetable' dict."""
    name = faculty.get("name", "Faculty")
    desig = faculty.get("designation", "")
    tt = faculty.get("timetable", {})
    if not tt:
        return (f'<p style="font-family:Segoe UI,sans-serif;color:#555">'
                f'Timetable information is not available for {name}.</p>')

    # Build table header
    th_style = ('style="border:1px solid #bbb;padding:8px 10px;text-align:center;'
                'font-size:11.5px;font-weight:700;background:#2d3a6b;color:#fff;white-space:nowrap"')
    slot_headers = "".join(
        f'<th {th_style}>{s.replace(":00-", "–").replace(":00", "")}</th>'
        for s in _FAC_SLOT_ORDER
    )
    header_row = f'<tr><th {th_style}>Day</th>{slot_headers}</tr>'

    # Build day rows
    rows = ""
    day_th = ('style="border:1px solid #bbb;padding:8px 10px;font-weight:700;'
              'font-size:12px;background:#fce4ec;color:#880e4f;'
              'white-space:nowrap;text-align:center;min-width:52px"')
    lunch_letters = {"Mon": "L", "Tue": "U", "Wed": "N", "Thu": "C", "Fri": "H", "Sat": "BREAK"}

    for day_key in _FAC_DAYS_ORDER:
        day_data = tt.get(day_key, {})
        full_day = _FAC_DAY_LABELS.get(day_key, day_key)
        cells = f'<td {day_th}>{full_day[:3]}</td>'
        for slot in _FAC_SLOT_ORDER:
            val = day_data.get(slot, "")
            if not val or val.strip() in _FAC_LUNCH_VALS or val.strip() == "":
                if "12:00-1:00" in slot:
                    lv = lunch_letters.get(day_key, "L")
                    cells += (f'<td style="border:1px solid #bbb;padding:6px 8px;text-align:center;'
                              f'background:#f0f0f0;width:52px;min-width:52px">'
                              f'<span style="font-weight:900;font-size:15px;letter-spacing:1px;'
                              f'color:#666;font-style:italic">{lv}</span></td>')
                elif val.strip() in _FAC_LUNCH_VALS:
                    cells += (f'<td style="border:1px solid #bbb;padding:6px 8px;text-align:center;'
                              f'background:#f0f0f0;"><span style="font-weight:700;color:#888">{val.strip()}</span></td>')
                else:
                    cells += (f'<td style="border:1px solid #bbb;padding:6px 8px;text-align:center;'
                              f'background:#fff;color:#ccc">—</td>')
            else:
                v = val.strip()
                # Parse "SUBJECT (...)" or plain
                m = re.match(r'^(.+?)\s*\((.+)\)$', v)
                subj = m.group(1).strip() if m else v
                extra = m.group(2).strip() if m else ""
                # Colour by lab
                if _is_lab(subj):
                    bg, fg = "#ffe066", "#5a4000"
                else:
                    bg, fg = "#fff", "#1a1a2e"
                    for code, colour in _SUBJ_COLORS.items():
                        if code in subj:
                            bg = colour
                            fg = _SUBJ_TEXT.get(code, "#1a1a2e")
                            break
                extra_html = (f'<div style="font-size:9.5px;color:{fg};opacity:.8;margin-top:2px">'
                              f'{extra}</div>') if extra else ""
                cells += (f'<td style="border:1px solid #bbb;padding:6px 8px;text-align:center;'
                          f'vertical-align:middle;font-size:11px;background:{bg};color:{fg};font-weight:700">'
                          f'<div style="font-size:11px">{subj}</div>{extra_html}</td>')
        rows += f'<tr>{cells}</tr>'

    desig_badge = (f'<span style="background:rgba(139,92,246,.12);color:#7c3aed;font-size:11px;'
                   f'font-weight:600;padding:2px 8px;border-radius:8px;margin-left:10px">'
                   f'{desig}</span>') if desig else ""
    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',Arial,sans-serif" class="timetable-card-container">
  <div style="background:linear-gradient(135deg,#2d3a6b,#4a5568);color:#fff;padding:12px 18px;border-radius:8px 8px 0 0;display:flex;align-items:center">
    <b style="font-size:14px">📅 {name}</b>{desig_badge}
    <span style="margin-left:8px;font-size:11.5px;opacity:.8">— Weekly Teaching Timetable</span>
  </div>
  <div style="overflow-x:auto;border:1px solid #bbb;border-top:none;border-radius:0 0 8px 8px;background:#fff">
    <table class="timetable-table" style="width:100%;border-collapse:collapse">
      <thead>{header_row}</thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <p style="font-size:11px;color:#888;margin-top:5px;font-family:Segoe UI,sans-serif">
    — = Free period &nbsp;|&nbsp; L/U/N/C/H = Lunch Break
  </p>
</div>"""


def build_all_faculty_schedule_list_html() -> str:
    """Build a beautiful HTML card listing all faculty members with buttons to view their timetable or free hours."""
    rows = ""
    for f in _FACULTY_DATA:
        name = f.get("name", "")
        desig = f.get("designation", "")
        # Add badge styling
        desig_badge = f'<span style="font-size:11px;color:#7c3aed;background:rgba(139,92,246,.1);padding:2px 6px;border-radius:4px">{desig}</span>' if desig else ""
        
        # HTML for each row
        rows += f"""
        <tr style="border-bottom: 1px solid #f0f0f0;">
          <td style="padding: 10px 12px; font-weight: 600; color: #1a1a2e; font-size: 13.5px; text-align: left;">
            {name} <div style="margin-top:2px">{desig_badge}</div>
          </td>
          <td style="padding: 10px 12px; text-align: right; white-space: nowrap;">
            <button onclick="quickSend('timetable of {name}')" onmouseover="this.style.opacity=0.85" onmouseout="this.style.opacity=1.0" style="background:#2d3a6b; color:#fff; border:none; padding:6px 11px; border-radius:6px; font-size:12px; font-weight:600; cursor:pointer; margin-right:4px; transition: opacity 0.2s;">📅 Timetable</button>
            <button onclick="quickSend('leisure hours of {name}')" onmouseover="this.style.opacity=0.85" onmouseout="this.style.opacity=1.0" style="background:#7c3aed; color:#fff; border:none; padding:6px 11px; border-radius:6px; font-size:12px; font-weight:600; cursor:pointer; transition: opacity 0.2s;">🕐 Free Hours</button>
          </td>
        </tr>
        """

    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',Arial,sans-serif" class="timetable-card-container">
  <div style="background:linear-gradient(135deg,#2d3a6b,#4a5568);color:#fff;padding:12px 18px;border-radius:8px 8px 0 0;display:flex;align-items:center">
    <b style="font-size:14px">👥 Faculty Schedule Directory</b>
  </div>
  <div style="border:1px solid #bbb;border-top:none;border-radius:0 0 8px 8px;background:#fff;max-height:350px;overflow-y:auto">
    <table style="width:100%;border-collapse:collapse;margin:0">
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>
</div>
"""


def render_faculty_table(faculty_list: list) -> str:
    """Render the canonical 3-column faculty overview table (Faculty name | Timetable | Leisure).
    Works for a single faculty entry (single-row) or the full department roster (all rows).
    Each row's buttons send quickSend messages that trigger the exact-phrase detail handlers.
    """
    header = (
        '<div style="margin:8px 0;font-family:\'Segoe UI\',Arial,sans-serif" class="timetable-card-container">'
        '<div style="background:linear-gradient(135deg,#2d3a6b,#4a5568);color:#fff;padding:12px 18px;'
        'border-radius:8px 8px 0 0;display:flex;align-items:center">'
        '<b style="font-size:14px">📋 Faculty Schedule Overview</b>'
        '</div>'
        '<div style="border:1px solid #bbb;border-top:none;border-radius:0 0 8px 8px;'
        'background:#fff;overflow-x:auto">'
        '<table style="width:100%;border-collapse:collapse;margin:0">'
        '<thead>'
        '<tr style="background:#f4f6fb">'
        '<th style="padding:10px 14px;text-align:left;font-size:12px;color:#2d3a6b;'
        'border-bottom:2px solid #e0e4f0;font-weight:700">Faculty Name</th>'
        '<th style="padding:10px 14px;text-align:center;font-size:12px;color:#2d3a6b;'
        'border-bottom:2px solid #e0e4f0;font-weight:700">Time Table View</th>'
        '<th style="padding:10px 14px;text-align:center;font-size:12px;color:#2d3a6b;'
        'border-bottom:2px solid #e0e4f0;font-weight:700">Leisure Periods View</th>'
        '</tr>'
        '</thead><tbody>'
    )
    rows = ""
    for f in faculty_list:
        name = f.get("name", "")
        desig = f.get("designation", "")
        desig_badge = (
            f'<div style="font-size:11px;color:#7c3aed;margin-top:2px">'
            f'{desig}</div>'
        ) if desig else ""
        tt_btn = (
            f'<button onclick="quickSend(\'timetable of {name}\')" '
            f'onmouseover="this.style.opacity=0.85" onmouseout="this.style.opacity=1.0" '
            f'style="background:#2d3a6b;color:#fff;border:none;padding:7px 13px;'
            f'border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;'
            f'transition:opacity 0.2s">📅 View Timetable</button>'
        )
        lei_btn = (
            f'<button onclick="quickSend(\'leisure hours of {name}\')" '
            f'onmouseover="this.style.opacity=0.85" onmouseout="this.style.opacity=1.0" '
            f'style="background:#7c3aed;color:#fff;border:none;padding:7px 13px;'
            f'border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;'
            f'transition:opacity 0.2s">🕐 View Leisure</button>'
        )
        rows += (
            f'<tr style="border-bottom:1px solid #f0f0f0">'
            f'<td style="padding:10px 14px;font-weight:600;color:#1a1a2e;font-size:13px">'
            f'{name}{desig_badge}</td>'
            f'<td style="padding:10px 14px;text-align:center">{tt_btn}</td>'
            f'<td style="padding:10px 14px;text-align:center">{lei_btn}</td>'
            f'</tr>'
        )
    footer = '</tbody></table></div></div>'
    return header + rows + footer


def build_faculty_leisure_html(faculty: Dict) -> str:
    """Identify and render leisure (free) hours for a faculty member."""
    name = faculty.get("name", "Faculty")
    desig = faculty.get("designation", "")
    tt = faculty.get("timetable", {})
    if not tt:
        return (f'<p style="font-family:Segoe UI,sans-serif;color:#555">'
                f'Timetable information is not available for {name}.</p>')

    # Teaching slots (exclude 12:00-1:00 lunch)
    work_slots = [s for s in _FAC_SLOT_ORDER if "12:00-1:00" not in s]

    free_by_day: Dict[str, List[str]] = {}
    for day_key in _FAC_DAYS_ORDER:
        day_data = tt.get(day_key, {})
        free = []
        for slot in work_slots:
            val = day_data.get(slot, "").strip()
            if not val or val in _FAC_LUNCH_VALS:
                # This slot is free
                slot_label = slot.replace(":00-", "–").replace(":00", "")
                free.append(slot_label)
        if free:
            free_by_day[_FAC_DAY_LABELS.get(day_key, day_key)] = free

    if not free_by_day:
        return (f'<div style="font-family:Segoe UI,sans-serif;padding:14px 18px;'
                f'background:#fff3cd;border-radius:10px;border:1px solid #ffc107">'
                f'<b>⚠️ {name}</b> appears to have no free periods in the timetable.</div>')

    # Render card
    rows_html = ""
    for day, slots in free_by_day.items():
        slot_pills = "".join(
            f'<span style="background:rgba(139,92,246,.12);color:#7c3aed;font-size:11.5px;'
            f'font-weight:600;padding:3px 9px;border-radius:12px;margin:2px;display:inline-block">'
            f'{s}</span>' for s in slots
        )
        rows_html += (f'<tr>'
                      f'<td style="border:1px solid #eee;padding:10px 14px;font-weight:700;'
                      f'background:#f8f9ff;color:#2d3a6b;font-size:13px;white-space:nowrap;min-width:90px">{day}</td>'
                      f'<td style="border:1px solid #eee;padding:8px 12px">{slot_pills}</td>'
                      f'</tr>')

    total = sum(len(v) for v in free_by_day.values())
    desig_badge = (f'<span style="background:rgba(255,255,255,.2);font-size:11px;'
                   f'padding:2px 8px;border-radius:8px;margin-left:10px">{desig}</span>') if desig else ""
    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',Arial,sans-serif">
  <div style="background:linear-gradient(135deg,#7c3aed,#6d28d9);color:#fff;padding:12px 18px;border-radius:8px 8px 0 0;display:flex;align-items:center">
    <b style="font-size:14px">🕐 Leisure / Free Hours — {name}</b>{desig_badge}
    <span style="margin-left:auto;font-size:11.5px;opacity:.85">{total} free slot(s) per week</span>
  </div>
  <div style="border:1px solid #ddd;border-top:none;border-radius:0 0 8px 8px;background:#fff;overflow:hidden">
    <table style="width:100%;border-collapse:collapse">
      <thead>
        <tr style="background:#f0f0f0">
          <th style="border:1px solid #eee;padding:9px 14px;text-align:left;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.04em">Day</th>
          <th style="border:1px solid #eee;padding:9px 14px;text-align:left;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.04em">Free Periods</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p style="font-size:11px;color:#888;padding:8px 14px;border-top:1px solid #eee;margin:0">
      🍽️ Lunch break (12:00–1:00) is excluded from free periods.
    </p>
  </div>
</div>"""


def is_faculty_timetable_query(query: str) -> bool:
    """Return True if query is asking for a specific faculty's timetable."""
    q = query.lower()
    tt_kw = {"timetable", "time table", "schedule", "classes", "teaching", "class timing"}
    fac_kw = {"faculty", "teacher", "professor", "lecturer", "sir", "mam", "madam", "dr", "prof"}
    has_tt = any(k in q for k in tt_kw)
    has_fac_kw = any(k in q for k in fac_kw)
    # Also match patterns like "timetable of dr xyz" or "classes of [name]"
    if has_tt:
        return True
    return False


def is_faculty_leisure_query(query: str) -> bool:
    """Return True if query is asking for faculty free/leisure periods."""
    q = query.lower()
    leisure_kw = {"free", "leisure", "free period", "free hour", "free time",
                  "not busy", "available", "when is", "when free", "break"}
    return any(k in q for k in leisure_kw)



# ─────────────────────────────────────────────────────────────────────────────
# Answer synthesis — RAG context → structured HTML
# ─────────────────────────────────────────────────────────────────────────────
def _fuzzy_name_match(query_name: str, faculty_name: str) -> float:
    """
    Score how well query_name matches faculty_name.
    Returns 0.0–1.0. Uses token overlap + substring checks.
    """
    from difflib import SequenceMatcher
    q = query_name.lower().strip()
    f = faculty_name.lower().strip()

    # Exact full match
    if q == f:
        return 1.0

    # Full substring
    if q in f or f in q:
        return 0.95

    # Token overlap — split both into parts, count matches
    q_parts = [p for p in re.split(r'\s+', q) if len(p) > 1]
    f_parts = [p for p in re.split(r'\s+', f) if len(p) > 1]

    if not q_parts:
        return 0.0

    matched = sum(1 for qp in q_parts if any(qp in fp or fp in qp for fp in f_parts))
    token_score = matched / len(q_parts)

    # Sequence similarity as tiebreaker
    seq_score = SequenceMatcher(None, q, f).ratio()

    return max(token_score, seq_score)


def _normalise_name(s: str) -> str:
    """Lowercase, strip titles, punctuation, extra spaces."""
    s = s.lower()
    for title in ("dr.", "dr ", "prof.", "prof ", "mr.", "mr ", "mrs.", "mrs ", "ms.", "ms "):
        s = s.replace(title, "")
    s = re.sub(r'[^a-z\s]', ' ', s)
    return " ".join(s.split())

def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two normalised name strings."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def _find_faculty_by_name(query: str) -> Optional[Dict]:
    """
    Advanced multi-strategy faculty name lookup:
      1. Exact normalised match             → score 1.0
      2. Token Jaccard overlap              → score weight
      3. Fuzzy match token_sort_ratio       → score weight
      4. Single token close ratio check
    Returns best match with score >= 0.60, else None.
    """
    # Strip common question prefixes
    clean = re.sub(
        r'\b(who is|tell me about|about|details of|detail of|info on|info about|'
        r'information about|information of|show me|find|search|get|what does|'
        r'what is the specialization of|qualification of|designation of|'
        r'phone of|email of|contact of|joining date of|'
        r'timetable of|time table of|schedule of|classes of|leisure hours of|'
        r'free periods of|free hours of|when is|free|timetable|schedule|'
        r'time table|classes|leisure|free period|free hour|free time|profile of|profile|sir|madam|madam|teacher|professor|lecturer|mrs|mr|dr|prof)\b',
        '', query, flags=re.IGNORECASE
    ).strip()

    q_norm = _normalise_name(clean)
    if not q_norm:
        return None

    best_score = 0.0
    best_match = None

    for f in _FACULTY_DATA:
        fname = f.get("name", "")
        fn = _normalise_name(fname)

        # 1. Exact match
        if q_norm == fn:
            return f

        # 2. Token Jaccard overlap
        overlap = _token_overlap(q_norm, fn)
        score = overlap * 0.90

        # 3. Fuzzy matching using rapidfuzz token_sort_ratio
        fuzzy_score = fuzz.token_sort_ratio(q_norm, fn) / 100.0
        score = max(score, fuzzy_score)

        # 4. Check if any query token is a substring of any faculty name token, fuzzy matched
        q_tokens = q_norm.split()
        fn_tokens = fn.split()
        for qt in q_tokens:
            if len(qt) >= 3:
                for ft in fn_tokens:
                    if len(ft) >= 3:
                        token_ratio = fuzz.ratio(qt, ft) / 100.0
                        if token_ratio >= 0.75:
                            score = max(score, token_ratio * 0.85)

        if score > best_score:
            best_score = score
            best_match = f

    return best_match if best_score >= 0.60 else None


def synthesize_answer(qa: QueryAnalysis, docs_with_scores: List[Tuple[Dict,float]], intent: str) -> Optional[str]:
    if not docs_with_scores or docs_with_scores[0][1] < CONFIDENCE_THRESHOLD:
        return None
    q = qa.original.lower()

    # ── Faculty intent ────────────────────────────────────────────────────
    if intent == "faculty":
        # 1. HOD / head query
        if any(w in q for w in ["hod","head of department","head of dept","head of the"]):
            hod = next((f for f in _FACULTY_DATA if "head" in f.get("designation","").lower()), None)
            if hod: return build_faculty_card(hod)

        # 2. Specific person — try NER extracted names first
        if qa.person_names:
            for pname in qa.person_names:
                match = _find_faculty_by_name(pname)
                if match:
                    return build_faculty_card(match)

        # 3. Specific person — fuzzy match on the whole query
        #    Only do this when query looks like a name lookup (not "list all")
        list_signals = {"list","all","show","every","members","staff","teachers","lecturers","how many"}
        is_list_query = bool(set(qa.tokens) & list_signals) or any(
            w in q for w in ["list","all faculty","faculty members","show faculty",
                             "who are","how many","all staff"]
        )

        if not is_list_query:
            name_match = _find_faculty_by_name(qa.original)
            if name_match:
                return build_faculty_card(name_match)

            # Also try individual tokens as partial names
            for tok in qa.tokens:
                if len(tok) > 3:
                    for f in _FACULTY_DATA:
                        if tok in f.get("name","").lower():
                            return build_faculty_card(f)

        # 4. Specialization query
        spec_map = {
            "Machine Learning":        ["machine","learn","ml"],
            "Deep Learning":           ["deep","learn","dl"],
            "Artificial Intelligence": ["artificial","intelligence","ai"],
            "Python Programming":      ["python"],
            "Computer Networks":       ["network","cn"],
            "Software Engineering":    ["software","engineer"],
            "DBMS / Database":         ["dbms","database"],
            "Computer Science":        ["computer","science","cs"],
        }
        for spec_label, lemmas in spec_map.items():
            if any(lm in qa.tokens for lm in lemmas) or spec_label.lower() in q:
                matched = [f for f in _FACULTY_DATA
                           if any(lm in f.get("specialization","").lower() for lm in lemmas)]
                if matched: return build_specialization_table(spec_label, matched)

        # 5. Explicit list request → full table
        if is_list_query:
            return build_faculty_list_table()

        # 6. Fallback: full list only if nothing else matched
        return build_faculty_list_table()

    # ── Timetable (safety fallback) ───────────────────────────────────────
    if intent == "timetable":
        return handle_timetable_query(qa)

    # ── Services / general — structured info card ─────────────────────────
    top_texts = [d["text"] for d, s in docs_with_scores[:4] if s >= CONFIDENCE_THRESHOLD]
    if not top_texts:
        return None

    seen, rows = set(), []
    for text in top_texts:
        for sentence in re.split(r"[.\n]", text):
            s = sentence.strip()
            if s and s not in seen and len(s) > 12:
                seen.add(s)
                rows.append(s)

    if not rows:
        return None

    # Render as a modern card with bullet points or paragraph
    if len(rows) == 1:
        # Single sentence → clean paragraph
        body_html = f'<p style="padding:12px 16px;font-size:13.5px;color:#333;line-height:1.7;margin:0">{rows[0]}</p>'
    else:
        # Multiple points → bullet list
        items = "".join(
            f'<div style="display:flex;align-items:flex-start;gap:8px;padding:6px 0">'
            f'<span style="color:#8b5cf6;font-size:11px;margin-top:3px;flex-shrink:0">✦</span>'
            f'<span style="font-size:13px;color:#333;line-height:1.6">{r}</span></div>'
            for r in rows[:6]
        )
        body_html = f'<div style="padding:12px 16px">{items}</div>'
    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif">
  <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:10px 16px;border-radius:10px 10px 0 0">
    <b>ℹ️ Information</b>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;border-radius:0 0 10px 10px;overflow:hidden">
    {body_html}
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Structured HTML for static responses
# ─────────────────────────────────────────────────────────────────────────────
def _info_card(title: str, rows: List[Tuple[str,str]]) -> str:
    """Render a modern key/value card with flex layout."""
    items_html = "".join(
        f'<div style="padding:10px 16px;{"background:#f8f9ff;" if i%2==0 else ""}'
        f'{"border-top:1px solid #eee;" if i > 0 else ""}'
        f'display:flex;flex-direction:column;gap:2px">'
        f'<span style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:.04em">{k}</span>'
        f'<span style="font-size:13px;color:#333;line-height:1.5">{v}</span></div>'
        for i,(k,v) in enumerate(rows)
    )
    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif">
  <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:12px 16px;border-radius:10px 10px 0 0">
    <b>{title}</b>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;border-radius:0 0 10px 10px;overflow:hidden">
    {items_html}
  </div>
</div>"""


def _fallback_answer_card(doc: Dict) -> str:
    """Generic card for displaying a hybrid-search match when no specific
    rule-based handler matched the query."""
    content = doc.get("display_text") or doc.get("text", "")
    doc_type = doc.get("type", "information").replace("_", " ").title()
    return _info_card(f"ℹ️ {doc_type} (Related Info)", [("Details", content)])

def _help_card() -> str:
    rows = [
        ("📅 Timetables",  '1st/2nd/3rd Year Section A/B/C/D weekly schedules — try: "all timetables"'),
        ("👥 Faculty",     'Names, designations, qualifications — try: "Who is the HOD?"'),
        ("📢 Circulars",   'Notices, fee circulars — try: "show circulars"'),
        ("💻 Services",    'Attendance, e-journals, portal login'),
        ("🏛️ College",     'About NBKRIST, NAAC, NBA, programs — try: "about nbkr"'),
        ("🤖 Department",  'AI&DS overview, subjects, labs, careers — try: "department overview"'),
        ("💼 Placements",  'Recruiters, companies — try: "placement companies"'),
        ("🛠️ Skills",      'Recommended tech stack — try: "recommended skills"'),
        ("🎉 Events",      'Techvyuha, Inspiron, Hackathons — try: "college events"'),
        ("👥 Clubs",       'IEEE, NSS, NCC, Coding Club — try: "student chapters"'),
        ("🎓 Students",    'Roll lookup, CGPA, section — try: "23KB1A3062"'),
        ("🚌 Bus Fees",    'Fee by location — try: "bus fee from Nellore"'),
    ]
    return _info_card("💡 What I can help you with", rows)


def _attendance_card() -> str:
    return _info_card("Attendance", [
        ("How to check", "To check attendance, please follow the given link below."),
        ("Attendance link", '<a href="https://t.me/NbkristQik_bot" target="_blank" rel="noopener">https://t.me/NbkristQik_bot</a>'),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Query Preprocessor — typo correction, abbreviation expansion, alias normalise
# ─────────────────────────────────────────────────────────────────────────────

from rapidfuzz import fuzz, process as _rf_process

# Known-good domain vocabulary the chatbot should recognize.
# Fuzzy matching corrects misspellings of these words automatically —
# no need to hand-write every typo variant.
_DOMAIN_VOCAB = [
    "attendance", "artificial", "intelligence", "machine", "learning",
    "database", "computer", "engineering", "python", "fullstack",
    "hostel", "library", "department", "placements", "placement",
    "recruiters", "recruiter", "facilities", "facility", "science",
    "programming", "development", "analytics", "semester", "syllabus",
    "admission", "information", "accreditation", "affiliation",
    "specialization", "timetable", "schedule", "faculty", "teacher",
    "lecturer", "professor", "circular", "circulars", "curriculum",
    "exam", "exams", "results", "marks", "grades", "cgpa", "sgpa",
    "bus", "fee", "fees", "route", "stop", "lab", "labs", "project",
    "projects", "internship", "internships", "workshop", "event",
    "events", "notice", "notification", "contact", "email", "phone",
    "qualification", "designation", "assistant", "associate",
    "principal", "dean", "chairman", "college", "institute", "hod",
    "scholarship", "scholarships", "eligibility", "counselling",
    "rank", "seat", "vacancy", "free",
]

def fuzzy_correct_word(word: str, threshold: int = 80) -> str:
    """Return the closest domain-vocab match for a misspelled word,
    or the original word if nothing is close enough / already correct."""
    if len(word) < 4 or word in _DOMAIN_VOCAB:
        return word
    match = _rf_process.extractOne(word, _DOMAIN_VOCAB, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return match[0]
    return word

def fuzzy_correct_query(text: str) -> str:
    """Run fuzzy spell-correction word-by-word over a query."""
    words = text.split()
    return " ".join(fuzzy_correct_word(w) for w in words)


_TYPO_MAP = {
    "attendence": "attendance", "attendece": "attendance",
    "artifical": "artificial", "artifcial": "artificial", "artiicial": "artificial",
    "intelligance": "intelligence", "intelligense": "intelligence",
    "machne": "machine", "machin": "machine", "mchine": "machine",
    "learnng": "learning", "lerning": "learning", "learing": "learning",
    "databse": "database", "dtabase": "database", "databas": "database",
    "compter": "computer", "computr": "computer", "copmuter": "computer",
    "engneering": "engineering", "engneer": "engineer", "engieneering": "engineering",
    "pyhtон": "python", "pyhton": "python", "pythn": "python", "pyton": "python",
    "fullstck": "fullstack", "fullstck": "full stack", "fullstac": "full stack",
    "hostal": "hostel", "hostl": "hostel", "hstel": "hostel",
    "libary": "library", "libarary": "library", "libray": "library",
    "departmnt": "department", "deparment": "department", "dept ": "department ",
    "placemnts": "placements", "placment": "placement",
    "recrutiers": "recruiters", "recrutier": "recruiter",
    "facilites": "facilities", "facilty": "facility",
    "sciense": "science", "sceince": "science",
    "prgramming": "programming", "porgramming": "programming",
    "developmnt": "development", "devlopment": "development",
    "analytcs": "analytics", "analytic": "analytics",
    "semster": "semester", "smester": "semester",
    "sylabbus": "syllabus", "sylabus": "syllabus",
    "admision": "admission", "admssion": "admission",
    "infomation": "information", "informaton": "information",
    "acreditation": "accreditation", "accrediattion": "accreditation",
    "affiliaton": "affiliation", "afiliation": "affiliation",
    "specilization": "specialization", "specalisation": "specialization",
}

_ABBR_MAP = {
    r'\bai\b':     "artificial intelligence",
    r'\bml\b':     "machine learning",
    r'\bdl\b':     "deep learning",
    r'\bnlp\b':    "natural language processing",
    r'\bcv\b':     "computer vision",
    r'\brag\b':    "retrieval augmented generation",
    r'\bmcp\b':    "model context protocol",
    r'\baws\b':    "amazon web services",
    r'\bdbms\b':   "database management systems",
    r'\bdsa\b':    "data structures algorithms",
    r'\boop\b':    "object oriented programming",
    r'\bfsd\b':    "full stack development",
    r'\bbda\b':    "big data analytics",
    r'\bdwdm\b':   "data warehousing mining",
    r'\bmlops\b':  "machine learning operations",
    r'\bai&ds\b':  "artificial intelligence data science",
    r'\baids\b':   "artificial intelligence data science",
    r'\baiml\b':   "artificial intelligence machine learning",
    r'\bcse\b':    "computer science engineering",
    r'\bece\b':    "electronics communication engineering",
    r'\beee\b':    "electrical electronics engineering",
}

_COLLEGE_ALIASES = {
    "nbkr", "nbkrist", "nbkr college", "nbkr institute",
    "n.b.k.r", "nbkr ist", "nbkr iot", "nbkr tech",
}

_DEPT_ALIAS_MAP = {
    "aids":  "artificial intelligence data science",
    "ai ds": "artificial intelligence data science",
    "ai&ds": "artificial intelligence data science",
    "aiml":  "artificial intelligence machine learning",
    "ai&ml": "artificial intelligence machine learning",
    "cse":   "computer science engineering",
    "ece":   "electronics communication engineering",
    "eee":   "electrical electronics engineering",
    "mech":  "mechanical engineering",
    "civil": "civil engineering",
}

# Casual/informal phrasing → canonical terms the router already recognizes.
# This has nothing to do with department names — purely phrasing normalization.
_QUERY_REWRITE_MAP = {
    r'\b(mam|madam|maam)\b': 'faculty',
    r'\bsir\b': 'faculty',
    r'\btime\s+table\b': 'timetable',
    r'\bfree\s*hrs?\b': 'free hours',
    r'\bfree\s*periods?\b': 'free hours',
}


def preprocess_query(query: str) -> str:
    """
    Apply typo correction + abbreviation expansion before routing.
    Preserves original casing for display but returns lowercased expanded form.
    """
    q = query.lower().strip()

    # 1. Fix typos (whole-word replacements) — hand-written known typos first
    for typo, correct in _TYPO_MAP.items():
        q = re.sub(r'\b' + re.escape(typo) + r'\b', correct, q)

    # 1b. Fuzzy-correct anything the fixed typo map missed
    q = fuzzy_correct_query(q)

    # 2. Expand department aliases
    for alias, full in _DEPT_ALIAS_MAP.items():
        q = re.sub(r'\b' + re.escape(alias) + r'\b', full, q)

    # 3. Expand abbreviations (only when standalone)
    for pattern, expansion in _ABBR_MAP.items():
        # Skip if already expanded
        if expansion not in q:
            q = re.sub(pattern, expansion, q, flags=re.IGNORECASE)

    # 4. Rewrite casual/informal phrasing to canonical router keywords
    for pattern, replacement in _QUERY_REWRITE_MAP.items():
        q = re.sub(pattern, replacement, q)

    return q.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main response function
# ─────────────────────────────────────────────────────────────────────────────
def classify_query_module(query: str, qa: QueryAnalysis) -> str:
    q = query.lower().strip()

    # ── Button-click exact phrases → skip overview, go straight to detail view ──
    # These are emitted by quickSend() from the 3-column table buttons.
    _exact_tt  = re.match(r'^timetable of (.+)$', q)
    _exact_lei = re.match(r'^leisure hours of (.+)$', q)
    _all_kw = {"all", "all faculty", "all teachers", "all staff", "all lecturers", "every", "everyone"}
    if _exact_tt:
        _tt_target = _exact_tt.group(1).strip()
        if not any(kw == _tt_target or kw in _tt_target for kw in _all_kw):
            return "faculty_timetable"
    if _exact_lei:
        _lei_target = _exact_lei.group(1).strip()
        if not any(kw == _lei_target or kw in _lei_target for kw in _all_kw):
            return "faculty_leisure"

    # ── Specific faculty name match ──
    # Check if a specific faculty is mentioned in the query
    fac = _find_faculty_by_name(q)
    if not fac and any(w in q for w in ["hod", "head of department", "head of dept"]):
        fac = next((f for f in _FACULTY_DATA if "head" in f.get("designation","").lower()), None)

    if fac:
        tt_kw = {"timetable", "time table", "schedule", "schedules", "classes", "class timing", "class timings", "timings", "timing", "working hours", "hours", "lectures"}
        leisure_kw = {"leisure", "free", "available", "not busy", "when free", "free periods", "free hours", "free time"}
        profile_kw = {"info", "information", "details", "profile", "qualification", "contact", "email", "phone", "about", "who is", "specialization", "joining date", "doj"}
        
        has_tt = any(w in q for w in tt_kw)
        has_leisure = any(w in q for w in leisure_kw)
        has_profile = any(w in q for w in profile_kw)
        
        if has_leisure:
            return "faculty_leisure"
        elif has_tt and has_profile:
            return "faculty_both"
        elif has_tt:
            return "faculty_timetable"
        else:
            return "faculty_profile"

    # ── 0a. Generic faculty schedule/leisure/timetable query (no specific name) ──
    # Use two-keyword presence so "faculty time table", "faculty free periods", etc. all match.
    _fac_actor_kw  = {"faculty", "teacher", "teachers", "lecturer", "lecturers",
                      "professor", "professors", "staff"}
    _fac_sched_kw  = {"timetable", "time table", "schedule", "schedules",
                      "timetables", "time tables", "class timing", "class timings"}
    _fac_leisure_kw = {"leisure", "free period", "free periods", "free hour",
                       "free hours", "free time", "not busy", "available"}
    _has_actor   = any(kw in q for kw in _fac_actor_kw)
    _has_sched   = any(kw in q for kw in _fac_sched_kw)
    _has_leisure = any(kw in q for kw in _fac_leisure_kw)

    # ── Direct-phrase shortcuts for "all faculty" overview ──
    _all_faculty_phrases = [
        "all faculty timetable", "all faculty time table", "all faculty schedule",
        "all teachers timetable", "all staff timetable", "timetable of all faculty",
        "schedule of all faculty", "show all faculty timetable", "show all timetable",
        "faculty timetable list", "all timetable", "full faculty schedule",
        "all faculty schedules", "show faculty timetable", "show faculty schedule",
    ]
    if any(phrase in q for phrase in _all_faculty_phrases):
        return "faculty_overview_all"

    if _has_actor and (_has_sched or _has_leisure):
        # Only promote to faculty_overview_all if NO specific faculty name is present.
        _fac_skip_gen = {"free", "when", "hour", "period", "time", "busy", "class",
                         "timetable", "schedule", "leisure", "table", "info", "show",
                         "what", "does", "teacher", "faculty", "staff", "lecturer"}
        _name_found = False
        for _f in _FACULTY_DATA:
            _fname = _f.get("name", "").lower()
            _ntoks = [p for p in re.sub(r'[^a-z\s]', ' ', _fname).split() if len(p) >= 4]
            if any(tok in q and tok not in _fac_skip_gen for tok in _ntoks):
                _name_found = True
                break
        if not _name_found:
            return "faculty_overview_all"

    # ── 0b. Query mentions a specific faculty name (any context) ──
    # Check against every faculty in the DB; if a name token ≥4 chars matches → single overview row
    _fac_skip = {"free", "when", "hour", "period", "time", "busy", "class", "timetable",
                 "schedule", "leisure", "table", "info", "show", "what", "does"}
    for f in _FACULTY_DATA:
        fname = f.get("name", "").lower()
        name_tokens = [p for p in re.sub(r'[^a-z\s]', ' ', fname).split() if len(p) >= 4]
        if any(tok in q and tok not in _fac_skip for tok in name_tokens):
            return "faculty_overview_single"

    # ── 0c. Legacy catch — any timetable/leisure keyword with name-like word ──
    if any(kw in q for kw in ["timetable", "schedule", "time table"]):
        for f in _FACULTY_DATA:
            fname = f.get("name", "").lower()
            name_tokens = [p for p in fname.split() if len(p) >= 4]
            if any(tok in q for tok in name_tokens):
                return "faculty_overview_single"

    # ── 0d. Bare faculty-actor + sched/leisure keyword that slipped past 0a ──
    # (catches edge cases where 0a was pre-empted by a name match that then didn't fire)
    if _has_actor and (_has_sched or _has_leisure):
        return "faculty_overview_all"

    # 1. Student Check
    if _ROLL_PAT.search(q) or any(w in q for w in ["student", "roll", "cgpa", "topper", "student info", "student list"]):
        timetable_kw = {"timetable","time table","schedule","class","period","timing","lecture","lab","monday","tuesday","wednesday","thursday","friday","saturday","weekly","slot","semester","sem "}
        if not any(kw in q for kw in timetable_kw):
            return "student"
        # Student + timetable → student_timetable (look up their section then show tt)
        return "student_timetable"

    # 2. Timetable Check (section-based) — skip if query contains faculty/teacher words
    _sect_tt_kw = ["timetable", "schedule", "time table", "class timing", "period",
                   "slot", "section a", "section b", "section c", "section d"]
    _no_faculty_word = not any(w in q for w in ["faculty", "teacher", "teachers",
                                                 "professor", "professors", "lecturer",
                                                 "lecturers", "staff"])
    if _no_faculty_word and any(re.search(rf'\b{re.escape(w)}s?\b', q) for w in _sect_tt_kw):
        return "timetable"

    # 3. Faculty Check
    faculty_keywords = ["faculty", "teacher", "professor", "lecturer", "staff", "designation", "specialization", "doj", "joining date", "teaches", "teaching"]
    if any(re.search(rf'\b{re.escape(w)}s?\b', q) for w in faculty_keywords) or any(re.search(rf'\b{re.escape(w)}s?\b', q) for w in ["hod", "head of department", "head of dept"]):
        return "faculty"
    skip_words = {"sir","mam","madam","prof","dr","mr","mrs","ms","faculty","teacher","lecturer","who","is","the","tell","me","show","find","get","what"}
    for word in re.sub(r'[^a-z\s]', ' ', q).split():
        if len(word) >= 3 and word not in skip_words:
            for f in _FACULTY_DATA:
                fname = f.get("name","").lower()
                if word in fname or word in fname.replace(" ",""):
                    return "faculty"

    # 4a. Circular / notice checks before fee lookup so "fee circular" is not
    # treated as a bus-fee amount query.
    if any(re.search(rf'\b{re.escape(w)}s?\b', q) for w in ["circular", "notice", "announcement", "latest notice", "fee circular"]):
        return "circular"

    # 4. Bus Fee Check
    if any(re.search(rf'\b{w}s?\b', q) for w in ["bus", "fee", "fare", "charge", "transport", "route", "stop", "boarding point"]):
        return "bus_fee"
    for route_stops in _BUS_FEES.values():
        for stop_entry in route_stops:
            stop = stop_entry.get("stop", "").lower()
            if stop and any(re.search(rf'\b{word}\b', q) for word in stop.replace("(", " ").replace(")", " ").split() if len(word) >= 4):
                return "bus_fee"

    # 5. Department Check
    department_keywords = ["department", "dept", "vision", "mission", "peo", "pso", "outcomes", "objectives", "about department", "about the department", "hod message", "message from hod"]
    def _match_dept_kw(w, text):
        if w in ["dept", "vision", "mission", "peo", "pso"]:
            return bool(re.search(rf'\b{w}s?\b', text))
        return bool(re.search(rf'\b{re.escape(w)}s?\b', text))
        
    if any(_match_dept_kw(w, q) for w in department_keywords):
        return "department"

    # 5a. Labs R23 Check — before college/general so "lab" queries are caught first
    labs_kw = [
        "lab", "labs", "laboratory", "laboratories", "workshop", "workshops",
        "practical", "practicals", "skill course", "r23 lab", "aids lab",
        "which labs", "what labs", "list of labs", "all labs",
        "cp lab", "ep lab", "ds lab", "ai lab", "ml lab", "dl lab",
        "oop lab", "java lab", "fsd", "dwdm", "bda lab", "tinkering",
        "phy lab", "chem lab", "english lab", "adsa", "dsa lab",
    ]
    if any(re.search(rf'\b{re.escape(w)}s?\b', q) for w in labs_kw):
        return "labs"

    # 5b. College Info Check
    college_kw = [
        "nbkr", "nbkrist", "college", "institute", "naac", "nba", "aicte", "ugc",
        "established", "affiliation", "jntua", "autonomous", "accreditation",
        "campus", "facilit", "hostel", "library", "placement", "recruiter",
        "program", "course", "undergraduate", "postgraduate", "btech", "b.tech",
        "career", "opportunit", "skill", "technolog", "stack", "lab", "laborator",
        "event", "activit", "techvyuha", "inspiron", "hackathon", "chapter",
        "club", "ieee", "nss", "ncc", "vision", "mission", "about college",
        "about nbkr", "intake", "ai&ds department", "aids department"
    ]
    if any(re.search(rf'\b{re.escape(w)}s?\b', q) for w in college_kw):
        return "college"

    # 7. General Check
    general_keywords = ["hostel", "admission", "admissions", "course", "courses", "library", "placement", "placements", "portal", "intranet", "e-journal", "ejournal", "assessment", "exam duties", "login"]
    if any(re.search(rf'\b{re.escape(w)}s?\b', q) for w in general_keywords):
        return "general"

    return "unknown"


def log_failed_query(query: str, module: str, score: float):
    file_path = "failed_queries.json"
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
    except Exception:
        logs = []
        
    logs.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "classified_module": module,
        "top_score": round(score, 4)
    })
    
    # Keep last 500 entries
    logs = logs[-500:]
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error logging failed query: {e}")


def web_search(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    url = "https://www.mojeek.com/search"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        r = requests.get(url, params={'q': query}, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, 'html.parser')
        results = []
        for h2 in soup.find_all('h2'):
            a = h2.find('a')
            if a:
                title = a.get_text(strip=True)
                href = a.get('href', '')
                if not href or 'mojeek.com' in href or href.startswith('/'):
                    continue
                li = h2.find_parent('li')
                snippet_el = li.find('p', class_='s') if li else None
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                results.append({
                    'title': title,
                    'url': href,
                    'snippet': snippet
                })
                if len(results) >= num_results:
                    break
        return results
    except Exception as e:
        print(f"Web search error: {e}")
        return []


def handle_web_fallback(query: str, is_college_context: bool = False) -> Optional[str]:
    search_results = []
    if is_college_context:
        # Simplify the query for Mojeek
        s = query.lower()
        s = re.sub(r'[^\w\s]', ' ', s)
        ignore_words = {
            "what", "who", "where", "when", "why", "how",
            "is", "are", "was", "were", "am", "do", "does", "did", "have", "has", "had",
            "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "about",
            "please", "tell", "me", "explain", "describe", "show", "find", "get",
            "nbkrist", "nbkr", "college", "institute", "vidyanagar"
        }
        words = s.split()
        filtered = [w for w in words if w not in ignore_words and len(w) > 1]
        key_terms = filtered[:2]
        if key_terms:
            search_query = "NBKRIST " + " ".join(key_terms)
        else:
            search_query = "NBKRIST"

        search_results = web_search(search_query, num_results=3)
        if not search_results:
            # Fallback to general search on the query
            search_results = web_search("NBKRIST", num_results=3)

    if is_college_context:
        if search_results:
            header_label = "🌐 Web Search Result"
            context_lines = []
            for i, res in enumerate(search_results):
                snippet = res['snippet'][:300]
                context_lines.append(f"Result {i+1}:\nTitle: {res['title']}\nURL: {res['url']}\nSnippet: {snippet}\n")
            context_text = "\n".join(context_lines)[:1200]

            system_prompt = (
                "You are the official AI Assistant for N.B.K.R. Institute of Science & Technology (NBKRIST). "
                "Your task is to answer the user's question using the provided web search results. "
                "If the search results contain relevant information, synthesize a helpful, comprehensive, and natural response "
                "in the style of ChatGPT. Prioritize details about NBKRIST if queried.\n"
                "IMPORTANT rules:\n"
                "1. Answer in a professional, warm, and natural tone.\n"
                "2. You MUST format your response strictly as clean HTML. Use <p>, <ul>, <li>, <strong>, and <em>. Do NOT use markdown notation like **, *, #, etc.\n"
                "3. Incorporate clickable HTML links for any sources or references using <a href=\"...\" target=\"_blank\" style=\"color:#1a237e;text-decoration:underline;\">Link Text</a>. Use descriptive link text.\n"
                "4. Keep the response concise but informative.\n"
                "5. If the search results do not contain the answer, politely state that you could not find the information."
            )
            user_prompt = f"User Question: {query}\n\nSearch Results:\n{context_text}"
        else:
            header_label = "🌐 Web Fallback Result"
            # No web search results available, call LLM directly for college info using its internal knowledge
            system_prompt = (
                "You are the official AI Assistant for N.B.K.R. Institute of Science & Technology (NBKRIST). "
                "Answer the user's question about NBKRIST comprehensively, warm, and naturally in the style of ChatGPT using your knowledge.\n"
                "IMPORTANT rules:\n"
                "1. Answer in a professional, warm, and natural tone.\n"
                "2. You MUST format your response strictly as clean HTML. Use <p>, <ul>, <li>, <strong>, and <em>. Do NOT use markdown notation like **, *, #, etc.\n"
                "3. Keep the response concise but informative."
            )
            user_prompt = f"User Question: {query}"
    else:
        # General knowledge query - Direct LLM synthesize without web search
        # Check scope first! If query contains no recognized academic/technical keywords → return scope message.
        q_lower = query.lower()
        academic_keywords = [
            # CS, coding, data science, AI, ML, DL, NLP, IT
            "ai", "ds", "ml", "dl", "nlp", "iot", "db", "sql", "git", "api", "web", "app",
            "code", "coding", "program", "software", "hardware", "computer", "comput",
            "network", "cloud", "cyber", "security", "data", "algorithm", "developer",
            "neural", "robot", "automat", "tech", "machine learning", "deep learning",
            "artificial intelligence", "data structure", "database", "programming",
            # Math, science, academic
            "math", "physic", "chem", "calculus", "algebra", "stat", "probability",
            "matrix", "matrices", "derivative", "integrat", "geometr", "engineer",
            "electronic", "electric", "mechanic", "civil", "academic", "course",
            "class", "subject", "study", "exam", "syllabus", "curriculum", "lecture",
            "learn", "teach", "educat", "school", "university", "college", "department",
            # STEM topics (Astronomy, physics, engineering, academic concepts)
            "moon", "space", "astronomy", "nuclear", "reactor", "gravity", "universe",
            "atom", "molecule", "energy", "speed", "light", "star", "planet", "solar",
            "concept", "definition", "formula", "theory", "experiment", "research",
            "calculat", "design", "process", "system", "analysis", "project", "lab"
        ]
        has_academic = any(w in q_lower for w in academic_keywords)
        if not has_academic:
            return "I'm the NBKRIST AI&DS Department assistant. Please ask me something about the college, department, faculty, or academics."

        header_label = "🤖 AI Response"
        system_prompt = (
            "You are a helpful, professional, and knowledgeable AI assistant. "
            "Answer the user's question comprehensively and naturally in the style of ChatGPT.\n"
            "IMPORTANT rules:\n"
            "1. Answer in a professional, warm, and natural tone.\n"
            "2. You MUST format your response strictly as clean HTML. Use <p>, <ul>, <li>, <strong>, and <em>. Do NOT use markdown notation like **, *, #, etc.\n"
            "3. Keep the response concise but informative."
        )
        user_prompt = f"User Question: {query}"

    url = "https://text.pollinations.ai/"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "model": "openai"
    }

    import time
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                llm_response = r.text.strip()
                if llm_response:
                    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif">
  <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:12px 16px;border-radius:10px 10px 0 0">
    <b>{header_label}</b>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;border-radius:0 0 10px 10px;padding:16px 18px;font-size:13.5px;color:#333;line-height:1.7;overflow:hidden">
    {llm_response}
  </div>
</div>"""
            time.sleep(3 * (attempt + 1))
        except Exception as e:
            if attempt == 2:
                print(f"Pollinations API error: {e}")
            time.sleep(3 * (attempt + 1))
    return None



def get_response(query: str, conn_id: str = "default") -> str:
    query = query.strip()
    if not query:
        return _info_card("⚠️ Empty Query", [("Tip","Please type a question.")])

    # ── Preprocess: fix typos + expand abbreviations ──────────────────────
    expanded = preprocess_query(query)

    if "attendance" in expanded:
        return _attendance_card()

    dept_about_signals = [
        "about", "overview", "information", "info", "details", "tell me",
        "what is", "branch", "department", "dept", "intake", "objective",
        "objectives", "highlight", "highlights", "strength", "strengths",
    ]
    specific_query_signals = [
        "timetable", "schedule", "faculty", "hod", "bus", "fee", "attendance",
        "lab", "labs", "subject", "student", "roll", "circular", "notice",
        "placement", "career", "skill", "event", "club",
    ]
    is_aids_query = any(w in expanded for w in ["artificial intelligence data science", "ai&ds", "ai ds"])
    if (
        is_aids_query
        and any(w in expanded for w in dept_about_signals)
        and not any(w in expanded for w in specific_query_signals)
    ):
        return build_department_info_card()

    qa     = analyse_query(expanded)
    intent = detect_intent(expanded, qa)

    # ── Admin access intercept — before everything else ──────────────────
    if re.search(r'\badmin\b', query.lower()):
        return """
<div style="font-family:'Segoe UI',sans-serif;max-width:380px">
  <div style="background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
              padding:13px 18px;border-radius:10px 10px 0 0;font-size:14px;font-weight:700">
    🔐 Admin Access
  </div>
  <div style="background:#fff;padding:16px 18px;border:1px solid #e5e7eb;
              border-top:none;border-radius:0 0 10px 10px">
    <p style="font-size:13px;color:#555;margin-bottom:12px">
      Enter the admin password to open the dashboard.
    </p>
    <div style="display:flex;gap:8px;align-items:center">
      <input id="adminPwdInp" type="password" placeholder="Password…"
        style="flex:1;padding:9px 12px;border:1px solid #c5cae9;border-radius:8px;
               font-size:13px;outline:none;font-family:inherit"
        onkeypress="if(event.key==='Enter')checkAdminPwd()">
      <button onclick="checkAdminPwd()"
        style="background:#1a237e;color:#fff;border:none;padding:9px 16px;
               border-radius:8px;cursor:pointer;font-size:13px;font-weight:600">
        Enter
      </button>
    </div>
    <div id="adminPwdErr" style="color:#c62828;font-size:12px;margin-top:8px;display:none">
      ❌ Wrong password. Try again.
    </div>
  </div>
 </div>"""

    # ── Static intents ────────────────────────────────────────────────────
    if intent == "greeting":
        return _info_card("👋 Hello! I'm the NBKR AI &amp; DS Assistant", [
            ("📅 Timetables", 'Try: "2nd year Section A timetable"'),
            ("👥 Faculty",    'Try: "Who is the HOD?"'),
            ("📢 Circulars",  'Try: "Show circulars" or "fee notice"'),
            ("💻 Services",   'Try: "How to check attendance?"'),
            ("💡 Tip",        "I'll tell you honestly if I don't know something."),
        ])

    if intent == "farewell":
        return _info_card("😊 You're Welcome!", [
            ("Status","Happy to help anytime."),
            ("Tip","Come back if you have more questions about NBKR Institute."),
        ])

    if intent == "help":
        return _help_card()

    # Classify the target module — use expanded (typo-fixed) query
    module = classify_query_module(expanded, qa)

    if module == "unknown":
        local_signals = {"nbkr", "nbkrist", "college", "institute", "vidyanagar", "nellore", "campus", "dept", "department", "syllabus", "curriculum", "timetable", "faculty", "hod", "principal", "admission", "placement", "bus", "fee", "hostel", "library", "labs", "attendance", "circular"}
        has_local_signal = any(w in expanded for w in local_signals)
        
        if not has_local_signal:
            # Pure general knowledge -> skip search -> go straight to LLM directly
            web_reply = handle_web_fallback(query, is_college_context=False)
            if web_reply:
                return web_reply
            log_failed_query(query, "unknown", 0.0)
            return "I couldn't find that information in the uploaded NBKRIST knowledge base."

        threshold = UNKNOWN_FALLBACK_THRESHOLD if has_local_signal else 0.78

        fallback_results = hybrid_retrieve(qa, top_k=3)
        if fallback_results and fallback_results[0][1] >= threshold:
            best_doc, best_score = fallback_results[0]
            return _fallback_answer_card(best_doc)
        # Web fallback
        web_reply = handle_web_fallback(query, is_college_context=has_local_signal)
        if web_reply:
            return web_reply
        log_failed_query(query, "unknown", fallback_results[0][1] if fallback_results else 0.0)
        return "I couldn't find that information in the uploaded NBKRIST knowledge base."

    # Route strictly to relevant data source
    if module == "student":
        student_reply = handle_student_query(expanded)
        if student_reply is not None:
            return student_reply
        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        log_failed_query(query, "student", 0.0)
        return "I couldn't find that information in the uploaded NBKRIST knowledge base."

    if module == "student_timetable":
        return handle_student_timetable_query(expanded)

    if module == "timetable":
        timetable_reply = handle_timetable_query(qa)
        if timetable_reply is not None:
            return timetable_reply
        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        log_failed_query(query, "timetable", 0.0)
        return "I couldn't find that information in the uploaded NBKRIST knowledge base."

    if module == "college":
        local_keywords = [
            "program","course","undergraduate","postgraduate","ug","pg","b.tech","btech","m.tech","mba","mca",
            "campus","facilit","infrastructure","hostel","library","gym","canteen","cafeteria","wi-fi","wifi","atm","bank",
            "placement","recruiter","company","compan","hiring","package","tcs","infosys","wipro","cognizant",
            "subject","syllabus","what do we study","what subjects",
            "lab","laborator",
            "career","job","opportunit","role","engineer","scientist","analyst","developer",
            "skill","technolog","stack","python","react","docker","aws","langchain","rag","mcp",
            "event","activit","techvyuha","inspiron","hackathon","symposium","workshop","fest",
            "chapter","club","ieee","nss","ncc","csi","iste","coding club","cultural",
            "department","ai&ds","aids","ai ds","artificial intelligence data science","dept overview","dept objective",
            "full", "form", "meaning"
        ]
        def _match_keyword(w, text):
            if w == "engineer":
                return bool(re.search(r'\bengineers?\b', text))
            if w in ["b.tech", "m.tech", "ai&ds", "ai ds"]:
                return w in text
            return bool(re.search(rf'\b{w}', text))
            
        is_local = any(_match_keyword(w, expanded) for w in local_keywords)
        is_general_about = any(w in expanded for w in ["about the college", "about college", "overview", "about nbkrist", "about nbkr"])
        
        # Check if we have college signal
        local_signals = {"nbkr", "nbkrist", "college", "institute", "vidyanagar", "nellore", "campus", "dept", "department", "syllabus", "curriculum", "timetable", "faculty", "hod", "principal", "admission", "placement", "bus", "fee", "hostel", "library", "labs", "attendance", "circular"}
        has_local_signal = any(w in expanded for w in local_signals)
        
        if not has_local_signal:
            web_reply = handle_web_fallback(query, is_college_context=False)
            if web_reply:
                return web_reply
            return "I couldn't find that information in the uploaded NBKRIST knowledge base."
            
        # Check for specific syllabus/subject queries
        is_specific_syllabus = "syllabus" in expanded and any(w in expanded for w in ["ai", "artificial intelligence", "ml", "machine learning", "python", "java", "data science", "ds"])

        has_web_kw = any(w in expanded for w in ["correspondence", "distance", "online"])
        if (is_local or is_general_about) and not has_web_kw and not is_specific_syllabus:
            return handle_college_query(expanded)
        else:
            web_reply = handle_web_fallback(query, is_college_context=True)
            if web_reply:
                return web_reply
            return handle_college_query(expanded)

    if module == "department":
        results = retrieve_filtered(qa, "department")
        if results and results[0][1] >= CONFIDENCE_THRESHOLD:
            doc = results[0][0]
            text = doc.get("display_text", doc["text"])
            body_html = f'<p style="padding:12px 16px;font-size:13.5px;color:#333;line-height:1.7;margin:0;white-space:pre-line">{text}</p>'
            return f"""
<div style="margin:8px 0;font-family:'Segoe UI',sans-serif">
  <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:10px 16px;border-radius:10px 10px 0 0">
    <b>ℹ️ Department Information</b>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;border-radius:0 0 10px 10px;overflow:hidden">
    {body_html}
  </div>
</div>"""
        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        log_failed_query(query, "department", results[0][1] if results else 0.0)
        return "I don't know the answer to that question."

    if module == "services":
        if "attendance" in expanded:
            return _attendance_card()
        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        log_failed_query(query, "services", 0.0)
        return "I couldn't find that information in the uploaded NBKRIST knowledge base."

    if module == "labs":
        labs_reply = handle_labs_query(expanded)
        if labs_reply is not None:
            return labs_reply
        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        log_failed_query(query, "labs", 0.0)
        return "I couldn't find that information in the uploaded NBKR AI&DS R23 curriculum."

    # ── faculty_overview_all: generic query → 3-column table with all faculty ──
    if module == "faculty_overview_all":
        return render_faculty_table(_FACULTY_DATA)

    # ── faculty_overview_single: name mentioned → single profile card ──
    if module == "faculty_overview_single":
        fac = _find_faculty_by_name(expanded)
        if fac:
            return build_faculty_card(fac, include_timetable=False)
        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        return render_faculty_table(_FACULTY_DATA)

    # ── Legacy alias kept for any external callers ──
    if module == "faculty_timetable_list":
        return render_faculty_table(_FACULTY_DATA)

    # ── Specific faculty profile sub-module ──
    if module == "faculty_profile":
        fac = _find_faculty_by_name(expanded)
        if not fac and any(w in expanded for w in ["hod","head of department","head of dept"]):
            fac = next((f for f in _FACULTY_DATA if "head" in f.get("designation","").lower()), None)
        if fac:
            return build_faculty_card(fac, include_timetable=False)
        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        log_failed_query(query, "faculty_profile", 0.0)
        return "I couldn't find that information in the uploaded NBKRIST knowledge base."

    # ── Specific faculty profile + timetable sub-module ──
    if module == "faculty_both":
        fac = _find_faculty_by_name(expanded)
        if not fac and any(w in expanded for w in ["hod","head of department","head of dept"]):
            fac = next((f for f in _FACULTY_DATA if "head" in f.get("designation","").lower()), None)
        if fac:
            return build_faculty_card(fac, include_timetable=True)
        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        log_failed_query(query, "faculty_both", 0.0)
        return "I couldn't find that information in the uploaded NBKRIST knowledge base."

    # ── Detail view: exact button-click phrase "timetable of <name>" ──
    if module == "faculty_timetable":
        fac = _find_faculty_by_name(expanded)
        if not fac and any(w in expanded for w in ["hod","head of department","head of dept"]):
            fac = next((f for f in _FACULTY_DATA if "head" in f.get("designation","").lower()), None)
        if fac:
            tt = fac.get("timetable", {})
            if tt:
                name = fac.get("name", "")
                desig = fac.get("designation", "")
                desig_badge = f'<div style="font-size:11px;color:#7c3aed;margin-top:2px">{desig}</div>' if desig else ""
                schedule_overview = f"""
<div style="font-family:'Segoe UI',sans-serif;margin-top:10px;border-radius:10px;overflow:hidden;border:1px solid #c5cae9">
  <div style="background:linear-gradient(135deg,#2d3a6b,#4a5568);color:#fff;padding:10px 16px;font-size:13px;font-weight:700;display:flex;align-items:center;gap:8px">
    &#128197; Faculty Schedule Overview
  </div>
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr>
        <th style="padding:9px 14px;background:#f4f6fb;color:#2d3a6b;font-size:11px;font-weight:700;text-align:left;border:1px solid #e0e4f0">FACULTY NAME</th>
        <th style="padding:9px 14px;background:#f4f6fb;color:#2d3a6b;font-size:11px;font-weight:700;text-align:center;border:1px solid #e0e4f0">TIME TABLE VIEW</th>
        <th style="padding:9px 14px;background:#f4f6fb;color:#2d3a6b;font-size:11px;font-weight:700;text-align:center;border:1px solid #e0e4f0">LEISURE PERIODS VIEW</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px 14px;border:1px solid #e8eaf6;font-size:13px;color:#1a1a2e">
          <div style="font-weight:600">{name}</div>
          {desig_badge}
        </td>
        <td style="padding:10px 14px;border:1px solid #e8eaf6;text-align:center">
          <button onclick="quickSend('timetable of {name}')"
            onmouseover="this.style.opacity=0.85" onmouseout="this.style.opacity=1.0"
            style="background:#2d3a6b;color:#fff;border:none;padding:7px 13px;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer">
            &#128197; View Timetable
          </button>
        </td>
        <td style="padding:10px 14px;border:1px solid #e8eaf6;text-align:center">
          <button onclick="quickSend('leisure hours of {name}')"
            onmouseover="this.style.opacity=0.85" onmouseout="this.style.opacity=1.0"
            style="background:#7c3aed;color:#fff;border:none;padding:7px 13px;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer">
            &#128336; View Leisure
          </button>
        </td>
      </tr>
    </tbody>
  </table>
</div>"""
                return build_faculty_timetable_html(fac) + schedule_overview
        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        log_failed_query(query, "faculty_timetable", 0.0)
        return "I couldn't find that information in the uploaded NBKRIST knowledge base."

    # ── Detail view: exact button-click phrase "leisure hours of <name>" ──
    if module == "faculty_leisure":
        fac = _find_faculty_by_name(expanded)
        if not fac and any(w in expanded for w in ["hod","head of department","head of dept"]):
            fac = next((f for f in _FACULTY_DATA if "head" in f.get("designation","").lower()), None)
        if fac:
            tt = fac.get("timetable", {})
            if tt:
                name = fac.get("name", "")
                desig = fac.get("designation", "")
                desig_badge = f'<div style="font-size:11px;color:#7c3aed;margin-top:2px">{desig}</div>' if desig else ""
                schedule_overview = f"""
<div style="font-family:'Segoe UI',sans-serif;margin-top:10px;border-radius:10px;overflow:hidden;border:1px solid #c5cae9">
  <div style="background:linear-gradient(135deg,#2d3a6b,#4a5568);color:#fff;padding:10px 16px;font-size:13px;font-weight:700;display:flex;align-items:center;gap:8px">
    &#128197; Faculty Schedule Overview
  </div>
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr>
        <th style="padding:9px 14px;background:#f4f6fb;color:#2d3a6b;font-size:11px;font-weight:700;text-align:left;border:1px solid #e0e4f0">FACULTY NAME</th>
        <th style="padding:9px 14px;background:#f4f6fb;color:#2d3a6b;font-size:11px;font-weight:700;text-align:center;border:1px solid #e0e4f0">TIME TABLE VIEW</th>
        <th style="padding:9px 14px;background:#f4f6fb;color:#2d3a6b;font-size:11px;font-weight:700;text-align:center;border:1px solid #e0e4f0">LEISURE PERIODS VIEW</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px 14px;border:1px solid #e8eaf6;font-size:13px;color:#1a1a2e">
          <div style="font-weight:600">{name}</div>
          {desig_badge}
        </td>
        <td style="padding:10px 14px;border:1px solid #e8eaf6;text-align:center">
          <button onclick="quickSend('timetable of {name}')"
            onmouseover="this.style.opacity=0.85" onmouseout="this.style.opacity=1.0"
            style="background:#2d3a6b;color:#fff;border:none;padding:7px 13px;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer">
            &#128197; View Timetable
          </button>
        </td>
        <td style="padding:10px 14px;border:1px solid #e8eaf6;text-align:center">
          <button onclick="quickSend('leisure hours of {name}')"
            onmouseover="this.style.opacity=0.85" onmouseout="this.style.opacity=1.0"
            style="background:#7c3aed;color:#fff;border:none;padding:7px 13px;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer">
            &#128336; View Leisure
          </button>
        </td>
      </tr>
    </tbody>
  </table>
</div>"""
                return build_faculty_leisure_html(fac) + schedule_overview
        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        log_failed_query(query, "faculty_leisure", 0.0)
        return "I couldn't find that information in the uploaded NBKRIST knowledge base."

    if module == "bus_fee":
        results = retrieve_filtered(qa, "bus_fee")
        if results and results[0][1] >= CONFIDENCE_THRESHOLD:
            doc = results[0][0]
            stop = doc.get("stop", "")
            fee = doc.get("fee", 0)
            route = doc.get("route", "")
            matches = [{"location": stop, "fee": fee, "route": route, "year": _BUS_YEAR}]
            return build_bus_fee_card(matches, stop)

        location = re.sub(
            r'\b(bus|fee|fees|fare|charge|charges|transport|transportation|'
            r'from|for|what|is|the|how|much|cost|amount|boarding|route|routes)\b',
            '', expanded, flags=re.IGNORECASE
        ).strip().strip('?').strip()
        if location:
            matches = _search_bus_fee(location)
            if matches:
                return build_bus_fee_card([matches[0]], location)

        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        log_failed_query(query, "bus_fee", results[0][1] if results else 0.0)
        return "I couldn't find that information in the uploaded NBKRIST knowledge base."

    if module == "faculty":
        if any(w in expanded for w in ["hod","head of department","head of dept"]):
            hod = next((f for f in _FACULTY_DATA if "head" in f.get("designation","").lower()), None)
            if hod: return build_faculty_card(hod, include_timetable=False)

        # 2. Check list
        list_signals = {"list","all","show","every","members","staff","teachers","lecturers","how many"}
        is_list = bool(set(qa.tokens) & list_signals) or any(
            w in query.lower() for w in ["list","all faculty","faculty members","show faculty","who are","how many"]
        )
        if is_list:
            return build_faculty_list_table()

        # 3. Direct name match
        direct = _find_faculty_by_name(query)
        if direct: return build_faculty_card(direct, include_timetable=False)

        # 4. Filtered semantic search
        results = retrieve_filtered(qa, "faculty")
        if results and results[0][1] >= CONFIDENCE_THRESHOLD:
            doc = results[0][0]
            fac_name = doc.get("name", "")
            match = next((f for f in _FACULTY_DATA if f.get("name","").lower() == fac_name.lower()), None)
            if match: return build_faculty_card(match, include_timetable=False)

        # 5. Specialisation queries
        spec_map = {
            "Machine Learning":        ["machine","learn","ml"],
            "Deep Learning":           ["deep","learn","dl"],
            "Artificial Intelligence": ["artificial","intelligence","ai"],
            "Computer Networks":       ["network","cn"],
            "Software Engineering":    ["software","engineer"],
            "DBMS / Database":         ["dbms","database"],
            "Computer Science":        ["computer","science","cs"],
        }
        for spec_label, lemmas in spec_map.items():
            if any(lm in qa.tokens for lm in lemmas) or spec_label.lower() in query.lower():
                matched = [f for f in _FACULTY_DATA
                           if any(lm in f.get("specialization","").lower() for lm in lemmas)]
                if matched: return build_specialization_table(spec_label, matched)

        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        log_failed_query(query, "faculty", results[0][1] if results else 0.0)
        return "I don't know the answer to that question."

    if module == "circular":
        q_lower = query.lower()
        for c in _CIRCULARS:
            if c.get("id","").lower() in q_lower:
                return build_circular_card(c)
        if any(w in q_lower for w in ["hostel", "hostels"]):
            for c in _CIRCULARS:
                text = f'{c.get("title","")} {c.get("content","")} {c.get("category","")}'.lower()
                if "hostel" in text:
                    return build_circular_card(c)
        if any(w in q_lower for w in ["bus", "transport", "transportation"]):
            for c in _CIRCULARS:
                text = f'{c.get("title","")} {c.get("content","")} {c.get("category","")}'.lower()
                if "bus" in text or "transport" in text:
                    return build_circular_card(c)
        if any(w in q_lower for w in ["fee circular", "fee circulars", "fee notice", "fee notices"]):
            return build_all_circulars_table()
        if any(w in q_lower for w in ["latest","recent","new","last","current"]):
            if _CIRCULARS: return build_circular_card(_CIRCULARS[-1])

        results = retrieve_filtered(qa, "circular")
        if results and results[0][1] >= CONFIDENCE_THRESHOLD:
            doc = results[0][0]
            circ_data = doc.get("data")
            if circ_data: return build_circular_card(circ_data)

        if any(w in q_lower for w in ["all notices", "list notices", "notices", "announcements", "circulars", "show circulars"]):
            return build_all_circulars_table()

        web_reply = handle_web_fallback(query, is_college_context=True)
        if web_reply:
            return web_reply
        log_failed_query(query, "circular", results[0][1] if results else 0.0)
        return "I don't know the answer to that question."

    if module == "general":
        local_signals = {"nbkr", "nbkrist", "college", "institute", "vidyanagar", "nellore", "campus", "dept", "department", "syllabus", "curriculum", "timetable", "faculty", "hod", "principal", "admission", "placement", "bus", "fee", "hostel", "library", "labs", "attendance", "circular"}
        has_local_signal = any(w in expanded for w in local_signals)
        threshold = CONFIDENCE_THRESHOLD if has_local_signal else 0.78

        results = hybrid_retrieve(qa, top_k=TOP_K)
        if results and results[0][1] >= threshold:
            doc = results[0][0]
            text = doc["text"]
            key = doc.get("key", "Information")
            val = text.replace(f"{key}: ", "")
            return _info_card(f"ℹ️ {key.title()}", [(key.upper(), val)])

        web_reply = handle_web_fallback(query, is_college_context=has_local_signal)
        if web_reply:
            return web_reply
        log_failed_query(query, "general", results[0][1] if results else 0.0)
        return "I don't know the answer to that question."

    web_reply = handle_web_fallback(query, is_college_context=False)
    if web_reply:
        return web_reply
    log_failed_query(query, module, 0.0)
    return "I don't know the answer to that question."



# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 70)
    print("🎓 NBKR Institute AI Chatbot v7.0 — RAG + NLP + ML + Students")
    print("=" * 70)
    initialize_nlp()
    train_ml_classifier()
    load_timetable_data()
    load_faculty_data()
    load_circulars()
    load_curriculum()
    load_bus_fees()
    load_placements()
    load_events()
    load_dept_info()
    load_college_info()
    load_labs_r23()
    load_subjects_list()
    _load_student_dw()
    ok = initialize_rag()
    print("✓ RAG ready" if ok else "⚠ RAG unavailable — check data files")
    print("=" * 70)
    yield

app = FastAPI(title="NBKR RAG+NLP+ML Chatbot v6", version="6.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def home():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <title>NBKR AI Assistant</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
  <meta name="description" content="NBKR Institute AI & DS Department Assistant powered by RAG, NLP and ML">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
  <style>
    *, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
    :root {
      --bg-primary: #0d0d12;
      --bg-secondary: #13131a;
      --bg-tertiary: #1e1e2a;
      --bg-hover: #2a2a3d;
      --text-primary: #eeeef2;
      --text-secondary: #9e9eb8;
      --text-muted: #5a5a78;
      --accent: #8b5cf6;
      --accent-hover: #a78bfa;
      --accent-glow: rgba(139,92,246,.2);
      --border: rgba(255,255,255,.06);
      --border-hover: rgba(255,255,255,.12);
      --sidebar-w: 260px;
      --header-h: 56px;
      --radius: 16px;
      --radius-sm: 12px;
      --transition: .2s cubic-bezier(.4,0,.2,1);
      --font: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    }
    html, body { height:100%; overflow:hidden; }
    body {
      font-family: var(--font);
      background: var(--bg-primary);
      background-image: radial-gradient(ellipse at 50% 0%, rgba(139,92,246,.07) 0%, transparent 50%);
      color: var(--text-primary);
      display: flex;
      -webkit-font-smoothing: antialiased;
    }

    /* Main Area */
    .main {
      flex: 1; display: flex; flex-direction: column;
      min-width: 0; position: relative;
    }

    /* Header */
    .header {
      height: var(--header-h);
      display: flex; align-items: center;
      padding: 0 16px;
      border-bottom: 1px solid var(--border);
      background: rgba(13,13,18,.8);
      backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      gap: 12px; flex-shrink: 0; z-index: 30;
    }
    .menu-toggle {
      width: 36px; height: 36px;
      display: flex; align-items: center; justify-content: center;
      border: none; background: none; cursor: pointer;
      border-radius: 8px;
      transition: background var(--transition);
      color: var(--text-secondary);
    }
    .menu-toggle:hover { background: var(--bg-hover); }
    .menu-toggle svg { width:20px; height:20px; }
    .header-title { font-size: 15px; font-weight: 600; flex: 1; }
    .header-badge {
      font-size: 11px; padding: 3px 10px; border-radius: 20px;
      background: var(--accent-glow); color: var(--accent); font-weight: 600;
    }
    .connection-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #ef4444; transition: background .3s; flex-shrink: 0;
    }
    .connection-dot.on { background: #22c55e; box-shadow: 0 0 8px rgba(34,197,94,.5); }

    /* Messages */
    .messages {
      flex: 1; overflow-y: auto; scroll-behavior: smooth;
      scrollbar-width: thin;
      scrollbar-color: rgba(255,255,255,.1) transparent;
    }
    .messages::-webkit-scrollbar { width: 6px; }
    .messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,.12); border-radius: 3px; }
    /* Messages */
    .msg-row { padding: 14px 16px; animation: msgSlide .3s ease-out; }
    @keyframes msgSlide { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
    .msg-row.user-row { background: transparent; }
    .msg-row.bot-row { background: transparent; border-bottom: 1px solid var(--border); }
    .msg-inner {
      max-width: 760px; margin: 0 auto;
      display: flex; gap: 12px; align-items: flex-start;
    }
    /* User bubble */
    .user-row .msg-inner { justify-content: flex-end; }
    .user-row .msg-avatar { display: none; }
    .user-row .msg-content {
      background: linear-gradient(135deg, #7c3aed, #6d28d9);
      padding: 11px 18px;
      border-radius: 20px 20px 6px 20px;
      max-width: 72%;
      font-size: 14px; line-height: 1.55;
      box-shadow: 0 2px 16px rgba(109,40,217,.3);
    }
    .user-row .msg-content,
    .user-row .msg-content * { color: #fff !important; }
    /* Bot message */
    .msg-avatar {
      width: 28px; height: 28px; border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 13px; flex-shrink: 0; margin-top: 2px;
    }
    .bot-row .msg-avatar {
      background: linear-gradient(135deg, #8b5cf6, #6d28d9);
      color: #fff; box-shadow: 0 2px 10px rgba(139,92,246,.25);
    }
    .msg-content {
      flex: 1; min-width: 0;
      font-size: 14.5px; line-height: 1.75; color: var(--text-primary);
    }
    .msg-content p { margin-bottom: 8px; }
    .msg-content p:last-child { margin-bottom: 0; }
    /* Rich HTML cards — premium floating embed */
    .msg-content .html-card {
      background: #fff;
      border-radius: 14px;
      overflow: hidden;
      color: #1a1a2e;
      box-shadow: 0 4px 24px rgba(0,0,0,.2), 0 0 0 1px rgba(255,255,255,.04);
      margin: 8px 0;
      border-left: 3px solid var(--accent);
    }
    .msg-content table {
      border-collapse: collapse; width: 100%;
      margin: 0; font-size: 13px;
    }
    .msg-content th, .msg-content td {
      border: 1px solid var(--border); padding: 10px 14px; text-align: left;
    }
    .msg-content th {
      background: var(--bg-tertiary); font-weight: 600;
      font-size: 12px; text-transform: uppercase;
      letter-spacing: .04em; color: var(--text-secondary);
    }
    .msg-content td { color: var(--text-primary); }
    .msg-content tr:nth-child(even) td { background: rgba(255,255,255,.02); }
    .msg-content .table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    /* Light-theme overrides inside html-card */
    .msg-content .html-card table { margin: 0; }
    .msg-content .html-card th,
    .msg-content .html-card td {
      border-color: #e5e7eb !important;
      color: #1a1a2e !important;
    }
    .msg-content .html-card th {
      background: #f0f4ff !important;
      color: #1a1a2e !important;
    }
    .msg-content .html-card tr:nth-child(even) td {
      background: #f8f9ff !important;
    }
    .msg-content .html-card a { color: #667eea !important; }

    /* Typing */
    .typing-dots { display: flex; gap: 6px; padding: 6px 0; }
    .typing-dots span {
      width: 8px; height: 8px;
      background: var(--accent); border-radius: 50%;
      animation: typingPulse 1.4s ease-in-out infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay: .2s; }
    .typing-dots span:nth-child(3) { animation-delay: .4s; }
    @keyframes typingPulse { 0%,80%,100%{transform:scale(.5);opacity:.25} 40%{transform:scale(1);opacity:.75} }

    /* Welcome */
    @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
    @keyframes iconFloat { 0%,100%{transform:translateY(0);box-shadow:0 8px 30px rgba(139,92,246,.3)} 50%{transform:translateY(-4px);box-shadow:0 12px 40px rgba(139,92,246,.45)} }
    .welcome {
      flex: 1; display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      padding: 40px 24px; text-align: center; gap: 14px;
    }
    .welcome-icon {
      width: 68px; height: 68px; border-radius: 20px;
      background: linear-gradient(135deg, #8b5cf6, #6d28d9, #4c1d95);
      display: flex; align-items: center; justify-content: center;
      font-size: 32px; margin-bottom: 8px;
      animation: iconFloat 3s ease-in-out infinite;
    }
    .welcome h1 {
      font-size: 28px; font-weight: 700;
      background: linear-gradient(135deg, #fff 0%, #c4b5fd 50%, #8b5cf6 100%);
      background-size: 200% auto;
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
      animation: gradientShift 4s ease infinite;
    }
    .welcome p { font-size: 14px; color: var(--text-secondary); max-width: 460px; line-height: 1.65; }
    .quick-actions {
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 12px; margin-top: 26px; max-width: 560px; width: 100%;
    }
    .quick-action {
      padding: 16px 18px; background: var(--bg-tertiary);
      border: 1px solid var(--border); border-radius: var(--radius-sm);
      cursor: pointer; text-align: left;
      transition: all var(--transition);
      position: relative; overflow: hidden;
    }
    .quick-action::before {
      content: ''; position: absolute; inset: 0;
      background: linear-gradient(135deg, rgba(139,92,246,.08), transparent);
      opacity: 0; transition: opacity var(--transition);
    }
    .quick-action:hover {
      border-color: rgba(139,92,246,.3);
      transform: translateY(-2px);
      box-shadow: 0 4px 20px rgba(139,92,246,.12);
    }
    .quick-action:hover::before { opacity: 1; }
    .quick-action .qa-icon { font-size: 20px; margin-bottom: 8px; position: relative; }
    .quick-action .qa-title { font-size: 13px; font-weight: 600; color: var(--text-primary); position: relative; }
    .quick-action .qa-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; position: relative; }

    /* Input Area */
    .input-area {
      padding: 12px 16px 20px;
      background: linear-gradient(to top, var(--bg-primary) 70%, transparent);
      flex-shrink: 0;
    }
    .input-wrap {
      max-width: 760px; margin: 0 auto;
      display: flex; align-items: flex-end; gap: 10px;
      background: var(--bg-tertiary);
      border: 1px solid var(--border); border-radius: 24px;
      padding: 8px 10px 8px 18px;
      transition: border-color var(--transition), box-shadow var(--transition);
    }
    .input-wrap:focus-within {
      border-color: rgba(139,92,246,.4);
      box-shadow: 0 0 0 3px var(--accent-glow), 0 4px 20px rgba(139,92,246,.08);
    }
    .input-wrap textarea {
      flex: 1; background: transparent; border: none; outline: none;
      color: var(--text-primary); font-family: var(--font);
      font-size: 14px; line-height: 1.5;
      resize: none; max-height: 150px; min-height: 24px; padding: 4px 0;
    }
    .input-wrap textarea::placeholder { color: var(--text-muted); }
    .send-btn {
      width: 36px; height: 36px; border-radius: 50%;
      border: none; background: var(--accent); color: #fff;
      cursor: pointer; display: flex; align-items: center; justify-content: center;
      flex-shrink: 0;
      transition: all var(--transition);
      opacity: .4;
    }
    .send-btn.active { opacity: 1; box-shadow: 0 2px 12px rgba(139,92,246,.3); }
    .send-btn.active:hover { background: var(--accent-hover); transform: scale(1.08); }
    .send-btn svg { width: 18px; height: 18px; }
    .input-hint {
      text-align: center; padding-top: 8px;
      font-size: 11px; color: var(--text-muted);
      max-width: 760px; margin: 0 auto;
    }

    /* Mobile Overlay */
    .overlay { display: none; }

    /* Responsive */
    @media (max-width: 768px) {
      .quick-actions { grid-template-columns: 1fr; gap: 8px; margin-top: 16px; }
      .welcome h1 { font-size: 20px; }
      .welcome p  { font-size: 13px; }
      .welcome    { padding: 24px 14px; gap: 10px; }
      .welcome-icon { width: 54px; height: 54px; font-size: 26px; }
      .msg-row    { padding: 10px 10px; }
      .msg-inner  { gap: 8px; }
      .bot-row .msg-content { font-size: 13.5px; }
      .msg-avatar { width: 24px; height: 24px; font-size: 11px; border-radius: 8px; }
      .user-row .msg-content { max-width: 88%; font-size: 13px; padding: 9px 13px; }
      .header-badge { display: none; }
      .msg-content .html-card { border-radius: 10px; }
      .msg-content .html-card div[style*="min-width:130px"] { min-width: 90px !important; font-size: 10px !important; }
      .msg-content .html-card div[style*="padding:14px 18px"] { padding: 10px 12px !important; }
      .msg-content .html-card div[style*="padding:10px 16px"] { padding: 8px 12px !important; }
      .msg-content .html-card span[style*="font-size:15px"] { font-size: 14px !important; }
      .msg-content .html-card span[style*="font-size:13px"] { font-size: 12px !important; }
      .input-area { padding: 10px 10px 16px; }
      .input-hint { font-size: 10px; }
    }
    @media (max-width: 480px) {
      .timetable-actions {
        flex-direction: column !important;
        align-items: stretch !important;
        gap: 8px !important;
      }
      .timetable-action-btn {
        width: 100% !important;
        justify-content: center !important;
        padding: 8px 12px !important;
      }
      .quick-actions { grid-template-columns: 1fr; }
      .quick-action  { padding: 12px 13px; }
      .user-row .msg-content { max-width: 92%; }
      .header { padding: 0 10px; }
      .header-title { font-size: 14px; }
    }
    @media (max-width: 400px) {
      .input-wrap { border-radius: 18px; padding: 6px 8px 6px 12px; }
      .input-area { padding: 8px 8px 12px; }
      .welcome { padding: 20px 12px; }
      .welcome h1 { font-size: 18px; }
    }

    /* ChatGPT-style Centered Layout Overrides */
    .main.is-home {
      justify-content: space-between;
      height: 100vh;
    }
    .main.is-home #welcomeScreen {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      align-items: center;
      padding-bottom: 24px;
    }
    .main.is-home .input-area {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      align-items: center;
      width: 100%;
      padding-top: 0;
      background: transparent;
    }
    .main.is-home .input-wrap {
      max-width: 640px;
      padding: 12px 14px 12px 20px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(139, 92, 246, 0.1);
      border-color: rgba(139, 92, 246, 0.2);
    }
    .new-chat-btn {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border: 1px solid var(--border);
      background: var(--bg-tertiary);
      color: var(--text-primary);
      border-radius: 18px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: all var(--transition);
      margin-right: 12px;
    }
    .new-chat-btn:hover {
      background: var(--bg-hover);
      border-color: rgba(139,92,246,.4);
      box-shadow: 0 0 8px rgba(139,92,246,.15);
    }
    .new-chat-btn svg {
      color: var(--text-secondary);
    }
    .input-wrap {
      width: 100%;
    }
    @media (max-width: 768px) {
      .new-chat-btn span { display: none; }
      .new-chat-btn { padding: 6px; border-radius: 50%; margin-right: 4px; }
    }
  </style>
</head>
<body>



<div class="main is-home">
  <header class="header">
    <span class="header-title">🎓 NBKR AI Assistant</span>
    <span class="header-badge">RAG + NLP + ML</span>
    <button class="new-chat-btn" id="newChatBtn" onclick="newChat()" title="Start a new conversation">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:14px; height:14px;"><path d="M12 5v14M5 12h14"/></svg>
      <span>New Chat</span>
    </button>
    <div class="connection-dot" id="connDot" title="Disconnected"></div>
  </header>

  <div class="welcome" id="welcomeScreen">
    <div class="welcome-icon">🎓</div>
    <h1>NBKR Department Chatbot</h1>
  </div>

  <div class="messages" id="msgs" style="display:none;"></div>

  <div class="input-area">
    <div class="input-wrap" id="inputWrap">
      <textarea id="inp" rows="1" placeholder="Ask me anything about NBKR AI &amp; DS..." autocomplete="off"></textarea>
      <button class="send-btn" id="sendBtn" onclick="send()" aria-label="Send message">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13"/><path d="M22 2 15 22 11 13 2 9z"/></svg>
      </button>
    </div>
  </div>
</div>

<script>
  let ws;
  const msgs     = document.getElementById('msgs');
  const inp      = document.getElementById('inp');
  const sendBtn  = document.getElementById('sendBtn');
  const connDot  = document.getElementById('connDot');
  const welcome  = document.getElementById('welcomeScreen');
  const isMobile = () => window.innerWidth <= 768;

  inp.addEventListener('input', () => {
    inp.style.height = 'auto';
    inp.style.height = Math.min(inp.scrollHeight, 150) + 'px';
    sendBtn.classList.toggle('active', inp.value.trim().length > 0);
  });

  function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/ws');
    ws.onopen = () => { connDot.className='connection-dot on'; connDot.title='Connected'; };
    ws.onclose = () => { connDot.className='connection-dot'; connDot.title='Disconnected'; setTimeout(connect,3000); };
    ws.onmessage = e => { const d=JSON.parse(e.data); removeTyping(); addMsg(d.message,'bot'); };
  }

  function showConversation() {
    welcome.style.display = 'none';
    msgs.style.display = 'flex';
    msgs.style.flexDirection = 'column';
    document.querySelector('.main').classList.remove('is-home');
  }

  function addMsg(text, who) {
    showConversation();
    const row = document.createElement('div');
    row.className = 'msg-row ' + who + '-row';
    const inner = document.createElement('div');
    inner.className = 'msg-inner';
    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.innerHTML = who === 'bot' ? '&#10022;' : 'U';
    const content = document.createElement('div');
    content.className = 'msg-content';
    if (who === 'bot') {
      // Wrap rich HTML cards (tables, timetables, faculty, circulars) in a
      // white-background container so inline light-coloured styles are readable.
      if (text.trim().startsWith('<div') || text.trim().startsWith('<table')) {
        content.innerHTML = '<div class="html-card">' + text + '</div>';
      } else {
        content.innerHTML = text;
      }
      content.querySelectorAll('table').forEach(t => {
        if (!t.parentElement.classList.contains('table-scroll')) {
          const w = document.createElement('div');
          w.className = 'table-scroll';
          t.parentNode.insertBefore(w, t);
          w.appendChild(t);
        }
      });
    } else {
      content.textContent = text;
    }
    inner.appendChild(avatar);
    inner.appendChild(content);
    row.appendChild(inner);
    msgs.appendChild(row);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function showTyping() {
    showConversation();
    const row = document.createElement('div');
    row.className = 'msg-row bot-row'; row.id = 'typing';
    row.innerHTML = '<div class="msg-inner"><div class="msg-avatar" style="background:linear-gradient(135deg,#8b5cf6,#6d28d9);color:#fff;box-shadow:0 2px 8px rgba(139,92,246,.3)">&#10022;</div><div class="msg-content"><div class="typing-dots"><span></span><span></span><span></span></div></div></div>';
    msgs.appendChild(row); msgs.scrollTop = msgs.scrollHeight;
  }

  function removeTyping() { const t=document.getElementById('typing'); if(t) t.remove(); }

  // ── Admin access flow ────────────────────────────────────────────────
  let _adminMode = false;  // true when waiting for password

  function showAdminPrompt() {
    showConversation();
    _adminMode = true;
    const row = document.createElement('div');
    row.className = 'msg-row bot-row';
    row.innerHTML = `
      <div class="msg-inner">
        <div class="msg-avatar" style="background:linear-gradient(135deg,#8b5cf6,#6d28d9);color:#fff">&#10022;</div>
        <div class="msg-content">
          <div class="html-card" style="max-width:360px">
            <div style="background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
                        padding:13px 18px;border-radius:10px 10px 0 0;font-size:14px;font-weight:700">
              🔐 Admin Access
            </div>
            <div style="background:#fff;padding:16px 18px;border:1px solid #e5e7eb;
                        border-top:none;border-radius:0 0 10px 10px">
              <p style="font-size:13px;color:#555;margin-bottom:12px">
                Enter the admin password to open the dashboard.
              </p>
              <div style="display:flex;gap:8px;align-items:center">
                <input id="adminPwdInp" type="password" placeholder="Password…"
                  style="flex:1;padding:9px 12px;border:1px solid #c5cae9;border-radius:8px;
                         font-size:13px;outline:none;font-family:inherit"
                  onkeypress="if(event.key==='Enter')checkAdminPwd()">
                <button onclick="checkAdminPwd()"
                  style="background:#1a237e;color:#fff;border:none;padding:9px 16px;
                         border-radius:8px;cursor:pointer;font-size:13px;font-weight:600">
                  Enter
                </button>
              </div>
              <div id="adminPwdErr" style="color:#c62828;font-size:12px;margin-top:8px;display:none">
                ❌ Wrong password. Try again.
              </div>
            </div>
          </div>
        </div>
      </div>`;
    msgs.appendChild(row);
    msgs.scrollTop = msgs.scrollHeight;
    setTimeout(() => { const el = document.getElementById('adminPwdInp'); if(el) el.focus(); }, 100);
  }

  function checkAdminPwd() {
    const pwd = (document.getElementById('adminPwdInp') || {}).value || '';
    fetch('/admin/faculty/list', { headers: {'X-Admin-Password': pwd} })
      .then(r => {
        if (r.ok) {
          _adminMode = false;
          addMsg('✅ Password correct! Opening admin dashboard…', 'bot');
          setTimeout(() => { window.open('/admin?pwd=' + encodeURIComponent(pwd), '_blank'); }, 600);
        } else {
          const err = document.getElementById('adminPwdErr');
          if (err) err.style.display = 'block';
          const inp2 = document.getElementById('adminPwdInp');
          if (inp2) { inp2.value = ''; inp2.focus(); }
        }
      });
  }

  function send() {
    const msg = inp.value.trim();
    if (!msg) return;

    // Admin keyword intercept
    if (msg.toLowerCase().match(/\badmin\b/)) {
      addMsg(msg, 'user');
      inp.value = ''; inp.style.height = 'auto';
      sendBtn.classList.remove('active');
      showAdminPrompt();
      return;
    }

    if (!ws || ws.readyState !== 1) return;
    addMsg(msg, 'user');
    showTyping();
    ws.send(JSON.stringify({ message: msg }));
    inp.value = ''; inp.style.height = 'auto';
    sendBtn.classList.remove('active');
  }

  function quickSend(text) { inp.value = text; send(); }

  function newChat() {
    msgs.innerHTML = '';
    msgs.style.display = 'none';
    welcome.style.display = 'flex';
    document.querySelector('.main').classList.add('is-home');
  }

  inp.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  function downloadAsPNG(btn) {
    const card = btn.closest('.timetable-card-container');
    if (!card) return;
    const actions = card.querySelector('.timetable-actions');
    if (actions) actions.style.setProperty('display', 'none', 'important');
    
    const originalPadding = card.style.padding;
    const originalBackground = card.style.background;
    card.style.setProperty('padding', '16px', 'important');
    card.style.setProperty('background', '#ffffff', 'important');
    
    html2canvas(card, {
      scale: 2,
      useCORS: true,
      backgroundColor: '#ffffff'
    }).then(canvas => {
      if (actions) actions.style.removeProperty('display');
      card.style.padding = originalPadding;
      card.style.background = originalBackground;
      
      const link = document.createElement('a');
      link.download = 'timetable_' + new Date().toISOString().slice(0, 10) + '.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    }).catch(err => {
      console.error('PNG download failed:', err);
      if (actions) actions.style.removeProperty('display');
      card.style.padding = originalPadding;
      card.style.background = originalBackground;
      alert('Failed to generate PNG image. Please try again.');
    });
  }

  function downloadAsExcel(btn) {
    const card = btn.closest('.timetable-card-container');
    if (!card) return;
    const table = card.querySelector('.timetable-table') || card.querySelector('table');
    if (!table) return;

    const rows = table.querySelectorAll('tr');
    const csvData = [];

    rows.forEach(tr => {
      const cols = tr.querySelectorAll('th, td');
      const rowData = [];
      cols.forEach(col => {
        let text = col.innerText.trim().replace(/\\r?\\n|\\r/g, ' ');
        text = text.replace(/"/g, '""');
        
        const colSpan = parseInt(col.getAttribute('colspan') || '1', 10);
        rowData.push('"' + text + '"');
        for (let i = 1; i < colSpan; i++) {
          rowData.push('""');
        }
      });
      csvData.push(rowData.join(','));
    });

    const legend = card.querySelector('table + table') || card.querySelector('div[style*="background"]');
    if (legend) {
      csvData.push('\\n');
      csvData.push('"Subject & Faculty Legend"');
      const legendRows = legend.querySelectorAll('tr');
      legendRows.forEach(tr => {
        const cols = tr.querySelectorAll('td');
        const rowData = [];
        cols.forEach(col => {
          let text = col.innerText.trim().replace(/\\r?\\n|\\r/g, ' ');
          text = text.replace(/"/g, '""');
          rowData.push('"' + text + '"');
        });
        csvData.push(rowData.join(','));
      });
    }

    const csvContent = csvData.join('\\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', 'timetable_' + new Date().toISOString().slice(0, 10) + '.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function downloadAsPDF(btn) {
    const card = btn.closest('.timetable-card-container');
    if (!card) return;
    
    const clone = card.cloneNode(true);
    const actions = clone.querySelector('.timetable-actions');
    if (actions) actions.remove();

    const printWindow = window.open('', '_blank', 'width=950,height=750');
    printWindow.document.write('<html><head><title>NBKR AI Assistant - Timetable</title>');
    printWindow.document.write('<style>');
    printWindow.document.write('body { font-family: "Segoe UI", Arial, sans-serif; margin: 30px; color: #333; background: #fff; }');
    printWindow.document.write('table { border-collapse: collapse; width: 100%; margin-top: 15px; }');
    printWindow.document.write('th, td { border: 1px solid #bbb !important; padding: 10px; text-align: center; font-size: 12px; }');
    printWindow.document.write('th { background-color: #f2f2f2; font-weight: bold; }');
    printWindow.document.write('td.day-cell { background-color: #fce4ec !important; color: #880e4f !important; font-weight: bold; -webkit-print-color-adjust: exact; print-color-adjust: exact; }');
    printWindow.document.write('.slot-inner { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 40px; }');
    printWindow.document.write('.subj-name { font-weight: bold; font-size: 12px; }');
    printWindow.document.write('.fac-name { font-size: 10.5px; opacity: 0.8; margin-top: 3px; }');
    printWindow.document.write('</style></head><body>');
    printWindow.document.write(clone.innerHTML);
    printWindow.document.write('</body></html>');
    printWindow.document.close();
    
    printWindow.focus();
    setTimeout(() => {
      printWindow.print();
      printWindow.close();
    }, 500);
  }

  connect();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    conn_id = str(id(websocket))
    active_connections.append(websocket)
    try:
        while True:
            raw      = await websocket.receive_text()
            data     = json.loads(raw)
            user_msg = data.get("message","").strip()
            chat_history.append({"ts":datetime.now().isoformat(),"user":user_msg,"bot":None})
            response = get_response(user_msg, conn_id)
            chat_history[-1]["bot"] = response
            await websocket.send_json({"message":response,"timestamp":datetime.now().isoformat()})
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# Admin Dashboard — Password-protected CRUD routes
# ─────────────────────────────────────────────────────────────────────────────
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "nbkr2026")


def _check_admin(request: Request) -> bool:
    """Validate admin password from header."""
    pwd = request.headers.get("X-Admin-Password", "")
    return pwd == ADMIN_PASSWORD


@app.get("/admin")
async def admin_page():
    """Serve the admin dashboard HTML."""
    html_path = os.path.join(os.path.dirname(__file__) or ".", "admin_dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Admin dashboard file not found</h1>", status_code=404)


@app.get("/admin/data")
async def admin_data(request: Request):
    """Return all editable data for the admin dashboard."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Load timetable data
    tt_data = {"timetable": {}, "subjects": {}, "faculty": {}}
    if os.path.exists("aids_timetable_data.json"):
        with open("aids_timetable_data.json", "r", encoding="utf-8") as f:
            tt_data = json.load(f)

    # Load faculty data
    fac_data = []
    if os.path.exists("aids_faculty_data.json"):
        with open("aids_faculty_data.json", "r", encoding="utf-8") as f:
            fac_data = json.load(f)

    # Load circulars
    circ_data = []
    if os.path.exists("nbkr_circulars.json"):
        with open("nbkr_circulars.json", "r", encoding="utf-8") as f:
            circ_data = json.load(f)

    # Load knowledge base
    kb_data = {}
    if os.path.exists("knowledge_updates.json"):
        with open("knowledge_updates.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, dict):
                kb_data = raw
            elif isinstance(raw, list):
                kb_data = {item.get("title", f"item_{i}"): item.get("text", "") for i, item in enumerate(raw)}

    # Log admin access
    _audit_log("LOGIN")

    return JSONResponse({
        "timetable_data": tt_data,
        "faculty_data": fac_data,
        "circular_data": circ_data,
        "knowledge_data": kb_data,
    })


@app.post("/admin/save-timetable")
async def save_timetable(request: Request):
    """Save timetable data and rebuild RAG index."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    data = await request.json()
    with open("aids_timetable_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Reload timetable in memory
    load_timetable_data()
    # Rebuild RAG index with updated data
    initialize_rag()

    _audit_log("SAVE_TIMETABLE")
    return JSONResponse({"status": "ok"})


@app.post("/admin/save-circular")
async def save_circular(request: Request):
    """Save circulars data."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    data = await request.json()
    with open("nbkr_circulars.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Reload circulars in memory
    load_circulars()
    initialize_rag()

    _audit_log("SAVE_CIRCULARS")
    return JSONResponse({"status": "ok"})


@app.post("/admin/save-faculty")
async def save_faculty(request: Request):
    """Save faculty data."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    data = await request.json()
    with open("aids_faculty_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Reload faculty in memory
    load_faculty_data()
    initialize_rag()

    _audit_log("SAVE_FACULTY")
    return JSONResponse({"status": "ok"})


@app.post("/admin/save-knowledge")
async def save_knowledge(request: Request):
    """Save knowledge base / FAQ data."""
    if not _check_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    data = await request.json()
    # Convert dict back to list format for storage
    if isinstance(data, dict):
        kb_list = []
        for key, val in data.items():
            kb_list.append({
                "title": key,
                "text": val,
                "category": "general",
                "added_by": "admin",
                "added_at": datetime.now().isoformat(),
            })
        with open("knowledge_updates.json", "w", encoding="utf-8") as f:
            json.dump(kb_list, f, indent=2, ensure_ascii=False)
    else:
        with open("knowledge_updates.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    initialize_rag()
    _audit_log("SAVE_KNOWLEDGE")
    return JSONResponse({"status": "ok"})


# ─────────────────────────────────────────────────────────────────────────────
# Admin CRUD — Faculty
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/admin/faculty/add")
async def admin_faculty_add(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    required = ["name","designation","Qualification","date_of_joining","phone","email"]
    for f in required:
        if not body.get(f):
            return JSONResponse({"error":f"Missing field: {f}"},status_code=400)
    if not body.get("timetable"):
        body["timetable"] = []
    _FACULTY_DATA.append(body)
    with open("aids_faculty_data.json","w",encoding="utf-8") as f:
        json.dump(_FACULTY_DATA, f, indent=2, ensure_ascii=False)
    initialize_rag()
    _audit_log(f"FACULTY_ADD:{body['name']}")
    return JSONResponse({"status":"added","name":body["name"],"total":len(_FACULTY_DATA)})


@app.delete("/admin/faculty/delete")
async def admin_faculty_delete(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    name = body.get("name","").strip().lower()
    idx = next((i for i,f in enumerate(_FACULTY_DATA) if f.get("name","").lower() == name), None)
    if idx is None:
        return JSONResponse({"error":f"Faculty '{body.get('name')}' not found"},status_code=404)
    removed = _FACULTY_DATA.pop(idx)
    with open("aids_faculty_data.json","w",encoding="utf-8") as f:
        json.dump(_FACULTY_DATA, f, indent=2, ensure_ascii=False)
    initialize_rag()
    _audit_log(f"FACULTY_DELETE:{removed['name']}")
    return JSONResponse({"status":"deleted","name":removed["name"],"total":len(_FACULTY_DATA)})


@app.get("/admin/faculty/list")
async def admin_faculty_list(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    return JSONResponse({"faculty":_FACULTY_DATA,"total":len(_FACULTY_DATA)})


# ─────────────────────────────────────────────────────────────────────────────
# Admin CRUD — Students
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/admin/student/add")
async def admin_student_add(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    if not body.get("roll_no"):
        return JSONResponse({"error":"roll_no is required"},status_code=400)
    ok, msg = student_create(body)
    if not ok: return JSONResponse({"error":msg},status_code=400)
    _audit_log(f"STUDENT_ADD:{body['roll_no'].upper()}")
    return JSONResponse({"status":"added","message":msg,"total":len(_students)})


@app.delete("/admin/student/delete")
async def admin_student_delete_admin(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    roll = body.get("roll_no","")
    ok, msg = student_delete(roll)
    if not ok: return JSONResponse({"error":msg},status_code=404)
    _audit_log(f"STUDENT_DELETE:{roll.upper()}")
    return JSONResponse({"status":"deleted","message":msg,"total":len(_students)})


@app.get("/admin/student/list")
async def admin_student_list(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    return JSONResponse({"students":_students,"total":len(_students)})


# ─────────────────────────────────────────────────────────────────────────────
# Admin CRUD — Curriculum
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/admin/curriculum/list")
async def admin_curriculum_list(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    return JSONResponse({"curriculum":_CURRICULUM,"semesters":list(_CURRICULUM.keys())})


@app.post("/admin/curriculum/add-course")
async def admin_curriculum_add_course(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    sem     = body.get("semester","")
    section = body.get("section","Electives")
    course  = body.get("course","")
    lab     = body.get("lab","None")
    if not sem or not course:
        return JSONResponse({"error":"semester and course are required"},status_code=400)
    if sem not in _CURRICULUM:
        _CURRICULUM[sem] = {}
    if section not in _CURRICULUM[sem]:
        _CURRICULUM[sem][section] = []
    _CURRICULUM[sem][section].append({"Course":course,"Lab":lab})
    with open("aids_curriculum.json","w",encoding="utf-8") as f:
        json.dump(_CURRICULUM, f, indent=2, ensure_ascii=False)
    _audit_log(f"CURRICULUM_ADD:{sem}:{course}")
    return JSONResponse({"status":"added","semester":sem,"course":course})


@app.delete("/admin/curriculum/delete-course")
async def admin_curriculum_delete_course(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    sem     = body.get("semester","")
    section = body.get("section","Electives")
    course  = body.get("course","")
    if sem not in _CURRICULUM or section not in _CURRICULUM.get(sem,{}):
        return JSONResponse({"error":"Semester or section not found"},status_code=404)
    items = _CURRICULUM[sem][section]
    idx = next((i for i,x in enumerate(items) if course.lower() in x.get("Course","").lower()), None)
    if idx is None:
        return JSONResponse({"error":f"Course '{course}' not found"},status_code=404)
    removed = items.pop(idx)
    with open("aids_curriculum.json","w",encoding="utf-8") as f:
        json.dump(_CURRICULUM, f, indent=2, ensure_ascii=False)
    _audit_log(f"CURRICULUM_DELETE:{sem}:{removed['Course']}")
    return JSONResponse({"status":"deleted","course":removed["Course"]})


# ─────────────────────────────────────────────────────────────────────────────
# Admin — Notices / Circulars CRUD
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/admin/notices/list")
async def admin_notices_list(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    return JSONResponse({"notices":_CIRCULARS,"total":len(_CIRCULARS)})

@app.post("/admin/notices/add")
async def admin_notices_add(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    if not body.get("title"): return JSONResponse({"error":"title required"},status_code=400)
    body.setdefault("id", f"N-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    body.setdefault("date", datetime.now().strftime("%Y-%m-%d"))
    body.setdefault("category","general")
    body.setdefault("content","")
    body.setdefault("key_dates",{})
    body.setdefault("applicable_to","")
    body.setdefault("payment_portal","")
    _CIRCULARS.append(body)
    with open("nbkr_circulars.json","w",encoding="utf-8") as f:
        json.dump(_CIRCULARS, f, indent=2, ensure_ascii=False)
    initialize_rag()
    _audit_log(f"NOTICE_ADD:{body['title']}")
    return JSONResponse({"status":"added","id":body["id"],"total":len(_CIRCULARS)})

@app.delete("/admin/notices/delete")
async def admin_notices_delete(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    nid = body.get("id","")
    idx = next((i for i,c in enumerate(_CIRCULARS) if c.get("id")==nid), None)
    if idx is None: return JSONResponse({"error":f"Notice {nid} not found"},status_code=404)
    removed = _CIRCULARS.pop(idx)
    with open("nbkr_circulars.json","w",encoding="utf-8") as f:
        json.dump(_CIRCULARS, f, indent=2, ensure_ascii=False)
    initialize_rag()
    _audit_log(f"NOTICE_DELETE:{removed.get('title','?')}")
    return JSONResponse({"status":"deleted","id":nid,"total":len(_CIRCULARS)})

# ─────────────────────────────────────────────────────────────────────────────
# Admin — Faculty Timetable CRUD
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/admin/faculty/timetable/add")
async def admin_fac_tt_add(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    name = body.get("name","").strip()
    day  = body.get("day","")
    slot = {"time":body.get("time",""),"subject":body.get("subject",""),
            "class":body.get("class",""),"type":body.get("type","theory")}
    if not name or not day or not slot["subject"]:
        return JSONResponse({"error":"name, day, subject required"},status_code=400)
    fac = next((f for f in _FACULTY_DATA if f.get("name","").lower()==name.lower()), None)
    if not fac: return JSONResponse({"error":f"Faculty '{name}' not found"},status_code=404)
    if not isinstance(fac.get("timetable"), dict): fac["timetable"] = {}
    fac["timetable"].setdefault(day, [])
    fac["timetable"][day].append(slot)
    with open("aids_faculty_data.json","w",encoding="utf-8") as f:
        json.dump(_FACULTY_DATA, f, indent=2, ensure_ascii=False)
    _audit_log(f"FAC_TT_ADD:{name}:{day}:{slot['subject']}")
    return JSONResponse({"status":"added","name":name,"day":day,"slot":slot})

@app.delete("/admin/faculty/timetable/delete")
async def admin_fac_tt_delete(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    name = body.get("name","").strip()
    day  = body.get("day","")
    subj = body.get("subject","").strip().lower()
    fac  = next((f for f in _FACULTY_DATA if f.get("name","").lower()==name.lower()), None)
    if not fac: return JSONResponse({"error":f"Faculty '{name}' not found"},status_code=404)
    slots = fac.get("timetable",{}).get(day,[])
    idx   = next((i for i,s in enumerate(slots) if subj in s.get("subject","").lower()), None)
    if idx is None: return JSONResponse({"error":f"Slot not found"},status_code=404)
    removed = slots.pop(idx)
    with open("aids_faculty_data.json","w",encoding="utf-8") as f:
        json.dump(_FACULTY_DATA, f, indent=2, ensure_ascii=False)
    _audit_log(f"FAC_TT_DELETE:{name}:{day}:{removed['subject']}")
    return JSONResponse({"status":"deleted","removed":removed})

# ─────────────────────────────────────────────────────────────────────────────
# Admin — Class Timetable CRUD
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/admin/timetable/list")
async def admin_tt_list(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    with open("aids_timetable_data.json","r",encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)

@app.post("/admin/timetable/update-slot")
async def admin_tt_update(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    section = body.get("section","")
    day     = body.get("day","")
    slot    = body.get("slot","")
    value   = body.get("value","")
    with open("aids_timetable_data.json","r",encoding="utf-8") as f:
        data = json.load(f)
    if section not in data.get("timetable",{}):
        return JSONResponse({"error":f"Section '{section}' not found"},status_code=404)
    if day not in data["timetable"][section]:
        data["timetable"][section][day] = {}
    if value:
        data["timetable"][section][day][slot] = value
    else:
        data["timetable"][section][day].pop(slot, None)
    with open("aids_timetable_data.json","w",encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    load_timetable_data()
    initialize_rag()
    _audit_log(f"TT_UPDATE:{section}:{day}:{slot}={value}")
    return JSONResponse({"status":"updated","section":section,"day":day,"slot":slot,"value":value})

# ─────────────────────────────────────────────────────────────────────────────
# Admin — Knowledge Base CRUD
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/admin/knowledge/list")
async def admin_kb_list(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    kb = []
    if os.path.exists("knowledge_updates.json"):
        with open("knowledge_updates.json","r",encoding="utf-8") as f:
            kb = json.load(f)
    if os.path.exists("nbkr_knowledge_base.json"):
        with open("nbkr_knowledge_base.json","r",encoding="utf-8") as f:
            base = json.load(f)
        for k,v in base.items():
            kb.append({"id":k,"title":k,"text":str(v),"source":"base"})
    return JSONResponse({"knowledge":kb,"total":len(kb)})

@app.post("/admin/knowledge/add")
async def admin_kb_add(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    title = body.get("title","").strip()
    text  = body.get("text","").strip()
    if not title or not text: return JSONResponse({"error":"title and text required"},status_code=400)
    kb = []
    if os.path.exists("knowledge_updates.json"):
        with open("knowledge_updates.json","r",encoding="utf-8") as f:
            kb = json.load(f)
    entry = {"id":f"kb_{datetime.now().strftime('%Y%m%d%H%M%S')}","title":title,"text":text,
             "category":"general","added_by":"admin","added_at":datetime.now().isoformat(),"source":"admin"}
    kb.append(entry)
    with open("knowledge_updates.json","w",encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    initialize_rag()
    _audit_log(f"KB_ADD:{title}")
    return JSONResponse({"status":"added","id":entry["id"],"total":len(kb)})

@app.delete("/admin/knowledge/delete")
async def admin_kb_delete(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    kid  = body.get("id","")
    kb   = []
    if os.path.exists("knowledge_updates.json"):
        with open("knowledge_updates.json","r",encoding="utf-8") as f:
            kb = json.load(f)
    idx = next((i for i,x in enumerate(kb) if x.get("id")==kid or x.get("title")==kid), None)
    if idx is None: return JSONResponse({"error":f"Entry '{kid}' not found"},status_code=404)
    removed = kb.pop(idx)
    with open("knowledge_updates.json","w",encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    initialize_rag()
    _audit_log(f"KB_DELETE:{removed.get('title','?')}")
    return JSONResponse({"status":"deleted","title":removed.get("title","?"),"total":len(kb)})

# ─────────────────────────────────────────────────────────────────────────────
# Admin — Audit Log
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/admin/audit/list")
async def admin_audit_list(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    if os.path.exists("admin_audit.json"):
        with open("admin_audit.json","r",encoding="utf-8") as f:
            entries = json.load(f)
        return JSONResponse({"audit":list(reversed(entries)),"total":len(entries)})
    return JSONResponse({"audit":[],"total":0})


@app.get("/admin/failed-queries")
async def admin_failed_queries_list(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    logs = []
    if os.path.exists("failed_queries.json"):
        try:
            with open("failed_queries.json", "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            pass
    return JSONResponse({"failed_queries": list(reversed(logs)), "total": len(logs)})



# ─────────────────────────────────────────────────────────────────────────────
# Admin — Bus Fees CRUD
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/admin/busfees/list")
async def admin_busfees_list(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    return JSONResponse({"bus_fees": _BUS_FEES, "academic_year": _BUS_YEAR})

@app.post("/admin/busfees/save")
async def admin_busfees_save(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    global _BUS_FEES, _BUS_YEAR
    body = await request.json()
    _BUS_FEES = body.get("bus_fees", _BUS_FEES)
    _BUS_YEAR = body.get("academic_year", _BUS_YEAR)
    data = {"institution":"N.B.K.R. Institute of Science and Technology, Vidyanagar",
            "academic_year": _BUS_YEAR, "bus_fees": _BUS_FEES}
    with open("nbkr_bus_fees.json","w",encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    _audit_log("BUSFEES_SAVE")
    return JSONResponse({"status":"saved","routes":len(_BUS_FEES)})

# ─────────────────────────────────────────────────────────────────────────────
# Admin — Placements CRUD
# ─────────────────────────────────────────────────────────────────────────────
_PLACEMENTS: List[Dict] = []

def load_placements():
    global _PLACEMENTS
    if os.path.exists("nbkr_placements.json"):
        with open("nbkr_placements.json","r",encoding="utf-8") as f:
            _PLACEMENTS = json.load(f)
        print(f"✓ Placements loaded: {len(_PLACEMENTS)} records")

@app.get("/admin/placements/list")
async def admin_placements_list(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    return JSONResponse({"placements":_PLACEMENTS,"total":len(_PLACEMENTS)})

@app.post("/admin/placements/add")
async def admin_placements_add(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    if not body.get("company"): return JSONResponse({"error":"company required"},status_code=400)
    body.setdefault("id", f"PL-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    _PLACEMENTS.append(body)
    with open("nbkr_placements.json","w",encoding="utf-8") as f:
        json.dump(_PLACEMENTS, f, indent=2, ensure_ascii=False)
    _audit_log(f"PLACEMENT_ADD:{body['company']}")
    return JSONResponse({"status":"added","total":len(_PLACEMENTS)})

@app.put("/admin/placements/update")
async def admin_placements_update(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    pid = body.get("id","")
    idx = next((i for i,p in enumerate(_PLACEMENTS) if p.get("id")==pid), None)
    if idx is None: return JSONResponse({"error":"Not found"},status_code=404)
    _PLACEMENTS[idx].update(body)
    with open("nbkr_placements.json","w",encoding="utf-8") as f:
        json.dump(_PLACEMENTS, f, indent=2, ensure_ascii=False)
    _audit_log(f"PLACEMENT_UPDATE:{pid}")
    return JSONResponse({"status":"updated"})

@app.delete("/admin/placements/delete")
async def admin_placements_delete(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    pid = body.get("id","")
    idx = next((i for i,p in enumerate(_PLACEMENTS) if p.get("id")==pid), None)
    if idx is None: return JSONResponse({"error":"Not found"},status_code=404)
    _PLACEMENTS.pop(idx)
    with open("nbkr_placements.json","w",encoding="utf-8") as f:
        json.dump(_PLACEMENTS, f, indent=2, ensure_ascii=False)
    _audit_log(f"PLACEMENT_DELETE:{pid}")
    return JSONResponse({"status":"deleted","total":len(_PLACEMENTS)})

# ─────────────────────────────────────────────────────────────────────────────
# Admin — Events CRUD
# ─────────────────────────────────────────────────────────────────────────────
_EVENTS: List[Dict] = []

def load_events():
    global _EVENTS
    if os.path.exists("nbkr_events.json"):
        with open("nbkr_events.json","r",encoding="utf-8") as f:
            _EVENTS = json.load(f)
        print(f"✓ Events loaded: {len(_EVENTS)} records")

@app.get("/admin/events/list")
async def admin_events_list(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    return JSONResponse({"events":_EVENTS,"total":len(_EVENTS)})

@app.post("/admin/events/add")
async def admin_events_add(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    if not body.get("title"): return JSONResponse({"error":"title required"},status_code=400)
    body.setdefault("id", f"EV-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    _EVENTS.append(body)
    with open("nbkr_events.json","w",encoding="utf-8") as f:
        json.dump(_EVENTS, f, indent=2, ensure_ascii=False)
    _audit_log(f"EVENT_ADD:{body['title']}")
    return JSONResponse({"status":"added","total":len(_EVENTS)})

@app.put("/admin/events/update")
async def admin_events_update(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    eid = body.get("id","")
    idx = next((i for i,e in enumerate(_EVENTS) if e.get("id")==eid), None)
    if idx is None: return JSONResponse({"error":"Not found"},status_code=404)
    _EVENTS[idx].update(body)
    with open("nbkr_events.json","w",encoding="utf-8") as f:
        json.dump(_EVENTS, f, indent=2, ensure_ascii=False)
    _audit_log(f"EVENT_UPDATE:{eid}")
    return JSONResponse({"status":"updated"})

@app.delete("/admin/events/delete")
async def admin_events_delete(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    eid = body.get("id","")
    idx = next((i for i,e in enumerate(_EVENTS) if e.get("id")==eid), None)
    if idx is None: return JSONResponse({"error":"Not found"},status_code=404)
    _EVENTS.pop(idx)
    with open("nbkr_events.json","w",encoding="utf-8") as f:
        json.dump(_EVENTS, f, indent=2, ensure_ascii=False)
    _audit_log(f"EVENT_DELETE:{eid}")
    return JSONResponse({"status":"deleted","total":len(_EVENTS)})

# ─────────────────────────────────────────────────────────────────────────────
# Admin — Department Info CRUD
# ─────────────────────────────────────────────────────────────────────────────
_DEPT_INFO: Dict = {}

def load_dept_info():
    global _DEPT_INFO
    if os.path.exists("nbkr_department_info.json"):
        with open("nbkr_department_info.json","r",encoding="utf-8") as f:
            _DEPT_INFO = json.load(f)
        print("✓ Department info loaded")


def build_department_info_card() -> str:
    if not _DEPT_INFO:
        return "I couldn't find that information in the uploaded NBKRIST knowledge base."

    dept = _DEPT_INFO.get("department", {})
    overview = dept.get("overview", {})
    programmes = dept.get("programmes", [])
    highlights = dept.get("highlights", [])
    strengths = dept.get("strengths", {})

    ai_ds = next(
        (p for p in programmes if "artificial intelligence" in p.get("branch", "").lower() or "ai" in p.get("branch", "").lower()),
        {}
    )
    objective_items = "".join(
        f'<li style="margin:3px 0">{item}</li>'
        for item in overview.get("objectives", [])[:3]
    )
    highlight_items = "".join(
        f'<li style="margin:3px 0">{item}</li>'
        for item in highlights[:3]
    )
    strength_text = " ".join(str(v) for v in strengths.values())

    return _info_card("AI & DS Branch Information", [
        ("College", _DEPT_INFO.get("collegeName", "NBKR IST")),
        ("Department", dept.get("name", "Department of Artificial Intelligence & Data Science")),
        ("About", overview.get("description") or _DEPT_INFO.get("about", "")),
        ("AI & DS Intake", str(ai_ds.get("intake", "240"))),
        ("Objectives", f'<ul style="margin:0;padding-left:18px">{objective_items}</ul>'),
        ("Highlights", f'<ul style="margin:0;padding-left:18px">{highlight_items}</ul>'),
        ("Strengths", strength_text),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# College Info — load & respond
# ─────────────────────────────────────────────────────────────────────────────
_COLLEGE_INFO: Dict = {}

def load_college_info():
    global _COLLEGE_INFO
    if os.path.exists("nbkr_college_info.json"):
        with open("nbkr_college_info.json", "r", encoding="utf-8") as f:
            _COLLEGE_INFO = json.load(f)
        print("✓ College info loaded")
    else:
        print("⚠ nbkr_college_info.json not found")


def _pill(text, bg="#e8eaf6", fg="#1a237e"):
    return (f'<span style="display:inline-block;background:{bg};color:{fg};'
            f'padding:2px 9px;border-radius:10px;font-size:11.5px;'
            f'font-weight:600;margin:2px 3px 2px 0">{text}</span>')


def _section_title(icon, title):
    return (f'<div style="padding:8px 14px 4px;font-size:11px;font-weight:700;'
            f'color:#888;text-transform:uppercase;letter-spacing:.05em;'
            f'border-top:1px solid #f0f0f0;margin-top:2px">{icon} {title}</div>')


def _kv_row(label, value, hi=False):
    bg = "background:#f8f9ff;" if hi else ""
    return (f'<div style="display:flex;align-items:flex-start;gap:8px;'
            f'padding:8px 14px;{bg}">'
            f'<span style="min-width:150px;font-size:11px;font-weight:700;'
            f'color:#888;text-transform:uppercase;letter-spacing:.03em;'
            f'padding-top:1px;flex-shrink:0">{label}</span>'
            f'<span style="font-size:13px;color:#222;line-height:1.6;'
            f'word-break:break-word">{value}</span></div>')


def build_college_info_card(section: str = "overview") -> str:
    """
    Build a rich HTML card for any section of college info.
    section: overview | programs | campus | placements | department |
             events | skills | careers | labs | chapters
    """
    global _COLLEGE_INFO
    if not _COLLEGE_INFO:
        load_college_info()
    ci = _COLLEGE_INFO
    if not ci:
        return '<p style="font-family:Segoe UI,sans-serif;color:#c62828">College info not loaded.</p>'

    col  = ci.get("college", {})
    dept = ci.get("department", {})
    camp = ci.get("campus", {})
    plac = ci.get("placements", {})
    prog = ci.get("programs_offered", {})
    evts = ci.get("events_and_activities", [])
    chap = ci.get("student_chapters", [])

    s = section.lower()

    # ── Overview ─────────────────────────────────────────────────────────────
    if s in ("overview", "about", "college", "nbkr", "general", "info"):
        loc  = col.get("location", {})
        acc  = col.get("accreditation", {})
        naac = acc.get("naac", {}).get("grade", "—")
        nba  = acc.get("nba", {}).get("type", "—")
        body = (
            _kv_row("Full Name",    col.get("name","—"), hi=True)
          + _kv_row("Established",  str(col.get("established","—")))
          + _kv_row("Location",     f'{loc.get("village","")}, {loc.get("district","")}, {loc.get("state","")}', hi=True)
          + _kv_row("Affiliation",  col.get("affiliation","—"))
          + _kv_row("Autonomous",   "✅ Yes" if col.get("autonomous_status") else "No", hi=True)
          + _kv_row("NAAC Grade",   _pill(f"Grade {naac}", "#e8f5e9", "#1b5e20"))
          + _kv_row("NBA",          _pill(nba, "#e3f2fd", "#0d47a1"), hi=True)
          + _kv_row("UGC Status",   col.get("approvals",{}).get("ugc_recognition","—"))
          + _section_title("🎯","Vision")
          + f'<div style="padding:8px 14px 12px;font-size:13px;color:#444;'
            f'font-style:italic;line-height:1.7">{col.get("vision","")}</div>'
        )
        title = f"🏛️ {col.get('name','NBKRIST')}"

    # ── Programs ─────────────────────────────────────────────────────────────
    elif s in ("programs", "courses", "ug", "pg", "undergraduate", "postgraduate"):
        ug_pills = "".join(_pill(p, "#e3f2fd", "#0d47a1") for p in prog.get("undergraduate", []))
        pg_pills = "".join(_pill(p, "#f3e5f5", "#4a148c") for p in prog.get("postgraduate", []))
        body = (
            _section_title("🎓", "Undergraduate Programs")
          + f'<div style="padding:6px 14px 10px">{ug_pills}</div>'
          + _section_title("📚", "Postgraduate Programs")
          + f'<div style="padding:6px 14px 10px">{pg_pills}</div>'
        )
        title = "📋 Programs Offered at NBKRIST"

    # ── Campus / Facilities ───────────────────────────────────────────────────
    elif s in ("campus", "facilities", "infrastructure", "hostel", "library"):
        fac_pills = "".join(_pill(f, "#e8f5e9", "#1b5e20") for f in camp.get("facilities", []))
        body = (
            _kv_row("Campus Area", camp.get("campus_area","—"), hi=True)
          + _section_title("🏢","Facilities")
          + f'<div style="padding:6px 14px 12px">{fac_pills}</div>'
        )
        title = "🏫 Campus & Facilities"

    # ── Placements ───────────────────────────────────────────────────────────
    elif s in ("placements", "recruiters", "companies", "jobs", "hiring"):
        rec_pills  = "".join(_pill(r, "#fff3e0", "#e65100") for r in plac.get("major_recruiters", []))
        focus_pills= "".join(_pill(f, "#e8eaf6", "#1a237e") for f in plac.get("placement_focus", []))
        body = (
            _section_title("🏢","Major Recruiters")
          + f'<div style="padding:6px 14px 10px">{rec_pills}</div>'
          + _section_title("🎯","Placement Focus Areas")
          + f'<div style="padding:6px 14px 12px">{focus_pills}</div>'
        )
        title = "💼 Placements & Recruiters"

    # ── Department Overview ───────────────────────────────────────────────────
    elif s in ("department", "ai&ds", "aids", "dept"):
        obj_items = "".join(
            f'<div style="display:flex;gap:8px;padding:3px 0">'
            f'<span style="color:#1a237e;flex-shrink:0">▸</span>'
            f'<span style="font-size:12.5px;color:#333">{o}</span></div>'
            for o in dept.get("objectives", [])
        )
        body = (
            _kv_row("Department", dept.get("name","—"), hi=True)
          + _kv_row("Intake",     str(dept.get("approved_intake","—")))
          + _kv_row("Overview",   dept.get("overview","—"), hi=True)
          + _section_title("🎯","Objectives")
          + f'<div style="padding:6px 14px 12px">{obj_items}</div>'
        )
        title = f"🤖 AI & DS Department"

    # ── Subjects ─────────────────────────────────────────────────────────────
    elif s in ("subjects", "syllabus", "curriculum dept"):
        sub_pills = "".join(_pill(s, "#e3f2fd", "#0d47a1") for s in dept.get("subjects", []))
        body = (
            _section_title("📖","Subjects Taught in AI & DS")
          + f'<div style="padding:6px 14px 12px">{sub_pills}</div>'
        )
        title = "📖 AI & DS — Subjects"

    # ── Labs ─────────────────────────────────────────────────────────────────
    elif s in ("labs", "laboratories", "lab"):
        lab_pills = "".join(_pill(l, "#e8f5e9", "#1b5e20") for l in dept.get("laboratories", []))
        body = (
            _section_title("🔬","Laboratories in AI & DS")
          + f'<div style="padding:6px 14px 12px">{lab_pills}</div>'
        )
        title = "🔬 AI & DS — Laboratories"

    # ── Career Opportunities ─────────────────────────────────────────────────
    elif s in ("careers", "career", "jobs", "opportunities", "roles"):
        car_pills = "".join(_pill(c, "#fce4ec", "#880e4f") for c in dept.get("career_opportunities", []))
        body = (
            _section_title("💼","Career Opportunities for AI & DS Graduates")
          + f'<div style="padding:6px 14px 12px">{car_pills}</div>'
        )
        title = "💼 Career Opportunities"

    # ── Recommended Skills ────────────────────────────────────────────────────
    elif s in ("skills", "recommended", "technologies", "tech stack", "stack"):
        sk_pills = "".join(_pill(sk, "#e8eaf6", "#1a237e") for sk in dept.get("recommended_skills", []))
        body = (
            _section_title("🛠️","Recommended Skills for AI & DS Students")
          + f'<div style="padding:6px 14px 12px">{sk_pills}</div>'
        )
        title = "🛠️ Recommended Skills"

    # ── Events ───────────────────────────────────────────────────────────────
    elif s in ("events", "activities", "fest", "techvyuha", "inspiron", "hackathon"):
        ev_pills = "".join(_pill(e, "#fff8e1", "#f57f17") for e in evts)
        body = (
            _section_title("🎉","Events & Activities")
          + f'<div style="padding:6px 14px 12px">{ev_pills}</div>'
        )
        title = "🎉 Events & Activities"

    # ── Student Chapters ─────────────────────────────────────────────────────
    elif s in ("chapters", "clubs", "student chapters", "ieee", "nss", "ncc"):
        ch_pills = "".join(_pill(c, "#f3e5f5", "#4a148c") for c in chap)
        body = (
            _section_title("👥","Student Chapters & Clubs")
          + f'<div style="padding:6px 14px 12px">{ch_pills}</div>'
        )
        title = "👥 Student Chapters & Clubs"

    # ── Default: full overview ────────────────────────────────────────────────
    else:
        return build_college_info_card("overview")

    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',Arial,sans-serif;max-width:520px">
  <div style="background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
              padding:11px 16px;border-radius:10px 10px 0 0">
    <span style="font-size:14px;font-weight:700">{title}</span>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;
              border-radius:0 0 10px 10px;overflow:hidden">
    {body}
  </div>
</div>"""


def handle_college_query(query: str) -> str:
    """Route college info queries to the right section card."""
    q = query.lower()

    def _has_word(words: List[str]) -> bool:
        for w in words:
            if w == "engineer":
                if re.search(r'\bengineers?\b', q):
                    return True
            elif w == "compan":
                if "company" in q or "companies" in q or "compan" in q:
                    return True
            elif w == "opportunit":
                if "opportunity" in q or "opportunities" in q or "opportunit" in q:
                    return True
            elif w == "facilit":
                if "facility" in q or "facilities" in q or "facilit" in q:
                    return True
            elif w == "laborator":
                if "laboratory" in q or "laboratories" in q or "laborator" in q:
                    return True
            elif w in ["b.tech", "m.tech", "ai&ds", "ai ds"]:
                if w in q:
                    return True
            else:
                if re.search(rf'\b{w}s?\b', q):
                    return True
        return False

    if _has_word(["program","course","undergraduate","postgraduate","ug","pg","b.tech","btech","m.tech","mba","mca"]):
        return build_college_info_card("programs")
    if _has_word(["campus","facilit","infrastructure","hostel","library","gym","canteen","cafeteria","wi-fi","wifi","atm","bank"]):
        return build_college_info_card("campus")
    if _has_word(["placement","recruiter","company","compan","hiring","package","tcs","infosys","wipro","cognizant"]):
        return build_college_info_card("placements")
    if _has_word(["subject","syllabus","what do we study","what subjects"]):
        return build_college_info_card("subjects")
    if _has_word(["lab","laborator"]):
        return build_college_info_card("labs")
    if _has_word(["career","job","opportunit","role","engineer","scientist","analyst","developer"]):
        return build_college_info_card("careers")
    if _has_word(["skill","technolog","stack","python","react","docker","aws","langchain","rag","mcp"]):
        return build_college_info_card("skills")
    if _has_word(["event","activit","techvyuha","inspiron","hackathon","symposium","workshop","fest"]):
        return build_college_info_card("events")
    if _has_word(["chapter","club","ieee","nss","ncc","csi","iste","coding club","cultural"]):
        return build_college_info_card("chapters")
    if _has_word(["department","ai&ds","aids","ai ds","artificial intelligence data science","dept overview","dept objective"]):
        return build_college_info_card("department")
    # Default: college overview
    return build_college_info_card("overview")


# ─────────────────────────────────────────────────────────────────────────────
# R23 Labs Knowledge Base — load & respond
# ─────────────────────────────────────────────────────────────────────────────
_LABS_R23: Dict = {}

def load_labs_r23():
    global _LABS_R23
    if os.path.exists("nbkr_labs_r23.json"):
        with open("nbkr_labs_r23.json", "r", encoding="utf-8") as f:
            _LABS_R23 = json.load(f)
        total = sum(len(s.get("labs", [])) for s in _LABS_R23.get("labs_by_year", []))
        print(f"✓ R23 Labs loaded: {total} labs across {len(_LABS_R23.get('labs_by_year',[]))} semesters")
    else:
        print("⚠ nbkr_labs_r23.json not found")


# ── Abbreviation / typo alias map ────────────────────────────────────────────
_LAB_ALIASES = {
    # Physics
    "phy lab": "Engineering Physics Lab",
    "physics lab": "Engineering Physics Lab",
    "ep lab": "Engineering Physics Lab",
    # Programming / CP
    "cp lab": "Computer Programming Lab",
    "c programming lab": "Computer Programming Lab",
    "programming lab": "Computer Programming Lab",
    "comp prog lab": "Computer Programming Lab",
    # EEE Workshop
    "eee ws": "Electrical & Electronics Engineering Workshop",
    "eee workshop": "Electrical & Electronics Engineering Workshop",
    "electrical lab": "Electrical & Electronics Engineering Workshop",
    # IT Workshop
    "it ws": "IT Workshop",
    "it workshop": "IT Workshop",
    # Chemistry
    "chem lab": "Chemistry Lab",
    "chemistry lab": "Chemistry Lab",
    # English
    "english lab": "Communicative English Lab",
    "comm english": "Communicative English Lab",
    "communicative english": "Communicative English Lab",
    # DS Lab
    "ds lab": "Data Structures Lab",
    "data structure lab": "Data Structures Lab",
    "data strcture lab": "Data Structures Lab",
    "dsa lab": "Data Structures Lab",
    # Engineering Workshop
    "eng workshop": "Engineering Workshop",
    "engineering ws": "Engineering Workshop",
    # Advanced DSA
    "adsa lab": "Advanced Data Structures & Algorithm Analysis Lab",
    "advanced ds lab": "Advanced Data Structures & Algorithm Analysis Lab",
    "advanced dsa": "Advanced Data Structures & Algorithm Analysis Lab",
    "algo lab": "Advanced Data Structures & Algorithm Analysis Lab",
    # OOP Java
    "oop lab": "Object Oriented Programming Through Java Lab",
    "java lab": "Object Oriented Programming Through Java Lab",
    "oop java": "Object Oriented Programming Through Java Lab",
    "oops lab": "Object Oriented Programming Through Java Lab",
    "oop through java": "Object Oriented Programming Through Java Lab",
    # AI Lab
    "ai lab": "Artificial Intelligence Lab",
    "artificial intelligence lab": "Artificial Intelligence Lab",
    # DS Python
    "ds python lab": "Data Science Using Python Lab",
    "data science lab": "Data Science Using Python Lab",
    "python lab": "Data Science Using Python Lab",
    # FSD I
    "fsd 1": "Full Stack Development - I",
    "fsd-1": "Full Stack Development - I",
    "fsd i": "Full Stack Development - I",
    "full stack 1": "Full Stack Development - I",
    "full stack development 1": "Full Stack Development - I",
    # DWM Lab
    "dwdm lab": "Data Warehousing & Mining Lab",
    "data mining lab": "Data Warehousing & Mining Lab",
    "data warehousing lab": "Data Warehousing & Mining Lab",
    "dm lab": "Data Warehousing & Mining Lab",
    # ML Lab
    "ml lab": "Machine Learning Lab",
    "machine learning lab": "Machine Learning Lab",
    # Tinkering
    "tinkering lab": "Tinkering Lab",
    # FSD II
    "fsd 2": "Full Stack Development - II",
    "fsd-2": "Full Stack Development - II",
    "fsd ii": "Full Stack Development - II",
    "full stack 2": "Full Stack Development - II",
    "full stack development 2": "Full Stack Development - II",
    # DL Lab
    "dl lab": "Deep Learning Lab",
    "deep learning lab": "Deep Learning Lab",
    # BDA Lab
    "bda lab": "Big Data Analytics & Data Visualization Lab",
    "big data lab": "Big Data Analytics & Data Visualization Lab",
    "data visualization lab": "Big Data Analytics & Data Visualization Lab",
    "bd dv lab": "Big Data Analytics & Data Visualization Lab",
}

# ── Semester code normalization ───────────────────────────────────────────────
_SEM_CODE_MAP = {
    # I Year Sem I
    "i-i": "I-I",   "1-1": "I-I",   "i year sem 1": "I-I",   "1st year sem 1": "I-I",
    "1st year semester 1": "I-I", "i year semester i": "I-I", "sem 1 1st year": "I-I",
    "first year first semester": "I-I", "first year sem 1": "I-I",
    # I Year Sem II
    "i-ii": "I-II", "1-2": "I-II",  "i year sem 2": "I-II",  "1st year sem 2": "I-II",
    "1st year semester 2": "I-II", "first year second semester": "I-II", "first year sem 2": "I-II",
    # II Year Sem I
    "ii-i": "II-I", "2-1": "II-I",  "ii year sem 1": "II-I", "2nd year sem 1": "II-I",
    "2nd year semester 1": "II-I", "second year sem 1": "II-I", "second year first semester": "II-I",
    # II Year Sem II
    "ii-ii": "II-II","2-2": "II-II", "ii year sem 2": "II-II","2nd year sem 2": "II-II",
    "2nd year semester 2": "II-II", "second year sem 2": "II-II", "second year second semester": "II-II",
    # III Year Sem I
    "iii-i": "III-I","3-1": "III-I", "iii year sem 1": "III-I","3rd year sem 1": "III-I",
    "3rd year semester 1": "III-I", "third year sem 1": "III-I", "third year first semester": "III-I",
    # III Year Sem II
    "iii-ii": "III-II","3-2": "III-II","iii year sem 2": "III-II","3rd year sem 2": "III-II",
    "3rd year semester 2": "III-II", "third year sem 2": "III-II", "third year second semester": "III-II",
    # IV Year Sem I
    "iv-i": "IV-I", "4-1": "IV-I",  "iv year sem 1": "IV-I", "4th year sem 1": "IV-I",
    "4th year semester 1": "IV-I", "fourth year sem 1": "IV-I", "fourth year first semester": "IV-I",
}

# ── Year keyword map ──────────────────────────────────────────────────────────
_YEAR_MAP = {
    "1st year": "I Year", "i year": "I Year",  "first year": "I Year",  "year 1": "I Year",  "1 year": "I Year",
    "2nd year": "II Year","ii year": "II Year","second year": "II Year","year 2": "II Year", "2 year": "II Year",
    "3rd year": "III Year","iii year": "III Year","third year": "III Year","year 3": "III Year","3 year": "III Year",
    "4th year": "IV Year","iv year": "IV Year","fourth year": "IV Year","year 4": "IV Year", "4 year": "IV Year",
}

_TYPE_ICONS = {"Laboratory": "🧪", "Workshop": "🔧", "Skill Course": "💻"}
_TYPE_COLORS = {
    "Laboratory":   ("#e8f5e9", "#1b5e20"),
    "Workshop":     ("#fff3e0", "#e65100"),
    "Skill Course": ("#e3f2fd", "#0d47a1"),
}


def _labs_badge(lab: Dict) -> str:
    t    = lab.get("type", "Laboratory")
    icon = _TYPE_ICONS.get(t, "📋")
    bg, fg = _TYPE_COLORS.get(t, ("#e8eaf6", "#1a237e"))
    name = lab.get("name", "—")
    sn   = lab.get("short_name", "")
    display = f'{name} <span style="font-size:10px;opacity:.75">({sn})</span>' if sn else name
    return (f'<div style="display:flex;align-items:center;gap:8px;'
            f'padding:7px 12px;border-bottom:1px solid #f0f0f0">'
            f'<span style="font-size:15px">{icon}</span>'
            f'<div style="flex:1">'
            f'<span style="font-size:13px;color:#222;font-weight:600">{display}</span>'
            f'</div>'
            f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:8px;font-size:10px;font-weight:700;white-space:nowrap">{t}</span>'
            f'</div>')


def build_labs_card(sem_data: Dict) -> str:
    """Build a card for a single semester's labs."""
    year = sem_data.get("year", "—")
    sem  = sem_data.get("semester", "—")
    code = sem_data.get("semester_code", "—")
    labs = sem_data.get("labs", [])
    note = sem_data.get("note", "")

    if not labs:
        body = (f'<div style="padding:14px 16px;font-size:13px;color:#e65100;'
                f'background:#fff8e1;border-radius:6px;margin:10px 14px">'
                f'⚠️ {note}</div>')
    else:
        items = "".join(_labs_badge(l) for l in labs)
        # Legend
        types_present = list(dict.fromkeys(l.get("type","Laboratory") for l in labs))
        legend = " ".join(
            f'<span style="font-size:10px;margin-right:8px">'
            f'{_TYPE_ICONS.get(t,"")} {t}</span>'
            for t in types_present
        )
        body = (f'<div style="padding:6px 14px 4px;font-size:10.5px;color:#888">'
                f'{legend}</div>' + items)

    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',Arial,sans-serif;max-width:520px">
  <div style="background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
              padding:10px 16px;border-radius:10px 10px 0 0;
              display:flex;justify-content:space-between;align-items:center">
    <span style="font-size:14px;font-weight:700">🔬 {year} — {sem}</span>
    <span style="background:rgba(255,255,255,.2);padding:2px 9px;
                 border-radius:8px;font-size:11px;font-weight:700">{code}</span>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;
              border-radius:0 0 10px 10px;overflow:hidden">
    {body}
  </div>
  <div style="font-size:10.5px;color:#888;margin-top:4px;padding:0 4px">
    R23 Regulation · AI&DS · B.Tech
  </div>
</div>"""


def build_all_labs_card() -> str:
    """Build a full year-wise grouped labs overview card."""
    semesters = _LABS_R23.get("labs_by_year", [])
    if not semesters:
        return '<p style="font-family:Segoe UI,sans-serif;color:#c62828">Labs data not loaded.</p>'

    # Group by year
    from collections import OrderedDict
    years: dict = OrderedDict()
    for s in semesters:
        yr = s.get("year","—")
        years.setdefault(yr, []).append(s)

    year_grads = {
        "I Year":   "linear-gradient(135deg,#1a237e,#3949ab)",
        "II Year":  "linear-gradient(135deg,#1b5e20,#388e3c)",
        "III Year": "linear-gradient(135deg,#bf360c,#e64a19)",
        "IV Year":  "linear-gradient(135deg,#4a148c,#7b1fa2)",
    }

    html = ""
    for yr, sems in years.items():
        grad = year_grads.get(yr, "linear-gradient(135deg,#2d3a6b,#4a5568)")
        sem_blocks = ""
        for s in sems:
            code = s.get("semester_code","")
            sem_name = s.get("semester","")
            labs = s.get("labs", [])
            note = s.get("note","")

            if not labs:
                content = (f'<div style="padding:6px 12px;font-size:12px;'
                           f'color:#e65100;font-style:italic">'
                           f'⚠️ {note}</div>')
            else:
                items = ""
                for l in labs:
                    t    = l.get("type","Laboratory")
                    icon = _TYPE_ICONS.get(t,"📋")
                    bg, fg = _TYPE_COLORS.get(t, ("#e8eaf6","#1a237e"))
                    items += (
                        f'<div style="display:flex;align-items:center;gap:7px;'
                        f'padding:5px 12px;border-bottom:1px solid #f5f5f5">'
                        f'<span>{icon}</span>'
                        f'<span style="flex:1;font-size:12.5px;color:#222">{l.get("name","—")}</span>'
                        f'<span style="background:{bg};color:{fg};padding:1px 7px;'
                        f'border-radius:7px;font-size:10px;font-weight:700">{t}</span>'
                        f'</div>'
                    )
                content = items

            sem_blocks += f"""
<div style="margin:6px 0">
  <div style="padding:5px 12px;background:#f5f7ff;font-size:11px;
              font-weight:700;color:#555;border-left:3px solid #3949ab">
    {sem_name} &nbsp;·&nbsp; <span style="color:#1a237e">{code}</span>
    &nbsp;·&nbsp; {len(labs)} lab(s)
  </div>
  {content}
</div>"""

        html += f"""
<div style="margin-bottom:10px">
  <div style="background:{grad};color:#fff;padding:9px 14px;
              border-radius:8px 8px 0 0;font-size:13px;font-weight:700">
    📅 {yr}
  </div>
  <div style="background:#fff;border:1px solid #ddd;border-top:none;
              border-radius:0 0 8px 8px;overflow:hidden">
    {sem_blocks}
  </div>
</div>"""

    # Legend
    legend = "".join(
        f'<span style="margin-right:12px;font-size:11px">'
        f'{icon} <b>{t}</b></span>'
        for t, icon in _TYPE_ICONS.items()
    )

    return f"""
<div style="margin:8px 0;font-family:'Segoe UI',Arial,sans-serif;max-width:520px">
  <div style="background:linear-gradient(135deg,#2d3a6b,#4a5568);color:#fff;
              padding:11px 16px;border-radius:10px 10px 0 0;
              display:flex;justify-content:space-between;align-items:center">
    <span style="font-size:14px;font-weight:700">🔬 AI&DS Labs — R23 Regulation</span>
    <span style="font-size:10px;opacity:.8">B.Tech All Years</span>
  </div>
  <div style="background:#f9f9f9;border:1px solid #ddd;border-top:none;
              padding:10px;border-radius:0 0 10px 10px">
    {html}
  </div>
  <div style="margin-top:6px;padding:6px 12px;background:#f0f4ff;
              border-radius:8px;font-size:11px;color:#555">
    {legend}
  </div>
</div>"""


def _normalize_sem_query(q: str) -> str:
    """Return canonical semester code like 'II-I' or '' if not found."""
    q = q.strip().lower()
    # Direct match
    if q in _SEM_CODE_MAP:
        return _SEM_CODE_MAP[q]
    # Substring match (longest first)
    for alias in sorted(_SEM_CODE_MAP.keys(), key=len, reverse=True):
        if alias in q:
            return _SEM_CODE_MAP[alias]
    return ""


def _normalize_year_query(q: str) -> str:
    """Return canonical year like 'II Year' or '' if not found."""
    for alias in sorted(_YEAR_MAP.keys(), key=len, reverse=True):
        if alias in q:
            return _YEAR_MAP[alias]
    return ""


def handle_labs_query(query: str) -> Optional[str]:
    """
    Handle all lab-related queries with typo/abbreviation tolerance.
    Returns HTML string or None if not a labs query.
    """
    q = query.lower().strip()

    labs_kw = {
        "lab", "labs", "laboratory", "laboratories", "workshop", "workshops",
        "practical", "practicals", "skill course", "skill courses",
        "hands on", "hands-on", "lab subjects", "lab list",
        "which labs", "what labs", "list of labs", "all labs",
        "r23 lab", "r23 labs", "aids lab", "ai&ds lab",
        "lab in", "labs in", "labs for", "lab for",
        "sem labs", "semester labs", "year labs",
    }

    # Also check alias map
    alias_match = any(alias in q for alias in _LAB_ALIASES)

    if not any(kw in q for kw in labs_kw) and not alias_match:
        return None

    semesters = _LABS_R23.get("labs_by_year", [])
    if not semesters:
        return '<p style="font-family:Segoe UI,sans-serif;color:#c62828">Labs data not loaded.</p>'

    # ── 1. Specific lab name query (alias match) ──────────────────────────
    if alias_match:
        matched_name = next(
            (full for alias, full in _LAB_ALIASES.items() if alias in q), None
        )
        if matched_name:
            # Find which semester it belongs to
            found = []
            for s in semesters:
                for l in s.get("labs", []):
                    if matched_name.lower() in l.get("name","").lower():
                        found.append((s, l))
            if found:
                s, l = found[0]
                t    = l.get("type","Laboratory")
                icon = _TYPE_ICONS.get(t,"📋")
                bg, fg = _TYPE_COLORS.get(t, ("#e8eaf6","#1a237e"))
                return f"""
<div style="margin:8px 0;font-family:'Segoe UI',Arial,sans-serif;max-width:480px">
  <div style="background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
              padding:10px 16px;border-radius:10px 10px 0 0">
    <span style="font-size:14px;font-weight:700">{icon} {l.get('name','—')}</span>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-top:none;
              border-radius:0 0 10px 10px;padding:12px 16px">
    <div style="margin-bottom:6px">
      <span style="background:{bg};color:{fg};padding:3px 10px;
                   border-radius:8px;font-size:12px;font-weight:700">{t}</span>
    </div>
    <div style="font-size:13px;color:#555">
      📅 <b>{s.get('year','—')}</b> · {s.get('semester','—')}
      &nbsp;(<b>{s.get('semester_code','—')}</b>)
    </div>
    <div style="font-size:11px;color:#888;margin-top:4px">R23 · AI&DS · B.Tech</div>
  </div>
</div>"""

    # ── 2. Specific semester code query ──────────────────────────────────
    sem_code = _normalize_sem_query(q)
    if sem_code:
        match = next((s for s in semesters if s.get("semester_code") == sem_code), None)
        if match:
            return build_labs_card(match)
        return (f'<p style="font-family:Segoe UI,sans-serif;color:#c62828">'
                f'No data found for semester <b>{sem_code}</b> in R23 AI&DS curriculum.</p>')

    # ── 3. Year-only query → show both semesters of that year ─────────────
    year_key = _normalize_year_query(q)
    if year_key:
        year_sems = [s for s in semesters if s.get("year") == year_key]
        if year_sems:
            cards = "".join(build_labs_card(s) for s in year_sems)
            return f'<div>{cards}</div>'

    # ── 4. "All labs" / general query → full overview ─────────────────────
    all_kw = {"all lab", "all labs", "complete", "full list", "every lab",
              "aids labs", "ai&ds labs", "r23 labs", "list of labs",
              "lab list", "show labs", "lab subjects"}
    if any(kw in q for kw in all_kw):
        return build_all_labs_card()

    # ── 5. Keyword-only "labs" with no year/sem → full overview ──────────
    return build_all_labs_card()

@app.get("/admin/deptinfo/get")
async def admin_deptinfo_get(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    return JSONResponse(_DEPT_INFO)

@app.post("/admin/deptinfo/save")
async def admin_deptinfo_save(request: Request):
    global _DEPT_INFO
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    _DEPT_INFO.update(body)
    with open("nbkr_department_info.json","w",encoding="utf-8") as f:
        json.dump(_DEPT_INFO, f, indent=2, ensure_ascii=False)
    initialize_rag()
    _audit_log("DEPTINFO_SAVE")
    return JSONResponse({"status":"saved"})

# ─────────────────────────────────────────────────────────────────────────────
# Admin — Subjects CRUD
# ─────────────────────────────────────────────────────────────────────────────
_SUBJECTS_LIST: List[Dict] = []

def load_subjects_list():
    global _SUBJECTS_LIST
    if os.path.exists("nbkr_subjects.json"):
        with open("nbkr_subjects.json","r",encoding="utf-8") as f:
            _SUBJECTS_LIST = json.load(f)
        print(f"✓ Subjects list loaded: {len(_SUBJECTS_LIST)} subjects")

@app.get("/admin/subjects/list")
async def admin_subjects_list(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    return JSONResponse({"subjects":_SUBJECTS_LIST,"total":len(_SUBJECTS_LIST)})

@app.post("/admin/subjects/add")
async def admin_subjects_add(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    if not body.get("name"): return JSONResponse({"error":"name required"},status_code=400)
    body.setdefault("id", f"SB-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    _SUBJECTS_LIST.append(body)
    with open("nbkr_subjects.json","w",encoding="utf-8") as f:
        json.dump(_SUBJECTS_LIST, f, indent=2, ensure_ascii=False)
    _audit_log(f"SUBJECT_ADD:{body['name']}")
    return JSONResponse({"status":"added","total":len(_SUBJECTS_LIST)})

@app.put("/admin/subjects/update")
async def admin_subjects_update(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    sid = body.get("id","")
    idx = next((i for i,s in enumerate(_SUBJECTS_LIST) if s.get("id")==sid), None)
    if idx is None: return JSONResponse({"error":"Not found"},status_code=404)
    _SUBJECTS_LIST[idx].update(body)
    with open("nbkr_subjects.json","w",encoding="utf-8") as f:
        json.dump(_SUBJECTS_LIST, f, indent=2, ensure_ascii=False)
    _audit_log(f"SUBJECT_UPDATE:{sid}")
    return JSONResponse({"status":"updated"})

@app.delete("/admin/subjects/delete")
async def admin_subjects_delete(request: Request):
    if not _check_admin(request): return JSONResponse({"error":"Unauthorized"},status_code=401)
    body = await request.json()
    sid = body.get("id","")
    idx = next((i for i,s in enumerate(_SUBJECTS_LIST) if s.get("id")==sid), None)
    if idx is None: return JSONResponse({"error":"Not found"},status_code=404)
    _SUBJECTS_LIST.pop(idx)
    with open("nbkr_subjects.json","w",encoding="utf-8") as f:
        json.dump(_SUBJECTS_LIST, f, indent=2, ensure_ascii=False)
    _audit_log(f"SUBJECT_DELETE:{sid}")
    return JSONResponse({"status":"deleted","total":len(_SUBJECTS_LIST)})


def _audit_log(action: str):
    if os.path.exists(audit_file):
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            entries = []

    entries.append({
        "ts": datetime.now().isoformat(),
        "user": "admin",
        "action": action,
    })

    # Keep last 200 entries
    entries = entries[-200:]
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


@app.get("/health")
async def health():
    return {
        "status":       "healthy",
        "version":      "7.0.0",
        "nlp_enabled":  nlp is not None,
        "ml_enabled":   ml_classifier is not None,
        "rag_enabled":  faiss_index is not None,
        "documents":    len(knowledge_docs),
        "students":     len(_students),
        "threshold":    CONFIDENCE_THRESHOLD,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Student CRUD REST API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/students")
async def api_get_all_students(branch: str = None, section: str = None):
    """GET /students  — list all students, optionally filter by branch or section."""
    data = _students
    if branch:
        data = [s for s in data if branch.upper() in s.get("branch","").upper()]
    if section:
        data = [s for s in data if s.get("section","").upper() == section.upper()]
    return {"count": len(data), "students": data}


@app.get("/students/{roll_no}")
async def api_get_student(roll_no: str):
    """GET /students/{roll_no}  — fetch a single student by roll number."""
    s = student_get(roll_no)
    if not s:
        return JSONResponse({"error": f"Student {roll_no.upper()} not found"}, status_code=404)
    return s


@app.post("/students")
async def api_create_student(request: Request):
    """POST /students  — add a new student record."""
    body = await request.json()
    ok, msg = student_create(body)
    if not ok:
        return JSONResponse({"error": msg}, status_code=400)
    _audit_log(f"STUDENT_CREATE:{body.get('roll_no','?').upper()}")
    return JSONResponse({"status": "created", "message": msg,
                         "student": student_get(body.get("roll_no","").upper())},
                        status_code=201)


@app.put("/students/{roll_no}")
async def api_update_student(roll_no: str, request: Request):
    """PUT /students/{roll_no}  — update fields of an existing student."""
    body = await request.json()
    ok, msg = student_update(roll_no, body)
    if not ok:
        return JSONResponse({"error": msg}, status_code=404)
    _audit_log(f"STUDENT_UPDATE:{roll_no.upper()}")
    return {"status": "updated", "message": msg,
            "student": student_get(roll_no.upper())}


@app.delete("/students/{roll_no}")
async def api_delete_student(roll_no: str):
    """DELETE /students/{roll_no}  — remove a student record."""
    ok, msg = student_delete(roll_no)
    if not ok:
        return JSONResponse({"error": msg}, status_code=404)
    _audit_log(f"STUDENT_DELETE:{roll_no.upper()}")
    return {"status": "deleted", "message": msg}


@app.get("/students/stats/summary")
async def api_student_stats():
    """GET /students/stats/summary  — aggregated analytics."""
    return _mine_stats()


if __name__ == "__main__":
    print("\n🚀 Starting NBKR RAG+NLP+ML Chatbot v7.0 …")
    print("📍 http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

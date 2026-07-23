import httpx
import re
from typing import Optional

GATEWAY_URL = "https://depweb-five.vercel.app/api/all-data"
NBKRIST_URL = "https://nbkrist.org"

def scrape_for_query(query: str) -> Optional[str]:
    try:
        return _fetch_gateway(query)
    except Exception as e:
        print(f"[scraper] Gateway error: {e}")
    try:
        return _fetch_nbkrist(query)
    except Exception as e:
        print(f"[scraper] NBKR error: {e}")
    return None

def _fetch_gateway(query: str) -> Optional[str]:
    import re
    from bs4 import BeautifulSoup
    
    r = httpx.get("https://depweb-five.vercel.app/all-data", timeout=10)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, 'html.parser')
    containers = soup.find_all('article', attrs={'data-id': True})

    query_lower = query.lower()
    stop_words = {
        "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
        "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot",
        "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each",
        "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd",
        "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
        "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
        "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
        "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
        "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
        "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
        "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
        "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
        "you're", "you've", "your", "yours", "yourself", "yourselves", "tell", "show", "find", "please"
    }
    query_words = [w for w in query_lower.split() if len(w) > 2 and w not in stop_words]
    food_signals = {"breakfast", "canteen", "food", "mess"}
    has_food_signal = any(s in query_lower for s in food_signals)

    best_score = 0
    best_record = None

    for c in containers:
        record_id = c.get('data-id', '').strip()
        
        # Extract title
        title_el = c.find('h3')
        title = title_el.get_text(strip=True) if title_el else ""
        
        # Extract description
        desc_div = c.find('div', class_=lambda cl: cl and 'font-sans' in cl and 'whitespace-pre-line' in cl)
        description = desc_div.get_text(strip=True) if desc_div else ""
        
        # Extract fields
        fields = []
        tbody = c.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) >= 2:
                    key = tds[0].get_text(strip=True)
                    val = tds[1].get_text(strip=True)
                    lbl = tds[2].get_text(strip=True) if len(tds) > 2 else ""
                    fields.append({"key": key, "value": val, "label": lbl})

        # Check hostel record match strictness
        is_hostel_record = (record_id.lower().startswith("hostel") or "hostel" in record_id.lower())
        if is_hostel_record:
            hostel_keywords = {"hostel", "accommodation", "warden", "fees"}
            if not any(k in query_lower for k in hostel_keywords):
                continue

        # Check canteen menu query strictness
        if "canteen" in query_lower:
            body_text = (title + " " + description + " " + " ".join(f['key'] + " " + f['value'] + " " + f['label'] for f in fields)).lower()
            if "canteen" not in record_id.lower() and "canteen" not in body_text:
                continue

        # Check relevance to food/canteen if has_food_signal is True
        if has_food_signal:
            food_related_words = {"breakfast", "canteen", "food", "mess", "lunch", "dinner", "dishes", "menu", "veg", "non-veg"}
            id_lower = record_id.lower()
            body_lower = (title + " " + description + " " + " ".join(f['key'] + " " + f['value'] + " " + f['label'] for f in fields)).lower()
            is_food_record = any(w in body_lower or w in id_lower for w in food_related_words)
            if not is_food_record:
                continue

        # Score the record based on query words
        body_lower = (title + " " + description + " " + " ".join(f['key'] + " " + f['value'] + " " + f['label'] for f in fields)).lower()
        score = 0
        if query_lower in body_lower:
            score += 10
        title_lower = title.lower()
        for w in query_words:
            if re.search(r'\b' + re.escape(w) + r'\b', title_lower):
                score += 5
            elif re.search(r'\b' + re.escape(w) + r'\b', body_lower):
                score += 1

        if score > best_score:
            best_score = score
            best_record = {
                "id": record_id,
                "title": title,
                "description": description,
                "fields": fields
            }

    if best_score > 0 and best_record:
        fields = best_record["fields"]
        desc = best_record["description"]
        title = best_record["title"]
        
        if len(fields) <= 2:
            # Format as a plain sentence
            parts = []
            if desc:
                parts.append(desc)
            else:
                if title:
                    parts.append(f"Information for {title}:")
            for f in fields:
                label = f["label"] or f["key"]
                val = f["value"]
                if val:
                    if val.lower() not in (desc or "").lower():
                        parts.append(f"The {label} is {val}.")
            res_sentence = " ".join(parts).strip()
            res_sentence = re.sub(r'\s+', ' ', res_sentence)
            res_sentence = re.sub(r'\.+', '.', res_sentence)
            return res_sentence
        else:
            # Format as an HTML table for multi-field data
            rows_html = []
            for f in fields:
                label = f["label"] or f["key"]
                val = f["value"]
                rows_html.append(f"""
<tr style="border-bottom:1px solid #e5e7eb;">
  <td style="padding:6px 8px;font-weight:bold;color:#4b5563;">{label}</td>
  <td style="padding:6px 8px;color:#111827;">{val}</td>
</tr>""")
            
            table_html = f"""
<div style="margin-bottom:8px;">
  <strong style="color:#111827;font-size:1.05em;">{title}</strong>
  <p style="color:#4b5563;font-size:0.95em;margin:4px 0 8px 0;">{desc}</p>
  <table style="width:100%;border-collapse:collapse;font-size:0.9em;margin-top:6px;">
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</div>"""
            return table_html

    return None

def _fetch_nbkrist(query: str) -> Optional[str]:
    r = httpx.get(NBKRIST_URL, timeout=10)
    r.raise_for_status()
    text = re.sub(r'<[^>]+>', ' ', r.text)
    text = re.sub(r'\s+', ' ', text)
    
    query_words = set(query.lower().split())
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 20]
    
    best_score = 0
    best_sentence = None
    for s in sentences:
        score = sum(1 for w in query_words if w in s.lower())
        if score > best_score:
            best_score = score
            best_sentence = s
    
    return f"[Live NBKR Site] {best_sentence}" if best_score > 0 else None

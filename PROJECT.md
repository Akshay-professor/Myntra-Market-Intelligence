# 🛍️ Myntra Market Intelligence Agent

A conversational retail-analytics app: ask in plain English to **shop products** or get
**market insights** over a ~100k-product Myntra catalog. Built with **Streamlit + Groq LLMs +
LangChain**, styled to look like myntra.com.

---

## What it does

- **Product search** — natural language like *"black slim-fit jeans under ₹1500"*,
  *"running shoes"*, or even a bare brand *"roadster"* → a Myntra-style **product grid**
  (image, brand, price, MRP, discount %, rating, buy link).
- **Market analytics** — *"top brands by discount"*, *"category pricing summary"*,
  *"rating distribution"* → answered by an LLM agent over pandas tools, plus a visual
  **Insights dashboard** (brand/category/rating/discount charts).
- **Intent-aware routing** — greetings, analytics, product search, and off-topic questions
  (*"who is India's PM?"*) are handled differently; off-topic is politely declined instead of
  returning random products.

---

## How it works (architecture)

```
User query
  └─ classifier.rule_intent()              # zero-token rules: greeting / analytics / product / unsure
       ├─ greeting   → canned welcome reply
       ├─ analytics  → agent.get_agent_response()   # LangChain ReAct + Groq 70B over pandas tools
       └─ else → agent.understand_query()           # cheap LLM extracts structured facts (JSON)
                   ├─ product_search → data_tools.structured_search()   # precise column filters
                   │                     └─ semantic_search (TF-IDF) fallback on a miss
                   └─ out_of_scope   → polite decline
```

The design is **cost-conscious**: greetings and obvious analytics are handled by rules with **zero
LLM tokens**; only genuine product/ambiguous queries spend a single cheap `llama-3.1-8b-instant`
call to extract structured facts. The heavy `llama-3.3-70b` model is reserved for the analytics
agent. If the API is unavailable, the app **degrades gracefully** to rule-based keyword search.

### File map

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI (Myntra theme, sticky nav bar, Shop/Insights tabs, product grid), routing, message rendering |
| `agent.py` | Groq LLMs — ReAct analytics agent, `classify_intent_llm`, `understand_query` (facts extraction), `apply_brand_override` |
| `classifier.py` | Rule fast-paths (greeting/analytics/product) + cheap LLM fallback |
| `data_tools.py` | Pandas analytics tools + `find_products` (keyword) + `structured_search` (fact-based) |
| `taxonomy.py` | Product-type rules, synonyms, `word_pattern` (whole-word, plural-tolerant matching) |
| `enrich.py` | Turns `product_url` + `product_name` into structured columns (type/gender/color/material/fit/pattern) |
| `semantic_search.py` | TF-IDF char-ngram fuzzy fallback (typo/loose-phrasing tolerance) |
| `preprocess_kaggle_data.py` | Offline cleaning + enrichment of the raw Kaggle dump |
| `tests/` | pytest suite — taxonomy, search, classifier, enrichment, brand |

---

## Data model & the core challenge

Source columns (11):

```
product_id, product_name, brand, category, original_price, discount_pct,
discounted_price, rating, num_reviews, image_url, product_url
```

Only **price, discount, rating, reviews, and brand** are truly structured. The things shoppers
filter on most — **item type, gender, color, material, fit, pattern** — live in *free text*.

**Key insight:** the `product_url` already encodes Myntra's full, structured product title:

```
https://www.myntra.com/{TYPE}/{BRAND}/{rich-descriptive-slug}/{id}/buy
   e.g. .../tshirts/roadster/roadster-men-navy-blue-typography-printed-cotton-t-shirt/...
```

So the app **manufactures the missing structured columns once, offline** (`enrich.py`):
- **product_type** from the URL path segment (Myntra's own taxonomy — no "other" gaps),
- **gender** from the slug (the `category` column alone is mostly "Unisex" and unreliable),
- **color / material / fit / pattern** from the slug + name via curated vocabularies and
  whole-word matching.

Filtering then becomes **precise column matching** instead of fuzzy guessing on a short name.

> **Honest ceiling:** recall is bounded by what the title text actually says. A product whose
> title omits its color simply has no color to filter on — the source data has no color field.
> This is a data limitation, not a code bug; the roadmap below is how we push past it.

---

## Known problems / limitations

- Attribute **recall** is limited to what the product title contains (source-data limit).
- Dictionary extraction can miss rare or misspelled colors/materials (mitigated by vocab + synonyms).
- Gender/type from URL slugs are mostly reliable but not 100% (malformed URLs fall back to the name).
- One cheap LLM call per product query (a latency/cost trade for accuracy); degrades to rules offline.
- No multi-turn memory yet — each query is treated as standalone.
- The dataset has **no review text** (only `num_reviews` counts), which limits opinion-based Q&A.

---

## Roadmap — making it better with agentic AI & GenAI

Built in phases on top of the enrichment foundation:

- **Phase 2 — Vector + visual search:** sentence-transformers + FAISS for true semantic/synonym
  matching; optional CLIP image embeddings over `image_url` for visual / "find similar" search.
- **Phase 3 — Multi-agent + tool-calling:** replace the ReAct *text* agent with Groq
  **function-calling**; a router dispatches to specialist agents (shopping / analytics / stylist /
  deals) plus an **LLM re-ranker** over candidates.
- **Phase 4 — RAG + eval + guardrails:** retrieval-grounded catalog Q&A, an **LLM-as-judge**
  search-quality eval harness, and prompt-injection / safety guardrails.

Further ideas: one-time **LLM bulk attribute extraction** (higher quality than dictionaries),
conversational **memory & refinement** (*"in black"*, *"cheaper"*), **recommendations**
("complete the look"), and cross-session **personalization**.

---

## Run locally

```bash
pip install -r requirements.txt
# .env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

streamlit run app.py
```

Run tests:

```bash
pytest tests/ -q
```

# Changelog: Myntra Market Intelligence Agent Optimizations

This document details the architectural, performance, and UI/UX improvements made to the Myntra Market Intelligence Agent to scale it for a 1M+ row dataset, bypass strict LLM rate limits, and provide a seamless, production-ready user experience.

## 🚀 1. Architecture & Token Optimization (The "Smart Router")
**Problem:** The Groq API (Llama 3 70B) has a strict 100K Tokens-Per-Day limit. Initially, every interaction (including "hi" or basic searches) was sent to the LLM, alongside 5 turns of full chat history (which included massive markdown tables and image URLs). This caused the app to hit the daily limit within ~30 queries.
**Solutions Implemented:**
* **Dual-Layer NLP Router (`app.py`):** Built `get_smart_response()`, a custom Python routing layer that intercepts user queries *before* they reach the LLM. 
  * **Greetings:** Automatically handled locally (0 tokens).
  * **Product Searches:** Parsed using Regex (to extract budgets like "under 2000" and strip filler words) and executed directly against the Pandas dataframe (0 tokens, instant response).
  * **Analytical Queries:** (e.g., "brand performance", "category summary") Intelligently identified and routed to the LangChain ReAct agent.
* **Context Truncation:** Modified `build_chat_context()` to only inject the last 3 *user queries* (truncated to 120 chars each) into the LLM prompt, entirely excluding the AI's massive markdown responses. **Result: Saved ~2,500+ tokens per query.**
* **Graceful Fallbacks (`agent.py`):** Added a `try/except` block to detect HTTP 429 Rate Limit errors. If the primary model fails, the system automatically falls back to `llama-3.1-8b-instant` or `gemma2-9b-it`. If all fail, it returns a friendly "high traffic" message instead of crashing with a raw JSON error.

## 🧠 2. Advanced Search Engine (`data_tools.py`)
**Problem:** The original search tool only checked the `product_name` column. Queries like "kids clothes" failed because products were named "Printed T-shirt Set" but categorized as "Kids" in a different column. 
**Solutions Implemented:**
* **Multi-Column Search:** Completely rewrote the core search algorithm (`find_products()`). It now splits the user's query into individual keywords and performs an intelligent `OR` search across `product_name`, `brand`, and `category` columns simultaneously.
* **Regex Price Extraction:** Built custom Regex in `app.py` to extract both `max_price` ("under 2000", "budget 500") and `min_price` ("above 2500", "over 1000"), passing them securely to the search engine.
* **Smart Word Boundaries:** Fixed a bug where `.replace("a", "")` would corrupt words (turning "jeans" into "je ns"). Now uses `re.sub(r'\bword\b')` for precise filler-word stripping.

## ⚡ 3. Data & Performance Optimizations
**Problem:** Re-parsing a 36MB CSV file on every single Streamlit re-run (button click, chat input) caused 2-4 seconds of UI lag per interaction.
**Solutions Implemented:**
* **`@st.cache_data` Implementation:** Wrapped the CSV loading logic in a cached function, ensuring the 1M+ row dataset is loaded into RAM exactly once per session.
* **Deterministic Sampling:** Added `random.seed(42)` to the fallback sample data generator to prevent the UI from thrashing when the cache resets.
* **Reduced Tool Payloads:** Sliced the Pandas dataframe returns in `data_tools.py` to prevent the LLM from choking on massive data context.
  * Brand Performance: Reduced from Top 50 to Top 20.
  * Search Results & High Discounts: Reduced from 20 items to 10 items.

## 🎨 4. UI / UX Enhancements
* **Rich Markdown Formatting:** Updated the local Python product search to output the exact same beautiful, rich Markdown format (with rendered images and clickable "View on Myntra" links) as the LLM agent, ensuring a seamless user experience regardless of which routing layer handled the query.
* **Theme Alignment:** Modified `.streamlit/config.toml` to enforce a dark theme (`#0f172a`), fixing a jarring visual bug where dark custom chat bubbles were rendered on a bright white background.
* **Loading Indicators:** Added `st.spinner()` during the initial heavy CSV data load to inform users the system is working.
* **Agent Timeout Handling:** Reduced `max_iterations` to 8 and increased `max_execution_time` to 60s in `agent.py`. Added specific logic to detect "Agent stopped due to time limit" and translate it into a friendly "I'm having trouble finding the answer, please rephrase" prompt.

## 📦 5. Environment Fixes
* **Missing Dependencies:** Added `langchain-classic` to `requirements.txt` to prevent deployment crashes on Streamlit Community Cloud.
* **Git Optimization:** Configured `.gitignore` to block raw `.csv` and `.zip` files while explicitly allowing `cleaned_myntra_data.csv` to ensure smooth GitHub pushes.

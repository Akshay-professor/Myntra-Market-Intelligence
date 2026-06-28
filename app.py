import os
import random
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import get_agent_response

logger = logging.getLogger(__name__)

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_FILE = BASE_DIR / "data" / "sample" / "myntra_sample_data.csv"
LOCAL_FULL_DATA_FILE = BASE_DIR / "cleaned_myntra_data.csv"

st.set_page_config(page_title="Myntra Market Intelligence", page_icon="🛍️", layout="wide")

st.markdown(
    """
    <style>
    footer {visibility: hidden;}

    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }

    .stChatMessage[data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #0f172a !important;
        border: 1px solid #1e293b;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #1f2937 !important;
        border: 1px solid #374151;
    }

    .stChatMessage[data-testid="stChatMessage"] p,
    .stChatMessage[data-testid="stChatMessage"] li,
    .stChatMessage[data-testid="stChatMessage"] span,
    .stChatMessage[data-testid="stChatMessage"] div,
    .stChatMessage[data-testid="stChatMessage"] td,
    .stChatMessage[data-testid="stChatMessage"] th,
    .stChatMessage[data-testid="stChatMessage"] code {
        color: #f8fafc !important;
    }

    .stChatMessage[data-testid="stChatMessage"] a {
        color: #93c5fd !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Data Loading (Cached) ----------

@st.cache_data
def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load and cache a CSV file so it's only parsed once."""
    return pd.read_csv(filepath)


@st.cache_data
def generate_sample_data() -> pd.DataFrame:
    random.seed(42)  # Fix #7: deterministic sample data
    brands = ["Nike", "Adidas", "H&M", "Zara", "Levis", "Puma", "UCB", "Roadster", "HRX", "Biba"]
    categories = ["Men", "Women", "Kids"]
    data = []

    for i in range(1, 51):
        brand = random.choice(brands)
        category = random.choice(categories)
        original_price = random.randint(200, 5000)
        discount_pct = random.randint(10, 70)
        discounted_price = round(original_price * (1 - discount_pct / 100), 2)
        rating = round(random.uniform(2.5, 5.0), 1)

        data.append(
            {
                "product_id": f"P{10000 + i}",
                "product_name": f"{brand} {category} Apparel {i}",
                "brand": brand,
                "category": category,
                "original_price": original_price,
                "discount_pct": discount_pct,
                "discounted_price": discounted_price,
                "rating": rating,
                "num_reviews": random.randint(0, 5000),
            }
        )

    return pd.DataFrame(data)


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    required_cols = {
        "product_id",
        "product_name",
        "brand",
        "category",
        "original_price",
        "discount_pct",
        "discounted_price",
        "rating",
        "num_reviews",
    }
    missing = required_cols - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    if df.empty:
        return False, "Dataset is empty."
    return True, ""


def prepare_rating_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    rating_bins = [0, 3.0, 3.5, 4.0, 4.5, 5.01]
    rating_labels = ["<3.0", "3.0-3.5", "3.5-4.0", "4.0-4.5", "4.5+"]
    bucketed = frame.copy()
    bucketed["rating_bucket"] = pd.cut(
        bucketed["rating"],
        bins=rating_bins,
        labels=rating_labels,
        include_lowest=True,
        right=False,
    )
    bucket_counts = (
        bucketed.groupby("rating_bucket", observed=False)
        .size()
        .reindex(rating_labels, fill_value=0)
        .reset_index(name="product_count")
    )
    return bucket_counts


# ---------- Greeting & Simple Query Handling (No LLM needed) ----------

_GREETING_WORDS = {"hi", "hello", "hey", "hii", "hiii", "hola", "sup", "yo", "greetings"}
_GREETING_PHRASES = {"how are you", "how r u", "what's up", "whats up", "good morning", "good evening", "good afternoon"}

def _is_greeting(text: str) -> bool:
    """Check if the user message is a simple greeting."""
    cleaned = text.strip().lower().rstrip("?!.,")
    if cleaned in _GREETING_WORDS:
        return True
    for phrase in _GREETING_PHRASES:
        if phrase in cleaned and len(cleaned) < 30:
            return True
    return False

_GREETING_RESPONSE = (
    "Hey there! 👋 Welcome to the Myntra Market Intelligence Agent!\n\n"
    "I can help you with:\n"
    "- 🔍 **Finding products** — *'show me jeans under ₹2000'*\n"
    "- 📊 **Brand analytics** — *'top brands by discount'*\n"
    "- ⭐ **Ratings & reviews** — *'best rated products'*\n"
    "- 📈 **Category insights** — *'category pricing summary'*\n\n"
    "What would you like to explore?"
)

_ANALYTICAL_KEYWORDS = [
    "top brands", "brand performance", "average", "summary", "compare",
    "distribution", "category", "rating", "discount strategy", "most reviewed",
    "highest", "lowest", "how many", "total", "count", "percentage",
    "analytics", "analysis", "insight", "trend", "overall",
]

def _is_analytical_query(text: str) -> bool:
    """Check if query needs LLM-powered analytics."""
    q = text.lower()
    return any(kw in q for kw in _ANALYTICAL_KEYWORDS)


def _try_direct_product_search(query: str, dataframe) -> str | None:
    """Attempt to answer product queries directly without the LLM.
    Returns a formatted response string, or None if the query needs the LLM.
    """
    import re
    import data_tools

    q = query.lower().strip()

    # Skip analytical queries — those need the LLM
    if _is_analytical_query(q):
        return None

    # Try to extract a max price
    price_match = re.search(
        r'(?:under|below|within|budget|max|upto|up to|less than)\s*(?:rs\.?|₹|inr)?\s*(\d+)', q
    )
    max_price = float(price_match.group(1)) if price_match else None

    # Filler phrases to strip (multi-word first, then single words)
    filler_phrases = [
        "show me", "find me", "i want to buy", "i want to purchase",
        "i want", "i need", "search for", "looking for", "get me",
        "up to", "less than",
    ]
    filler_words = [
        "show", "find", "buy", "purchase", "suggest", "recommend",
        "under", "below", "within", "budget", "max", "upto",
        "please", "pls", "plz", "some", "any",
        "rs", "₹", "inr", "rupees", "rupee",
    ]

    search_term = q
    # Strip multi-word fillers first (order matters)
    for filler in filler_phrases:
        search_term = search_term.replace(filler, " ")
    # Strip single filler words using word boundaries (so "a" doesn't break "jeans")
    for word in filler_words:
        search_term = re.sub(r'\b' + re.escape(word) + r'\b', ' ', search_term)
    # Remove price numbers
    search_term = re.sub(r'\d+', '', search_term).strip()
    # Clean up extra spaces
    search_term = re.sub(r'\s+', ' ', search_term).strip()

    if not search_term or len(search_term) < 2:
        return None

    # Build the search input
    search_input = f"{search_term}|{int(max_price)}" if max_price else search_term
    result = data_tools.search_products(dataframe, search_input)

    if "No products found" in result:
        return (
            f"Sorry, I couldn't find any **{search_term}** products"
            + (f" under ₹{int(max_price)}" if max_price else "")
            + " in the catalog. Try a different keyword!"
        )

    # Format nicely
    header = f"Here are the top results for **{search_term}**"
    if max_price:
        header += f" under **₹{int(max_price)}**"
    header += ":\n\n"

    return header + result


def build_chat_context(messages: list, current_query: str) -> str:
    """Build a compact context string from recent user messages only."""
    recent_user_msgs = [
        m["content"][:120] for m in messages if m["role"] == "user"
    ][-3:]

    if not recent_user_msgs:
        return current_query

    context_lines = [f"- {msg}" for msg in recent_user_msgs]
    return (
        f"Recent user queries for context:\n"
        + "\n".join(context_lines)
        + f"\n\nCurrent question: {current_query}"
    )


def get_smart_response(query: str, dataframe, messages: list) -> str:
    """Smart router: handles greetings and product searches locally,
    only calls the LLM for complex analytical queries. Saves tokens.
    """
    # 1. Handle greetings without calling the LLM
    if _is_greeting(query):
        return _GREETING_RESPONSE

    # 2. Try direct product search (no LLM needed for shopping queries)
    direct_result = _try_direct_product_search(query, dataframe)
    if direct_result is not None:
        return direct_result

    # 3. Analytical or complex query → use the LLM agent
    full_prompt = build_chat_context(messages, query)
    return get_agent_response(dataframe, full_prompt)


# ---------- Session State ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------

with st.sidebar:
    st.title("🛍️ Myntra Market Intelligence")
    st.write("Upload a dataset or use generated sample data.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            df = pd.DataFrame()
    else:
        # Fix #1 & #13: Cached loading with spinner
        if LOCAL_FULL_DATA_FILE.exists():
            with st.spinner("Loading full dataset..."):
                df = load_csv_data(str(LOCAL_FULL_DATA_FILE))
            st.info(f"Using local cleaned dataset ({len(df):,} products).")
        elif SAMPLE_DATA_FILE.exists():
            df = load_csv_data(str(SAMPLE_DATA_FILE))
            st.info("Using default sample dataset (50 rows).")
        else:
            df = generate_sample_data()
            st.info("Using generated sample dataset (50 rows).")

    if not df.empty:
        st.markdown("### Dataset Stats")
        c1, c2 = st.columns(2)
        c1.metric("Total Products", f"{len(df):,}")
        c2.metric("Brands", f"{df['brand'].nunique() if 'brand' in df.columns else 0:,}")

        c3, c4 = st.columns(2)
        c3.metric("Categories", f"{df['category'].nunique() if 'category' in df.columns else 0}")
        avg_discount = df["discount_pct"].mean() if "discount_pct" in df.columns else 0
        c4.metric("Avg Discount %", f"{avg_discount:.1f}%")

        with st.expander("Dataset Preview"):
            st.dataframe(df.head(5), use_container_width=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        chat_export = "Myntra AI Insights Report\n" + "=" * 30 + "\n\n"
        for msg in st.session_state.messages:
            role = "User Query" if msg["role"] == "user" else "AI Analyst"
            chat_export += f"{role}:\n{msg['content']}\n{'-' * 30}\n"

        st.download_button(
            label="📄 Download Insights Report",
            data=chat_export,
            file_name="myntra_ai_insights.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ---------- Main Content ----------

st.header("Myntra Market Intelligence Agent")
st.subheader("Powered by Groq AI + LangChain")

if not os.getenv("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY not found. Create a .env file in the project root with:\n\n"
        "GROQ_API_KEY=your_groq_api_key_here"
    )
    st.stop()

is_valid, validation_msg = validate_dataframe(df)
if not is_valid:
    st.warning(f"Dataset issue: {validation_msg}")
    st.stop()

st.markdown("## Visual Insights")
st.caption("These charts update automatically for either the uploaded CSV or the built-in sample dataset.")

visual_tabs = st.tabs([
    "Brand Performance",
    "Category Pricing",
    "Rating Insights",
    "Discount Strategy",
])

with visual_tabs[0]:
    st.markdown("### Brand Performance")
    st.caption("The first chart shows the top 10 brands by average discount percentage. The second chart shows which brands have the largest product catalogs.")

    brand_discount_df = (
        df.groupby("brand", as_index=False)
        .agg(avg_discount_pct=("discount_pct", "mean"), total_products=("product_id", "count"))
        .round(2)
    )

    left_brand, right_brand = st.columns(2)
    with left_brand:
        st.markdown("**Top 10 Brands by Average Discount**")
        st.bar_chart(
            brand_discount_df.sort_values("avg_discount_pct", ascending=False).head(10),
            x="brand",
            y="avg_discount_pct",
            use_container_width=True,
        )
    with right_brand:
        st.markdown("**Top 10 Brands by Product Count**")
        st.bar_chart(
            brand_discount_df.sort_values("total_products", ascending=False).head(10),
            x="brand",
            y="total_products",
            use_container_width=True,
        )

with visual_tabs[1]:
    st.markdown("### Category Pricing")
    st.caption("This chart compares average original price versus average discounted price by category.")

    category_pricing_df = (
        df.groupby("category", as_index=False)
        .agg(
            mean_original_price=("original_price", "mean"),
            mean_discounted_price=("discounted_price", "mean"),
        )
        .round(2)
    )
    st.bar_chart(
        category_pricing_df,
        x="category",
        y=["mean_original_price", "mean_discounted_price"],
        use_container_width=True,
    )

with visual_tabs[2]:
    st.markdown("### Rating Insights")
    st.caption("Ratings are bucketed into business-friendly tiers so the distribution is easier to read.")

    rating_distribution_df = prepare_rating_buckets(df)
    st.bar_chart(
        rating_distribution_df,
        x="rating_bucket",
        y="product_count",
        use_container_width=True,
    )

with visual_tabs[3]:
    st.markdown("### Discount Strategy")
    st.caption("Each dot represents a product. The X-axis shows discount percentage, the Y-axis shows rating, and color separates categories.")

    scatter_df = df[["discount_pct", "rating", "category"]].copy()
    st.scatter_chart(
        scatter_df,
        x="discount_pct",
        y="rating",
        color="category",
        use_container_width=True,
    )

# ---------- Quick Insight Buttons ----------

st.markdown("**Quick Insights**")
q1, q2, q3, q4 = st.columns(4)

quick_query = None
if q1.button("🏆 Top Brands", use_container_width=True):
    quick_query = "What are the top 10 brands with the highest average discount?"
if q2.button("💰 High Discounts", use_container_width=True):
    quick_query = "Show me products with a discount greater than 60%."
if q3.button("⭐ Best Rated", use_container_width=True):
    quick_query = "What is the overall brand performance sorted by average rating?"
if q4.button("📊 Category Summary", use_container_width=True):
    quick_query = "Give me a summary of average price and discount for each category."

# ---------- Chat Display ----------

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ---------- Chat Handlers ----------

if quick_query:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(quick_query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Agent is analyzing..."):
            try:
                response = get_smart_response(quick_query, df, st.session_state.messages)
            except Exception as e:
                logger.error("Quick query failed: %s", e)
                response = "Something went wrong while processing your request. Please try again!"
            st.markdown(response)
            st.session_state.messages.append({"role": "user", "content": quick_query})
            st.session_state.messages.append({"role": "assistant", "content": response})

if prompt := st.chat_input("Ask anything about Myntra data..."):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Agent is analyzing..."):
            try:
                response = get_smart_response(prompt, df, st.session_state.messages)
            except Exception as e:
                logger.error("Chat query failed: %s", e)
                response = "Something went wrong while processing your request. Please try again!"
            st.markdown(response)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})


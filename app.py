import os
import random
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import get_agent_response

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


@st.cache_data
def generate_sample_data() -> pd.DataFrame:
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


if "messages" not in st.session_state:
    st.session_state.messages = []

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
        if LOCAL_FULL_DATA_FILE.exists():
            df = pd.read_csv(LOCAL_FULL_DATA_FILE)
            st.info("Using local cleaned dataset (Full Data).")
        elif SAMPLE_DATA_FILE.exists():
            df = pd.read_csv(SAMPLE_DATA_FILE)
            st.info("Using default sample dataset (50 rows).")
        else:
            df = generate_sample_data()
            st.info("Using generated sample dataset (50 rows).")

    if not df.empty:
        st.markdown("### Dataset Stats")
        c1, c2 = st.columns(2)
        c1.metric("Total Products", f"{len(df)}")
        c2.metric("Brands", f"{df['brand'].nunique() if 'brand' in df.columns else 0}")

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

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if quick_query:
    st.session_state.messages.append({"role": "user", "content": quick_query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(quick_query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Agent is analyzing..."):
            try:
                # Compile chat history to give the agent context
                history_context = "\n".join([
                    f"{m['role']}: {m['content']}" 
                    for m in st.session_state.messages[-5:-1]
                ])
                full_prompt = f"Previous Chat History:\n{history_context}\n\nNew User Query: {quick_query}" if history_context else quick_query
                
                response = get_agent_response(df, full_prompt)
            except Exception as e:
                response = f"An unexpected error occurred: {e}"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if prompt := st.chat_input("Ask anything about Myntra data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Agent is analyzing..."):
            try:
                # Compile chat history to give the agent context
                history_context = "\n".join([
                    f"{m['role']}: {m['content']}" 
                    for m in st.session_state.messages[-5:-1]
                ])
                full_prompt = f"Previous Chat History:\n{history_context}\n\nNew User Query: {prompt}" if history_context else prompt
                
                response = get_agent_response(df, full_prompt)
            except Exception as e:
                response = f"An unexpected error occurred: {e}"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

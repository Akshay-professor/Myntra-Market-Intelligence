import os

from langchain_classic.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

import data_tools


DEFAULT_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]


DOMAIN_KEYWORDS = {
    "myntra",
    "product",
    "products",
    "brand",
    "brands",
    "category",
    "categories",
    "discount",
    "discounted",
    "price",
    "pricing",
    "rating",
    "ratings",
    "review",
    "reviews",
    "revenue",
    "catalog",
    "apparel",
    "women",
    "men",
    "kids",
}


def is_query_in_scope(df, user_query: str) -> bool:
    """Returns True if query appears related to Myntra dataset analytics."""
    q = (user_query or "").strip().lower()
    if not q:
        return False

    if any(word in q for word in DOMAIN_KEYWORDS):
        return True

    # Dynamic signal: match known brands/categories from the loaded dataset.
    try:
        brands = set(df["brand"].dropna().astype(str).str.lower().unique())
        categories = set(df["category"].dropna().astype(str).str.lower().unique())
        if any(b in q for b in brands) or any(c in q for c in categories):
            return True
    except Exception:
        # If schema is unexpected, don't block agent calls here.
        return True

    return False


def get_agent_response(df, user_query: str) -> str:
    """Runs a LangChain ReAct agent over Myntra dataframe tools and returns a user-friendly answer."""
    if not is_query_in_scope(df, user_query):
        return (
            "I can only answer Myntra dataset questions (brands, categories, discounts, "
            "prices, ratings, and reviews). Please ask a data question like: "
            "'Top 5 brands by average discount' or 'Show products above 60% discount'."
        )

    if not os.getenv("GROQ_API_KEY"):
        return "Error: GROQ_API_KEY is not set. Please add it to your .env file."

    preferred_model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    def safe_tool_call(tool_fn, fallback_msg):
        def _runner(x):
            try:
                return tool_fn(x)
            except Exception as exc:
                return f"{fallback_msg}. Details: {exc}"

        return _runner

    tools = [
        Tool(
            name="TopBrandsByDiscount",
            func=safe_tool_call(
                lambda _x: data_tools.get_top_brands_by_discount(df),
                "Tool failed while calculating top brands by discount",
            ),
            description="Returns top 10 brands with highest average discount percent. Ignores input.",
        ),
        Tool(
            name="CategorySummary",
            func=safe_tool_call(
                lambda _x: data_tools.get_category_summary(df),
                "Tool failed while generating category summary",
            ),
            description="Returns category-wise average price, average discount and product count. Ignores input.",
        ),
        Tool(
            name="MostReviewedProducts",
            func=safe_tool_call(
                lambda _x: data_tools.get_most_reviewed_products(df),
                "Tool failed while fetching most reviewed products",
            ),
            description="Returns top 10 most reviewed products with ratings. Ignores input.",
        ),
        Tool(
            name="HighDiscountProducts",
            func=safe_tool_call(
                lambda x: data_tools.get_high_discount_products(df, x),
                "Tool failed while finding high discount products",
            ),
            description="Returns products above a discount threshold. Input must be a single numeric threshold like 50.",
        ),
        Tool(
            name="RatingDistribution",
            func=safe_tool_call(
                lambda _x: data_tools.get_rating_distribution(df),
                "Tool failed while calculating rating distribution",
            ),
            description="Returns product counts in rating buckets: <3, 3-3.5, 3.5-4, 4-4.5, 4.5+. Ignores input.",
        ),
        Tool(
            name="BrandPerformance",
            func=safe_tool_call(
                lambda _x: data_tools.get_brand_performance(df),
                "Tool failed while generating brand performance",
            ),
            description="Returns brand-wise average rating, average discount and total products sorted by average rating. Ignores input.",
        ),
    ]

    template = """You are a Myntra Business Analyst AI.
Answer clearly using only the data returned by tools.
If tool outputs are unavailable, explain what went wrong in a friendly way.
If the question is not related to Myntra dataset analytics, directly respond with a short
out-of-scope message and do not attempt any tool call.

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    def run_with_model(model_name: str) -> str:
        llm = ChatGroq(model_name=model_name, temperature=0)
        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=30,
        )
        result = executor.invoke({"input": user_query})
        output = result.get("output", "I could not generate an answer based on the data.")

        # Convert generic executor stopping text into a useful UX message.
        if "agent stopped due to iteration limit or time limit" in str(output).lower():
            return (
                "I could not finish that query in time. Please ask a more specific Myntra data "
                "question (for example: 'Top brands by discount' or 'Category-wise average price')."
            )

        return output

    try:
        return run_with_model(preferred_model)
    except Exception as e:
        primary_err = str(e)
        if "decommissioned" in primary_err.lower() or "model_decommissioned" in primary_err.lower():
            for fallback_model in FALLBACK_MODELS:
                if fallback_model == preferred_model:
                    continue
                try:
                    return run_with_model(fallback_model)
                except Exception:
                    continue
            return (
                "The selected Groq model appears to be unavailable or decommissioned. "
                "Please set a working model in your .env, for example:\n"
                "GROQ_MODEL=llama-3.3-70b-versatile"
            )

        return f"An error occurred while analyzing the data: {e}"

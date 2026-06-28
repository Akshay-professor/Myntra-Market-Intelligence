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


def get_agent_response(df, user_query: str) -> str:
    """Runs a LangChain ReAct agent over Myntra dataframe tools and returns a user-friendly answer."""
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
        Tool(
            name="ProductSearch",
            func=safe_tool_call(
                lambda x: data_tools.search_products(df, x),
                "Tool failed while searching products",
            ),
            description="Searches for products by keyword and optional maximum price. Input format: 'keyword' OR 'keyword|max_price'. Example: 'polo green' or 'jeans|2000'.",
        ),
    ]

    template = """You are a Myntra Business Analyst AI.
Answer clearly using only the data returned by tools.
If tool outputs are unavailable, explain what went wrong in a friendly way.

IMPORTANT VISUAL FORMATTING:
If the tool data includes 'image_url' and 'product_url', you MUST format each product in your Final Answer using real markdown images and links. 
Example format:
**[Product Name]**
![Product Image](image_url)
Price: ₹[discounted_price] | Discount: [discount_pct]%
[🔗 View on Myntra](product_url)
---

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
        return result.get("output", "I could not generate an answer based on the data.")

    try:
        return run_with_model(preferred_model)
    except Exception as e:
        primary_err = str(e).lower()
        
        # Try fallback models for rate limits or decommissioned models
        if "rate_limit" in primary_err or "429" in primary_err or "decommissioned" in primary_err or "rate limit" in primary_err:
            for fallback_model in FALLBACK_MODELS:
                if fallback_model == preferred_model:
                    continue
                try:
                    return run_with_model(fallback_model)
                except Exception:
                    continue
            
            # If all fallbacks fail
            if "rate_limit" in primary_err or "429" in primary_err or "rate limit" in primary_err:
                return "Our AI is currently experiencing high traffic and has reached its daily limit. Please try again a bit later!"
            else:
                return "The AI models are currently unavailable. Please try again later."

        return "Oops! I encountered an unexpected error while trying to process your request. Please try asking in a different way."

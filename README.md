# 🛍️ Myntra Market Intelligence & AI Agent

> **Chat with your data. No dashboards. No filters. Just ask.**

An end-to-end retail analytics platform that combines traditional BI reporting with a conversational AI agent — allowing category managers to extract business insights from Myntra product data through natural language queries.

**Built by:** Akshay Kumar — Data Analyst | AI & Automation  
**Live Demo:** https://myntra-market-intelligence.streamlit.app/  
**GitHub:** [Akshay-professor](https://github.com/Akshay-professor) · **LinkedIn:** [akshay-poddar](https://www.linkedin.com/in/akshay-poddar-4299a5243/)

---

## 📸 Live Demo Screenshots

### Visual Dashboard & Data Insights
The app automatically generates interactive charts for brand performance, category pricing, rating analysis, and discount strategy.

![Visual Dashboard](./Myntra_Project_Screenshots/POWERBI%20DASHBOARD.png)

### Visual Insights
Detailed breakdown of key metrics and analytics across the dataset.

![Visual Insights](./Myntra_Project_Screenshots/Visual%20Insight.png)

### Conversational AI Chat Interface
Ask natural language questions and get instant, detailed insights powered by Groq's LLM and LangChain reasoning.

![Agentic Chat](./Myntra_Project_Screenshots/Agentic%20Chat.png)

---

## 🎯 What Problem Does This Solve?

Most e-commerce analysts spend hours building dashboards that still require manual interpretation. This project eliminates that gap.

| Traditional Approach | This Project |
|---|---|
| Static Power BI dashboards | Live conversational AI agent |
| Manual filter navigation | Natural language queries |
| Fixed metric views | Dynamic, on-demand analysis |
| Requires BI tool expertise | Anyone can just ask a question |

---

## 🧠 How It Works

The project is built in **two phases:**

### Phase 1 — Traditional BI Foundation
Raw Myntra data cleaned in Excel → analyzed via Pivot Tables → visualized in Power BI with DAX measures and KPI cards.

### Phase 2 — AI Agent (Primary Focus)
The same data is exposed through a **LangChain ReAct agent** backed by **Groq**. Pandas analytical functions are wrapped as agent tools — the AI calls the right tool, runs the analysis, and returns a human-readable answer.

```
User Query → LangChain ReAct Agent → Selects Tool → Pandas Function → Returns Insight
```

---

## 📊 Key Business Findings

| Metric | Value |
|---|---|
| 🏆 Top Brand by Revenue | Roadster — **8.95%** of total revenue |
| 📦 Top Category by Revenue | T-Shirts — **6.61%** of total revenue |
| 💰 Discount Sweet Spot | **50–60%** band shows strongest conversion |
| ⚠️ Concentration Risk | Top 10 brands = **19.30%** of total orders |
| ⭐ Rating Lift | Products rated **>4.0** sell **3.19×** more |

---

## 💬 Example Agent Queries

Paste these directly into the chat interface:

```
"Which brand has the highest average discount?"
"Show all products with more than 60% discount"
"Compare average ratings across Men, Women, and Kids categories"
"Which company needs to improve their product quality?"
"What is the revenue concentration risk across top brands?"
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Data Cleaning | Microsoft Excel |
| Data Processing | Pandas |
| Agent Framework | LangChain (ReAct) |
| LLM Inference | Groq API — `llama-3.3-70b-versatile` |
| UI | Streamlit |
| Environment | python-dotenv |

## 🎛️ Visual Dashboard

The app includes a native Streamlit visual layer above the chat interface so the user can inspect the dataset before asking questions.

1. Brand Performance: top brands by average discount and product count
2. Category Pricing: average original price vs discounted price by category
3. Rating Insights: bucketed rating distribution for clean readability
4. Discount Strategy: discount vs rating scatter plot colored by category

The dashboard automatically uses the uploaded CSV if one is provided, otherwise it falls back to the bundled sample dataset.

---

## 🗂️ Repository Structure

```
Myntra-Market-Intelligence-Agent/
│
├── app.py                        # Streamlit UI — chat interface, sidebar, quick insights
├── agent.py                      # LangChain ReAct agent + Groq integration
├── data_tools.py                 # 6 Pandas analytical tools exposed to the agent
├── requirements.txt
├── .gitignore
├── README.md
│
├── Myntra_PowerBI_Dashboard.pbix # Phase 1 Power BI report
├── myntra_datadict.pdf           # Data dictionary (PDF)
│
├── docs/
│   └── data_dictionary.md        # Data dictionary (Markdown)
│
├── .streamlit/
│   └── config.toml               # Theme configuration
│
└── Myntra_Project_Screenshots/
    ├── RAW DATASET.png
    ├── PreProcessed dataset.png
    ├── CLEANED DATA.png
    ├── PIVOT TABLE 1.png
    ├── PIVOT TABLE 2.png
    ├── POWERBI DATASET.png
    ├── POWERBI DASHBOARD.png
    └── SQL DAX FN.png
```

---

## ⚙️ Agent Tools (data_tools.py)

The agent has access to 6 purpose-built analytical tools:

| Tool | What It Does |
|---|---|
| `get_top_brands_by_discount` | Top 10 brands ranked by average discount % |
| `get_category_summary` | Avg price, avg discount, product count per category |
| `get_most_reviewed_products` | Top 10 products by review count with ratings |
| `get_high_discount_products` | All products above a discount threshold |
| `get_rating_distribution` | Product count across rating buckets |
| `get_brand_performance` | Brand-wise rating, discount, and product count |

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone <your-github-repo-url>
cd Myntra-Market-Intelligence-Agent
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Groq API key**  
Get a free key at [console.groq.com](https://console.groq.com), then:
```bash
# .env file
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set main file as `app.py`
4. Add `GROQ_API_KEY` in the **Secrets** section
4. Add `GROQ_API_KEY` and `GROQ_MODEL` in the **Secrets** section
5. Click **Deploy**


## 📐 KPI Definitions

| KPI | Formula |
|---|---|
| Revenue | `SUM(discounted_price × quantity)` |
| Brand Revenue Share | `brand_revenue / total_revenue` |
| Category Revenue Share | `category_revenue / total_revenue` |
| Rating Lift | `avg_sales(rating > 4.0) / avg_sales(rating ≤ 4.0)` |

---

## ✅ Data Quality Controls

- Duplicate records removed before aggregate reporting
- Missing ratings excluded to avoid biased quality metrics
- Category and brand text normalized before grouping
- All KPI values cross-validated between Excel and Power BI outputs

---

## ⚠️ Limitations

- Observational analysis only — does not establish causation
- Insights are bounded by the available time window and source quality
- Customer-level segmentation limited by available attributes

---

## 🗺️ Roadmap

- [ ] Add public sample dataset for full reproducibility
- [ ] Statistical hypothesis testing for discount and rating effects
- [ ] Forecasting module for category-level revenue trends
- [ ] Publish Power BI report as public URL

---

## 📄 License

This project is owned and maintained by **Akshay Kumar**.  
Feel free to fork and build on it with attribution.
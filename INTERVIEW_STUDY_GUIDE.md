# Interview Study Guide — Myntra Market Intelligence Agent

This guide is written the way you should actually *speak* in the interview — full sentences,
first person, plain English. Each answer below is roughly what you'd say out loud. Read them,
then say them in your own words until they feel natural. Don't memorise word-for-word; understand
the reasoning so you can defend it when the interviewer pushes back.

---

## The 90-second pitch (say this when they ask "tell me about your project")

"I built a conversational shopping and analytics app on top of a Myntra catalog of about a
hundred thousand products. You can talk to it in plain English — either to shop, like *'black
slim-fit jeans under 1500'*, or to ask business questions, like *'which brands give the highest
discounts'*. It's built with Streamlit for the UI, Groq's LLMs for the language understanding, and
pandas for the actual data work, and I styled the front end to look like the real Myntra site.

The interesting part isn't that it calls an LLM — anyone can do that. The interesting part is how
I made the search *accurate*. Early on it behaved like a dumb keyword matcher: if you searched
'black jeans' it would return black mascara, because 'black' appeared in the product name. So I
reworked it into a pipeline that first *understands* the query, pulls out structured facts like
item, colour, budget and gender, and then filters on real structured columns instead of doing
fuzzy text matching. And to get those structured columns, I realised the product URL already
contains Myntra's full product title — the category, the gender, the colour, the fabric — so I mine
that to build the columns the raw data was missing. That one insight fixed most of the accuracy
problems."

That's your opener. Everything below is you defending the details.

---

## Q1. Walk me through exactly what happens when a user types a query.

When the user types something in the chat box, the first thing I do is *not* call the LLM. I run a
cheap rule-based check first, because most messages are obvious and I don't want to pay for a model
call on every keystroke. If the message is a greeting like 'hi', I just return a canned welcome. If
it clearly contains analytics words like 'top brands' or 'average discount', I send it to the
analytics agent. Those two paths cost zero tokens.

If the rules can't confidently decide — which is the case for most real shopping queries — then I
make one call to a small, cheap model (Llama 3.1 8B on Groq) that does two jobs at once: it tells me
the intent, and it extracts the query into a structured JSON object. So 'i have a budget of 1500,
show me black jeans' comes back as something like `{intent: product_search, item: jeans, colour:
black, max_price: 1500}`.

From there it branches. If it's a product search, I hand those facts to my structured search
function, which filters the catalog and returns a grid of products. If it's an analytics question,
I hand it to a LangChain agent that has a set of pandas functions as tools, and it computes the
answer. If it's off-topic — someone asking 'who is the PM of India' — I politely decline and tell
them what I can actually help with. The result gets stored in Streamlit's session state and
rendered back into the chat, either as a product grid or as text.

The thing I'd emphasise is that this is a *router*, not one big prompt. Cheap deterministic rules
handle the easy cases, and I only spend an LLM call when there's genuine ambiguity.

---

## Q2. Your dataset only has eleven columns. How can you possibly filter by colour, fit, or gender?

This is the heart of the project, so let me be honest about the problem first. The raw data has
product name, brand, a coarse category, prices, discount, rating, review count, an image URL and a
product URL. Out of those, only price, discount, rating and brand are genuinely structured. The
things people actually shop by — the item type, the colour, the fabric, the fit, the gender — all
live inside the free-text product name. And the product name is often short and missing half of
that. So you literally cannot filter on data that isn't there as a column.

My first instinct was to parse the product name, but it's too sparse. Then I looked closely at the
product URL and realised it's a goldmine. Every Myntra URL follows the pattern
`myntra.com/{type}/{brand}/{a-long-descriptive-slug}/{id}/buy`. So the first part of the path is the
*real* product type — Myntra's own category, like 'tshirts' or 'heels' or 'kurtas' — and the long
slug is the full descriptive title with the gender, colour and fabric in it. For example a product
whose name is just 'Printed Round Neck T-shirt' has a URL slug that says
'fido-dido-men-black-printed-round-neck-t-shirt'. The name didn't tell me the gender or colour, but
the URL did.

So what I do is enrich the data once, offline: I parse the URL to get the product type and the rich
slug, combine that with the product name, and extract structured columns — product_type, gender,
colour, material, fit and pattern — using curated vocabularies and whole-word matching. After that,
filtering becomes precise column matching instead of guessing on text. The honest part I always add:
this is still bounded by what the title actually says. I measured it — colour is present for about
58% of products. The other 42% genuinely don't state a colour anywhere, so I can't invent one.
That's a data limitation, not a bug, and I think being upfront about that is the right engineering
posture.

---

## Q3. How do you decide what the user actually wants — the intent?

I use a hybrid, and the reason is cost. A pure LLM classifier would be accurate but I'd pay for a
call on every single message, including 'hi' and 'thanks'. A pure rule-based classifier is free but
brittle — it can't reliably tell 'running shoes' (a product) from 'running nose' (nonsense).

So I layer them. First, deterministic rules catch the obvious cases for free: greetings from a small
word list, analytics from a keyword list, and product searches when the query contains a known
product-type word or a price pattern. Only when the rules genuinely can't decide do I fall back to
the cheap model. That way the common cases are instant and free, and I spend tokens only on the hard,
ambiguous ones — which, importantly, is exactly where off-topic questions live, so the model is what
catches 'who is the PM of India' and lets me decline it.

One bug worth mentioning here: my analytics keyword matching originally used plain substring search,
and the word 'discount' contains 'count', which was in my analytics list — so 'above 70% discount'
got misrouted to analytics. I fixed it by switching to whole-word matching with regex word
boundaries. It's a small thing but it's a good example of the kind of subtle bug that separates a
demo from something that actually works.

---

## Q4. How do you turn a messy natural-language query into something you can search with?

I let the model do the understanding, but I keep it on a very tight leash. I send the query to the
small model with a strict instruction: return only a JSON object with a fixed set of fields — intent,
item, colour, a list of other attributes, brand, gender, min price, max price, min discount. I give
it a handful of examples in the prompt so it knows, for instance, that 'budget of 1500' means a
maximum price and 'above 50% off' means a minimum discount.

Then I never trust the output blindly. I have a normaliser that validates and cleans whatever comes
back — it lowercases things, coerces numbers, throws away invalid intents, and defaults safely. If
the model returns garbage or the API is down, the whole thing returns nothing and I fall back to the
old rule-based keyword path, so the app never crashes because of a bad model response.

I also added a specific guard for a failure I saw: if someone types just a brand name like
'roadster', the model sometimes hallucinated an item like 'watch'. So I check the query against the
actual set of brand names in the data, and if the whole query is a known brand, I force it into a
brand search and drop whatever item the model guessed. That's a nice example of using the data itself
to correct the model.

---

## Q5. Once you have the facts, how does the search actually work?

I split the facts into two groups: hard filters and soft attributes. The hard filters define the set
of candidates — gender maps to the category column, the item maps to the product_type column, brand
is a match on the brand column, and price and discount are simple numeric comparisons. These are
exact, so they're reliable.

The soft attributes — colour, fabric, fit, pattern — I apply as an AND on the enriched columns, but
with what I call graceful relaxation. So 'black slim cotton jeans' filters to products that are
black AND slim AND cotton AND jeans. But if that combination returns nothing — say there are no
brown caps in stock — instead of showing an empty page, I drop the least important attribute and try
again, until I get results. So the user always sees the closest thing, ranked by how many of their
criteria actually matched. I picked that behaviour deliberately; for shopping, showing 'close
matches' is better UX than a dead end.

There's also a relevance and ranking layer. I down-rank products that only mention the item as an
accessory — for example a 'bath robe with belt' should not rank above an actual belt when you search
'belts' — and then I sort by how many attributes matched and by popularity. And if the structured
filters still come back empty, there's a semantic fallback that I'll explain next.

---

## Q6. Tell me about the hardest bug you fixed.

The one I always tell is the 'black jeans returns mascara' bug, because it taught me the most. The
original search did keyword matching with OR logic across the product name, brand and category. So
'black jeans' matched anything containing 'black' *or* 'jeans' — which meant black mascara and black
kajal showed up under a jeans search. The fix was two-fold: require all keywords to match, not any,
and search on the structured product_type column instead of raw text.

But fixing that exposed a subtler version of the same problem. I was matching with a leading word
boundary, so 'cap' would still match anything *starting* with cap — cappuccino, captain, capris. So
a search for 'caps' returned cappuccino body wash and a 'Captain America' phone case, because my
enrichment had tagged those as product_type 'cap'. The real fix was whole-word matching on both
sides — a leading and trailing boundary plus optional plural — so 'cap' matches 'cap' and 'caps' but
not 'cappuccino'. I wrote that as one shared helper and used it everywhere, which also fixed 'men'
matching 'women' and 'tshirt' matching 'sweatshirt'.

What I'd want the interviewer to take from that story is the method, not just the fix: I found these
by actually testing the thing with real queries, I traced each one to a root cause in the matching
logic, and I fixed the root cause once in a shared place rather than patching symptoms. And I wrote
tests for each so they can't come back.

---

## Q7. What happens when there's no exact match, or the user makes a typo?

Two things. First, the relaxation I mentioned — if the attribute filters are too strict, I loosen
them step by step rather than returning empty. Second, for typos and unusual phrasing, I have a
semantic fallback built on TF-IDF over character n-grams. So if the structured search finds nothing
for 'runing shooes', the fallback still ranks the running shoes highly because the character
sequences overlap, even though the words are misspelled.

I chose TF-IDF on character n-grams deliberately over heavier neural embeddings, and I can defend
that choice: it's fast, it has no model download, it's deterministic so I can unit-test it, and it
runs comfortably on a free hosting tier. It won't catch true synonyms — 'denim' versus 'jeans' — so
for those I keep a small synonym map. I know the next step up is real sentence embeddings with a
vector index, and I've scoped that as a follow-up phase, but I made a conscious cost-versus-accuracy
trade-off for what's deployed today.

---

## Q8. This calls LLMs. How do you keep cost and latency under control?

I treat the LLM as an expensive resource and route around it whenever I can. Greetings and obvious
analytics queries are handled by rules and cost nothing. When I do need the model for understanding
a product query, I use the small, cheap 8B model, not the big one — it's a couple of hundred tokens
per query, which is fractions of a cent. I only bring out the large 70B model for the analytics
agent, where the multi-step reasoning actually needs it.

So the expensive model runs on a minority of queries, and the rest are either free or nearly free.
On top of that, the whole pipeline degrades gracefully: if the API is rate-limited or down, I have
model fallbacks, and ultimately the app drops to rule-based keyword search so it still returns
something instead of erroring. The principle is: spend tokens where they add accuracy, and nowhere
else.

---

## Q9. How did you test a system that depends on a non-deterministic LLM?

I drew a hard line between the deterministic parts and the model. Almost all of my logic — the
enrichment, the whole-word matching, the structured filtering, the relaxation, the ranking, the
brand correction, the intent rules — is pure functions that don't touch the network. I have about 66
pytest tests covering those, with fixed inputs and exact expected outputs. For example, I assert that
'cappuccino body wash' does not get tagged as a cap, that 'black slim cotton jeans' filters on all
three attributes, and that a bath robe ranks below real belts.

For the parts that do involve the LLM, I either test the pure helpers around it — like the JSON
normaliser and the brand override, which I can test with hand-built inputs — or I stub the model out
so the test is deterministic. I deliberately don't rely on asserting exact LLM output, because that's
flaky. I also run Streamlit's app-test harness to confirm the whole UI renders without exceptions.
The honest summary is: I made the system mostly deterministic *so that* it could be tested, and I
think that design choice is as important as the tests themselves.

---

## Q10. What are the limitations? What would you not claim this does?

I'm careful here because over-claiming is how you lose credibility. The biggest limitation is recall:
I can only filter on attributes that the product title actually mentions. Colour is present on about
58% of products, so a colour filter can't find the rest — not because my code is weak, but because
the source data doesn't contain it. The second limitation is that there's no review *text* in the
dataset, only review counts, so I can't honestly answer opinion questions like 'is this good for
summer' from real customer feedback. Third, my attribute extraction is dictionary-based, so it can
miss a rare colour or a misspelled fabric. And finally, each query is standalone — there's no
multi-turn memory yet, so 'now show me cheaper ones' wouldn't refine the previous search.

I list these because each one has a clear next step, and being able to name your own system's
weaknesses is what makes the strengths believable.

---

## Q11. If you had more time, how would you make it better?

I have a concrete roadmap, and I sequenced it by dependency. The foundation — the enrichment and the
accurate filtering — had to come first because everything else sits on top of clean structured data.

After that, the first upgrade is real semantic search: sentence-transformer embeddings with a vector
index like FAISS, which would handle true synonyms and intent, and optionally CLIP image embeddings
on the product images so you could do visual 'find similar' search. The second upgrade is on the
agent side: moving from the text-based ReAct agent to proper function calling, which is more
reliable, and splitting it into specialist agents — a shopping agent, an analytics agent, a stylist
agent that does 'complete the look', a deals agent — behind a router, plus an LLM re-ranker on the
final results. The third is retrieval-augmented Q&A grounded in the catalog, a proper evaluation
harness that uses an LLM as a judge to score search quality so I can catch regressions, and basic
guardrails against prompt injection.

The point I'd make is that I'm not just listing buzzwords — each phase is testable on its own,
depends on the previous one, and I've already been honest about the constraints, like the memory
cost of holding embeddings for a hundred thousand products on a free tier.

---

## Q12. Why these tools — Streamlit, Groq, pandas? Wouldn't a real system look different?

Yes, and I'd build it differently at scale — I'll come to that. For a project that has to be built,
demoed and reasoned about, these were pragmatic choices. Streamlit lets me ship a real interactive
UI in Python without a separate front-end stack, so I could spend my time on the actual logic. Groq
gives very fast LLM inference with a generous free tier and access to good open models, which keeps
the demo responsive and cheap. Pandas is the right tool for a hundred-thousand-row dataset that fits
comfortably in memory.

None of these are what I'd pick for production at massive scale, and I think saying that is a
strength, not a weakness — it shows I chose tools that fit the problem in front of me rather than
over-engineering.

---

## Q13. How would this scale from a hundred thousand rows to fifty million?

The first thing that breaks is the in-memory pandas model — I can't hold fifty million rows in RAM
and scan them on every query. So the data layer moves to a proper store: a columnar warehouse or a
search engine like Elasticsearch for the structured filtering, and a vector database for the semantic
side. My enrichment, which I currently do once at load, becomes a batch job in a pipeline that writes
the structured columns back to the store.

The good news is the *shape* of my architecture survives. The query-understanding layer that turns
language into structured facts stays exactly the same — it doesn't care whether the facts are then
run against pandas or against a SQL query or an Elasticsearch filter. So I'd swap the execution
engine underneath the structured-search function and keep the rest. I'd also add caching for common
queries and precompute popular aggregates for the analytics side. That's the answer that shows I
understand the prototype is a prototype, but I designed the boundaries so it can grow.

---

## How to deliver this in the room

Lead with the problem, not the tech. The strongest version of this whole story is: "search was
returning mascara for 'black jeans', I figured out why, and I rebuilt it to understand the query and
filter on structured data I mined from the product URL." That single narrative shows you can find
problems, reason about root causes, work with messy real-world data, and make honest trade-offs —
which is everything an intern interview is actually testing. Let the architecture details come out as
the interviewer digs in, and never be afraid to say what your system *can't* do. The honesty is what
makes the rest believable.

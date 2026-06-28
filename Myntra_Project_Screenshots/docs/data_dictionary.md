# Myntra Data Dictionary

## Ownership
This project and documentation are maintained by Akshay Kumar.

## Table Structure

| Column | Type | Description | Example |
|---|---|---|---|
| product_id | text/integer | Unique identifier for each product record | 100245 |
| product_name | text | Product display name | Roadster Men Solid T-Shirt |
| product_category | text | Product category label | tshirts |
| brand_name | text | Brand associated with the product | Roadster |
| marked_price | numeric | Price before discount | 1299 |
| discounted_price | numeric | Selling price after discount | 649 |
| discount_percentage | numeric | Percent discount applied | 50 |
| rating | numeric | Average user rating on a 1-5 scale | 4.2 |
| rating_count | integer | Number of rating submissions | 183 |
| revenue | numeric | Revenue generated for the product/row | 118767 |

## Derived Metrics

| Metric | Formula | Business Use |
|---|---|---|
| Revenue | discounted_price * quantity (or row-level sales value) | Track financial contribution by product, category, and brand |
| Brand Revenue Share | brand_revenue / total_revenue | Evaluate concentration and key brand dependence |
| Category Revenue Share | category_revenue / total_revenue | Prioritize merchandising and inventory focus |
| Rating Lift | sales_high_rating / sales_low_rating | Estimate relationship between product quality perception and sales |

## Data Quality Notes

1. Remove duplicate product rows before computing aggregations.
2. Standardize category and brand naming (case, spacing, spelling variants).
3. Treat missing ratings carefully to avoid biased rating-related insights.
4. Confirm monetary fields are numeric and in a single currency.
5. Re-validate all KPI formulas after transformation steps.

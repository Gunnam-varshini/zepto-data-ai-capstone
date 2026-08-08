# /data_pipeline

Scrapes book listings from books.toscrape.com, cleans and types the fields, converts
price to INR using a fixed baseline rate, and loads everything into a normalized
SQLite database queried with both SQL and pandas.

## Setup
pip install -r requirements.txt

## Run
Open and run `data_pipeline.ipynb` top to bottom in Jupyter/Colab.

## Design decisions
- Categories are discovered dynamically from the site's sidebar nav, not hardcoded.
- Currency conversion: price_inr = price_gbp * 105.50 (fixed project baseline rate,
  not a live market rate).
- Rows that fail to parse are dropped rather than median-imputed: price/rating/
  availability are scraped text labels, not noisy numeric measurements, so a parse
  failure means a malformed row, not a natural outlier a median would represent.
- Schema: two tables (categories, books) linked by category_id as a foreign key.

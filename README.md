# Amazon India Customer Review Analysis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458.svg)](https://pandas.pydata.org/)
[![NLTK VADER](https://img.shields.io/badge/NLTK-VADER%20Sentiment-3776AB.svg)](https://www.nltk.org/)
[![Laya Engine](https://img.shields.io/badge/NLP-Zero--Shot%20Intent%20Routing-FF6F00.svg)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Business Problem
On Amazon, products are sold by third-party sellers while delivery and service are handled by Amazon. A low rating alone doesn't show who should fix the problem. This project analyzes customer reviews to understand how customers feel about products and to separate product problems (seller side) from service problems (delivery and fulfillment), so improvement efforts can be focused in the right place.

## Dataset
Amazon Sales Dataset (Kaggle). After cleaning: 1,307 products and 11,873 individual reviews.

## Approach
- Cleaned the data and removed duplicate and variant listings
- Split bundled reviews into individual reviews
- Scored sentiment with VADER
- Classified each review as Product or Service using Laya (zero-shot)
- Built a product health scorecard using Wilson 95% confidence intervals

## Key Findings
- Customers are largely satisfied: 81.4% of reviews are positive and only 10.3% are negative
- Product quality is the main problem, not delivery: 208 products are flagged for product quality defects versus only 21 for logistics issues, about a 10 to 1 ratio
- Service reviews are less positive than product reviews: 73.1% positive versus 82.8%, though service makes up just 15.1% of all reviews
- 252 products (19%) are at-risk and need attention, while 533 are customer favorites
- Recommendation: focus first on seller and product quality control, with fulfillment as a secondary priority

## Tech Stack
Python, pandas, NumPy, NLTK (VADER), Laya, matplotlib, seaborn

## Output Files
- `product_health_scorecard.csv`: Product-level scorecard with confidence intervals and bottleneck classifications
- `granular_reviews_analyzed.csv`: 11,873 unbundled individual reviews with sentiment scores and intent tags
- `label_sample.csv`: Sample of 150 reviews exported for manual verification and auditing

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk tqdm laya
   ```
2. Open and run the notebook:
   ```bash
   jupyter notebook Customer_review_analysis.ipynb
   ```

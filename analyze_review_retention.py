import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')

orders  = pd.read_csv('orders.csv',  parse_dates=['order_date'])
reviews = pd.read_csv('reviews.csv', parse_dates=['review_date'])
prods   = pd.read_csv('products.csv')

reviews = reviews.merge(prods[['product_id','category','segment']], on='product_id', how='left')
reviews['order_year']  = reviews['review_date'].dt.year
reviews['review_month'] = reviews['review_date'].dt.month

first_ord = orders.groupby('customer_id')['order_date'].min().reset_index()
first_ord.columns = ['customer_id','first_order_date']
first_ord['cohort_year'] = first_ord['first_order_date'].dt.year

reviews_cust = reviews.merge(orders[['order_id','order_date']], on='order_id', how='left')
reviews_cust = reviews_cust.merge(first_ord[['customer_id','cohort_year']], on='customer_id', how='left')

print('=== 1. AVG RATING BY ORDER YEAR ===')
yr_rating = reviews_cust.groupby('order_year').agg(
    n_reviews   = ('review_id','count'),
    avg_rating  = ('rating','mean'),
    pct_5star   = ('rating', lambda x: (x==5).mean()*100),
    pct_1star   = ('rating', lambda x: (x==1).mean()*100),
    pct_low     = ('rating', lambda x: (x<=2).mean()*100),
).reset_index()
pd.set_option('display.float_format','{:.3f}'.format,'display.width',160)
print(yr_rating.to_string(index=False))
print()

print('=== 2. AVG RATING BY COHORT YEAR ===')
cohort_rating = reviews_cust.groupby('cohort_year').agg(
    n_reviews   = ('review_id','count'),
    avg_rating  = ('rating','mean'),
    pct_5star   = ('rating', lambda x: (x==5).mean()*100),
    pct_1star   = ('rating', lambda x: (x==1).mean()*100),
).reset_index().dropna(subset=['cohort_year'])
cohort_rating['cohort_year'] = cohort_rating['cohort_year'].astype(int)
print(cohort_rating.to_string(index=False))
print()

yr1_ret = {
    2012:64.7, 2013:50.8, 2014:35.1, 2015:27.4,
    2016:21.2, 2017:16.1, 2018:9.4,  2019:7.6,
    2020:6.8,  2021:6.7
}
cohort_rating['yr1_ret'] = cohort_rating['cohort_year'].map(yr1_ret)
merged = cohort_rating.dropna(subset=['yr1_ret'])
print('Correlation (cohort metrics vs Year+1 retention):')
for col in ['avg_rating','pct_5star','pct_1star']:
    r = merged['yr1_ret'].corr(merged[col])
    sig = '*** STRONG' if abs(r)>0.6 else ('** moderate' if abs(r)>0.35 else 'weak')
    print(f'  {col:12s}: r={r:+.3f}  {sig}')
print()

print('=== 3. RATING DISTRIBUTION BY ORDER YEAR (%) ===')
rating_dist = (reviews_cust.groupby(['order_year','rating'])['review_id']
               .count().unstack(fill_value=0))
rating_pct = rating_dist.div(rating_dist.sum(axis=1), axis=0) * 100
print(rating_pct.round(1).to_string())
print()

print('=== 4. AVG RATING BY CATEGORY x YEAR ===')
cat_yr = reviews_cust.groupby(['order_year','category'])['rating'].mean().unstack()
print(cat_yr.round(3).to_string())
print()

print('=== 5. RATING -> REPEAT PURCHASE LINK ===')
rev_with_cust = reviews_cust[['order_id','customer_id','rating','order_date']].dropna(subset=['customer_id','order_date'])
all_orders = orders[['order_id','customer_id','order_date']].sort_values('order_date')

rev_sample = rev_with_cust.copy()
rev_sample['review_order_date'] = rev_sample['order_date']

subsequent = all_orders.rename(columns={'order_id':'next_order','order_date':'next_date'})
rev_joined = rev_sample.merge(subsequent[['customer_id','next_date']], on='customer_id', how='left')
rev_joined = rev_joined[rev_joined['next_date'] > rev_joined['review_order_date']]
has_repeat = rev_joined.groupby('order_id')['next_date'].count().rename('n_subsequent') > 0
rev_sample = rev_sample.merge(has_repeat.rename('has_repeat'), on='order_id', how='left')
rev_sample['has_repeat'] = rev_sample['has_repeat'].fillna(False)

repeat_by_rating = rev_sample.groupby('rating').agg(
    n_reviews    = ('order_id','count'),
    repeat_rate  = ('has_repeat','mean'),
).reset_index()
repeat_by_rating['repeat_rate'] *= 100
print('Repeat purchase rate AFTER giving a specific rating:')
print(repeat_by_rating.to_string(index=False))
print()
print(f'Diff (5-star vs 1-star): '
      f'{repeat_by_rating[repeat_by_rating["rating"]==5]["repeat_rate"].values[0]:.1f}% '
      f'vs {repeat_by_rating[repeat_by_rating["rating"]==1]["repeat_rate"].values[0]:.1f}%')
print()

print('=== 6. RATING DISTRIBUTION: PRE vs POST DISCOUNT SYSTEM (2013) ===')
pre  = reviews_cust[reviews_cust['order_year'] <= 2012]['rating']
post = reviews_cust[reviews_cust['order_year'] >= 2013]['rating']
print(f'Pre-2013  (n={len(pre):,}): avg={pre.mean():.3f}  5-star={( pre==5).mean()*100:.1f}%  1-star={(pre==1).mean()*100:.1f}%')
print(f'Post-2013 (n={len(post):,}): avg={post.mean():.3f}  5-star={(post==5).mean()*100:.1f}%  1-star={(post==1).mean()*100:.1f}%')
print()

print('=== 7. NPS-LIKE SCORE BY YEAR ===')
def nps(s):
    promoters  = (s==5).sum()
    detractors = (s<=3).sum()
    return (promoters - detractors) / len(s) * 100

nps_yr = reviews_cust.groupby('order_year')['rating'].apply(nps).reset_index()
nps_yr.columns = ['order_year','nps_like']
nps_yr = nps_yr.merge(yr_rating[['order_year','avg_rating','n_reviews']], on='order_year')
print(nps_yr.to_string(index=False))

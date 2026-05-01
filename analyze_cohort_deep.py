import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')

orders = pd.read_csv('orders.csv',      parse_dates=['order_date'])
items  = pd.read_csv('order_items.csv', low_memory=False)
prods  = pd.read_csv('products.csv')
rets   = pd.read_csv('returns.csv',     parse_dates=['return_date'])
pays   = pd.read_csv('payments.csv')

items = items.merge(prods[['product_id','category','cogs','price']], on='product_id', how='left')
items['rev_line']  = items['quantity'] * items['unit_price']
items['cogs_line'] = items['quantity'] * items['cogs']

prod_gm = items.groupby('product_id').agg(
    gross_rev=('rev_line','sum'), cogs_total=('cogs_line','sum'),
    disc_total=('discount_amount','sum'),
).reset_index()
rets_pay = rets.merge(pays[['order_id','payment_value']], on='order_id', how='left')
rets_pay['refund_capped'] = rets_pay[['refund_amount','payment_value']].min(axis=1)
prod_refund = rets_pay.groupby('product_id')['refund_capped'].sum().reset_index()
prod_refund.columns = ['product_id','refund_capped']
prod_gm = prod_gm.merge(prod_refund, on='product_id', how='left')
prod_gm['refund_capped'] = prod_gm['refund_capped'].fillna(0)
prod_gm['net_rev'] = prod_gm['gross_rev'] - prod_gm['disc_total'] - prod_gm['refund_capped']
prod_gm['gm_pct']  = (prod_gm['gross_rev'] - prod_gm['cogs_total']) / prod_gm['net_rev'].replace(0,np.nan) * 100
PORT_GM = (prod_gm['gross_rev'] - prod_gm['cogs_total']).sum() / prod_gm['net_rev'].sum() * 100
prod_gm['is_neg'] = (prod_gm['gm_pct'] < 0).astype(int)
pid_gm  = dict(zip(prod_gm['product_id'], prod_gm['gm_pct']))
pid_neg = dict(zip(prod_gm['product_id'], prod_gm['is_neg']))

items['prod_gm']  = items['product_id'].map(pid_gm)
items['is_neg']   = items['product_id'].map(pid_neg).fillna(0)

order_agg = items.groupby('order_id').agg(
    order_rev   = ('rev_line','sum'),
    order_cogs  = ('cogs_line','sum'),
    order_disc  = ('discount_amount','sum'),
    n_items     = ('quantity','sum'),
    n_skus      = ('product_id','nunique'),
    has_neg     = ('is_neg','max'),          # 1 if any item is neg-margin
    avg_prod_gm = ('prod_gm','mean'),
    cat_mode    = ('category','first'),      # dominant category (approx)
).reset_index()
order_agg['order_gm_pct'] = (order_agg['order_rev'] - order_agg['order_cogs']) / order_agg['order_rev'].replace(0,np.nan) * 100
order_agg['disc_pct']     = order_agg['order_disc'] / order_agg['order_rev'].replace(0,np.nan) * 100

ret_set  = set(rets['order_id'])
ord_full = orders.merge(order_agg, on='order_id', how='left')
ord_full['returned']   = ord_full['order_id'].isin(ret_set).astype(int)
ord_full['order_year'] = ord_full['order_date'].dt.year

first_ord = ord_full.groupby('customer_id')['order_date'].min().reset_index()
first_ord.columns = ['customer_id','first_order_date']
first_ord['cohort_year'] = first_ord['first_order_date'].dt.year
ord_full = ord_full.merge(first_ord[['customer_id','cohort_year']], on='customer_id', how='left')
ord_full['period_offset'] = ord_full['order_year'] - ord_full['cohort_year']

cohort_size = ord_full.groupby('cohort_year')['customer_id'].nunique().rename('cohort_size')
ret_counts  = (ord_full.groupby(['cohort_year','period_offset'])['customer_id']
               .nunique().reset_index().rename(columns={'customer_id':'n_active'}))
ret_counts  = ret_counts.merge(cohort_size, on='cohort_year')
ret_counts['retention_pct'] = ret_counts['n_active'] / ret_counts['cohort_size'] * 100

ret_matrix  = ret_counts.pivot(index='cohort_year', columns='period_offset', values='retention_pct')
cols_show   = [c for c in range(10) if c in ret_matrix.columns]
ret_matrix  = ret_matrix[cols_show]

print('=== COHORT RETENTION MATRIX (First-Order, annual) ===')
print(ret_matrix.round(1).to_string())
print()

yr1 = ret_matrix[1].dropna()
print('Year+1 retention by cohort year:')
for yr, val in yr1.items():
    sz = int(cohort_size.get(yr, 0))
    print(f'  {int(yr)}: {val:5.1f}%  (n={sz:,})')
print(f'Trend: {yr1.iloc[0]:.1f}% -> {yr1.iloc[-1]:.1f}%  (delta={yr1.iloc[-1]-yr1.iloc[0]:+.1f} pp)')
print()

cohort_profile = (ord_full[ord_full['period_offset']==0]
                  .groupby('cohort_year').agg(
    n_orders    = ('order_id','nunique'),
    return_rate = ('returned','mean'),
    avg_gm_pct  = ('order_gm_pct','mean'),
    pct_neg     = ('has_neg','mean'),
    avg_disc    = ('disc_pct','mean'),
    avg_rev     = ('order_rev','mean'),
    avg_n_items = ('n_items','mean'),
).reset_index())
cohort_profile['return_rate'] *= 100
cohort_profile['pct_neg']     *= 100

print('=== COHORT PROFILE (do first-year orders) ===')
pd.set_option('display.float_format','{:.2f}'.format,'display.width',160)
print(cohort_profile.to_string(index=False))
print()

merged = yr1.rename('yr1_ret').reset_index()
merged.columns = ['cohort_year','yr1_ret']
merged = merged.merge(cohort_profile, on='cohort_year')

print('=== CORRELATION: Year+1 Retention vs Cohort Metrics ===')
for col in ['return_rate','avg_gm_pct','pct_neg','avg_disc','avg_rev','avg_n_items']:
    r = merged['yr1_ret'].corr(merged[col])
    sig = '*** STRONG' if abs(r)>0.6 else ('** moderate' if abs(r)>0.35 else 'weak')
    direction = '+' if r>0 else '-'
    print(f'  {col:18s}: r={r:+.3f}  {sig}')
print()

rets_yr = rets.merge(orders[['order_id','order_date']], on='order_id', how='left')
rets_yr['year'] = pd.to_datetime(rets_yr['order_date']).dt.year
reason_yr = (rets_yr.groupby(['year','return_reason'])['return_id']
             .count().unstack(fill_value=0))
reason_pct = reason_yr.div(reason_yr.sum(axis=1), axis=0) * 100

print('=== RETURN REASON MIX (% per year) ===')
print(reason_pct.round(1).to_string())
print()

quality_yr = (ord_full.groupby('order_year').agg(
    n_orders    = ('order_id','nunique'),
    return_rate = ('returned','mean'),
    avg_gm      = ('order_gm_pct','mean'),
    pct_neg     = ('has_neg','mean'),
    avg_disc    = ('disc_pct','mean'),
).reset_index())
quality_yr['return_rate'] *= 100
quality_yr['pct_neg']     *= 100

print('=== ORDER QUALITY TREND by Year ===')
print(quality_yr.to_string(index=False))
print()

ret_by_offset = (ord_full[ord_full['period_offset'].between(0,5)]
                 .groupby('period_offset').agg(
    n_orders    = ('order_id','nunique'),
    return_rate = ('returned','mean'),
    avg_gm      = ('order_gm_pct','mean'),
    pct_neg     = ('has_neg','mean'),
).reset_index())
ret_by_offset['return_rate'] *= 100
ret_by_offset['pct_neg']     *= 100

print('=== ORDER QUALITY by PERIOD OFFSET (Year+N sau cohort) ===')
print('(Khach mua lan thu N co bi return/margin te hon khong?)')
print(ret_by_offset.to_string(index=False))

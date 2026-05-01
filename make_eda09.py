"""
EDA-09: Order Fulfilment & SLA | Return Analysis | Inventory Health
Sections 3.1, 3.2, 3.3
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json, pathlib, textwrap
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')

NB = pathlib.Path('eda_09_fulfilment_returns_inventory.ipynb')

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

def md(text):
    return {"cell_type":"markdown","metadata":{},"source":text}

SETUP = r"""
import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import warnings; warnings.filterwarnings('ignore')

orders  = pd.read_csv('orders.csv',      parse_dates=['order_date'])
ship    = pd.read_csv('shipments.csv',   parse_dates=['ship_date','delivery_date'])
geo     = pd.read_csv('geography.csv')
inv     = pd.read_csv('inventory.csv',   parse_dates=['snapshot_date'])
rets    = pd.read_csv('returns.csv',     parse_dates=['return_date'])
prods   = pd.read_csv('products.csv')
items   = pd.read_csv('order_items.csv', low_memory=False)
pays    = pd.read_csv('payments.csv')

ship_full = ship.merge(orders[['order_id','order_date','zip','order_status']], on='order_id', how='left')
ship_full = ship_full.merge(geo[['zip','region']], on='zip', how='left')
ship_full['fulfil_days'] = (ship_full['delivery_date'] - ship_full['order_date']).dt.days
ship_full['time_to_ship'] = (ship_full['ship_date']    - ship_full['order_date']).dt.days
ship_full['order_year']   = ship_full['order_date'].dt.year

items_p = items.merge(prods[['product_id','category','cogs']], on='product_id', how='left')
items_p['rev_line']  = items_p['quantity'] * items_p['unit_price']
items_p['cogs_line'] = items_p['quantity'] * items_p['cogs']

rets_p = rets.merge(pays[['order_id','payment_value']], on='order_id', how='left')
rets_p['refund_capped'] = rets_p[['refund_amount','payment_value']].min(axis=1)

print('Data loaded successfully.')
print(f"  orders     : {len(orders):,}")
print(f"  shipments  : {len(ship):,}")
print(f"  returns    : {len(rets):,}")
print(f"  inventory  : {len(inv):,}")
"""


P31_STATUS = r"""
total_orders = len(orders)
status_ct    = orders['order_status'].value_counts()
status_pct   = status_ct / total_orders * 100

print('=== ORDER STATUS MIX ===')
for s, n in status_ct.items():
    print(f'  {s:12s}: {n:7,}  ({n/total_orders*100:5.1f}%)')
print()

cancel_rate = status_pct.get('cancelled', 0)
return_rate = status_pct.get('returned',  0)
print(f'Cancellation rate : {cancel_rate:.2f}%')
print(f'Post-ship return  : {return_rate:.2f}%')
print(f'Loss rate total   : {cancel_rate + return_rate:.2f}%')
"""

P31_SLA = r"""
SLA_DAYS = 7          # business SLA threshold

print(f'=== FULFILMENT TIME DISTRIBUTION (SLA = {SLA_DAYS}d) ===')
print(ship_full['fulfil_days'].describe().round(2).to_string())
print()

sla_breach = (ship_full['fulfil_days'] > SLA_DAYS).mean() * 100
print(f'Overall SLA breach rate (>{SLA_DAYS}d) : {sla_breach:.1f}%')
print()

bins   = [0,3,5,7,10]
labels = ['<=3d','4-5d','6-7d','8-10d']
ship_full['fd_bucket'] = pd.cut(ship_full['fulfil_days'], bins=bins, labels=labels)
bucket_ct  = ship_full['fd_bucket'].value_counts().sort_index()
bucket_pct = bucket_ct / len(ship_full) * 100
print('Fulfilment-day bucket breakdown:')
for b, pct in bucket_pct.items():
    print(f'  {b} : {pct:5.1f}%  (n={bucket_ct[b]:,})')
print()
print('Time-to-ship (order→dispatch) stats:')
print(ship_full['time_to_ship'].describe().round(2).to_string())
"""

P31_REGION = r"""
SLA_DAYS = 7

region_sla = ship_full.groupby('region').agg(
    n_shipments   = ('order_id',    'count'),
    avg_fulfil    = ('fulfil_days', 'mean'),
    median_fulfil = ('fulfil_days', 'median'),
    p90_fulfil    = ('fulfil_days', lambda x: x.quantile(0.9)),
    sla_breach_pct= ('fulfil_days', lambda x: (x > SLA_DAYS).mean() * 100),
    avg_tts       = ('time_to_ship','mean'),
).reset_index().sort_values('sla_breach_pct', ascending=False)

pd.set_option('display.float_format','{:.2f}'.format,'display.width',160)
print('=== REGIONAL FULFILMENT METRICS ===')
print(region_sla.to_string(index=False))
print()

ord_geo = orders.merge(geo[['zip','region']], on='zip', how='left')
cancel_region = (ord_geo.groupby('region')
                 .apply(lambda x: (x['order_status']=='cancelled').mean() * 100)
                 .rename('cancel_pct').reset_index())
print('Cancellation rate by region:')
print(cancel_region.to_string(index=False))
"""

P31_TREND = r"""
ful_yr = ship_full.groupby('order_year').agg(
    n_ship     = ('order_id',    'count'),
    avg_fd     = ('fulfil_days', 'mean'),
    sla_breach = ('fulfil_days', lambda x: (x>7).mean()*100),
    avg_tts    = ('time_to_ship','mean'),
).reset_index()
print('=== FULFILMENT TREND by YEAR ===')
print(ful_yr.to_string(index=False))
"""

P31_CHART = r"""
SLA_DAYS = 7
region_sla = ship_full.groupby('region').agg(
    n_shipments   = ('order_id',    'count'),
    avg_fulfil    = ('fulfil_days', 'mean'),
    sla_breach_pct= ('fulfil_days', lambda x: (x > SLA_DAYS).mean() * 100),
    avg_tts       = ('time_to_ship','mean'),
).reset_index().sort_values('sla_breach_pct', ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

ax = axes[0]
clrs = ['#e74c3c','#e67e22','#27ae60']
ax.bar(region_sla['region'], region_sla['sla_breach_pct'], color=clrs)
ax.axhline(region_sla['sla_breach_pct'].mean(), ls='--', color='gray', label='avg')
ax.set_title('SLA Breach % by Region (>7d)')
ax.set_ylabel('Breach %')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
ax.legend()

ax = axes[1]
ax.bar(region_sla['region'], region_sla['avg_fulfil'], color=['#3498db','#9b59b6','#1abc9c'])
ax.set_title('Avg Fulfilment Days by Region')
ax.set_ylabel('Days')
ax.set_ylim(0, 10)

ax = axes[2]
for reg, grp in ship_full.groupby('region'):
    ax.hist(grp['fulfil_days'], bins=range(2,12), alpha=0.5, label=reg, density=True)
ax.axvline(SLA_DAYS, color='red', ls='--', label=f'SLA={SLA_DAYS}d')
ax.set_title('Fulfilment Days Distribution by Region')
ax.set_xlabel('Days')
ax.legend()

plt.suptitle('3.1  Order Fulfilment & SLA  —  Regional Breakdown', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_eda09_31_sla.png', dpi=130, bbox_inches='tight')
plt.show()
print('fig_eda09_31_sla.png saved.')
"""


P32_RATE = r"""
items_cat = items.merge(prods[['product_id','category']], on='product_id', how='left')
rets_cat  = rets.merge(prods[['product_id','category']], on='product_id', how='left')

cat_ord   = items_cat.groupby('category')['order_id'].nunique().rename('n_orders')
cat_ret   = rets_cat.groupby('category')['return_id'].count().rename('n_returns')
cat_qty   = rets_cat.groupby('category')['return_quantity'].sum().rename('ret_units')
cat_ref   = rets_cat.groupby('category')['refund_amount'].sum().rename('gross_refund')

cat_df = pd.concat([cat_ord, cat_ret, cat_qty, cat_ref], axis=1).reset_index()
cat_df['ret_rate_pct'] = cat_df['n_returns'] / cat_df['n_orders'] * 100

pd.set_option('display.float_format','{:,.0f}'.format)
print('=== RETURN RATE BY CATEGORY ===')
print(cat_df.to_string(index=False))
print()

print('=== TOP RETURN REASONS (overall) ===')
reason_ct = rets['return_reason'].value_counts()
for r, n in reason_ct.items():
    print(f'  {r:20s}: {n:6,}  ({n/len(rets)*100:.1f}%)')
print()

print('=== RETURN REASON MIX BY CATEGORY (%) ===')
rc = rets_cat.groupby(['category','return_reason'])['return_id'].count().unstack(fill_value=0)
rc_pct = rc.div(rc.sum(axis=1), axis=0) * 100
pd.set_option('display.float_format','{:.1f}'.format)
print(rc_pct.to_string())
"""

P32_REV = r"""
gross_rev   = (items_p['rev_line']).sum()
total_cogs  = (items_p['cogs_line']).sum()
total_disc  = items['discount_amount'].sum()
gross_refund= rets['refund_amount'].sum()
rets_pay    = rets.merge(pays[['order_id','payment_value']], on='order_id', how='left')
rets_pay['refund_capped'] = rets_pay[['refund_amount','payment_value']].min(axis=1)
total_refund_capped = rets_pay['refund_capped'].sum()

net_rev     = gross_rev - total_disc - total_refund_capped
gross_margin= gross_rev - total_cogs
gm_pct      = gross_margin / net_rev * 100
npm_pct     = (net_rev - total_cogs) / net_rev * 100

fmt = lambda x: f'{x/1e9:,.1f}B'
print('=== NET REVENUE EROSION WATERFALL ===')
print(f'  Gross Revenue           : {fmt(gross_rev)}')
print(f'  – Discounts             : {fmt(total_disc)}  ({total_disc/gross_rev*100:.1f}%)')
print(f'  – Refunds (capped)      : {fmt(total_refund_capped)}  ({total_refund_capped/gross_rev*100:.1f}%)')
print(f'  = Net Revenue           : {fmt(net_rev)}')
print(f'  – COGS                  : {fmt(total_cogs)}')
print(f'  = Gross Profit          : {fmt(gross_margin)}')
print()
print(f'  Gross Margin %          : {gm_pct:.2f}%')
print(f'  Net Profit Margin %     : {npm_pct:.2f}%')
print()
print(f'  Uncapped refund total   : {fmt(gross_refund)}')
print(f'  Overpay shielded        : {fmt(gross_refund-total_refund_capped)}')

rets_yr = rets_pay.merge(orders[['order_id','order_date']], on='order_id', how='left')
rets_yr['year'] = pd.to_datetime(rets_yr['order_date']).dt.year
yr_erosion = rets_yr.groupby('year').agg(
    n_returns       = ('return_id','count'),
    refund_capped   = ('refund_capped','sum'),
    gross_refund    = ('refund_amount','sum'),
).reset_index()
yr_rev = items_p.merge(orders[['order_id','order_date']], on='order_id', how='left')
yr_rev['year'] = yr_rev['order_date'].dt.year
yr_gross = yr_rev.groupby('year')['rev_line'].sum().rename('gross_rev')
yr_erosion = yr_erosion.merge(yr_gross, on='year')
yr_erosion['erosion_pct'] = yr_erosion['refund_capped'] / yr_erosion['gross_rev'] * 100

pd.set_option('display.float_format','{:,.2f}'.format,'display.width',160)
print()
print('=== REFUND EROSION BY YEAR ===')
print(yr_erosion.to_string(index=False))
"""

P32_CHART = r"""
items_cat = items.merge(prods[['product_id','category']], on='product_id', how='left')
rets_cat  = rets.merge(prods[['product_id','category']], on='product_id', how='left')
cat_ord   = items_cat.groupby('category')['order_id'].nunique().rename('n_orders')
cat_ret   = rets_cat.groupby('category')['return_id'].count().rename('n_returns')
cat_df    = pd.concat([cat_ord, cat_ret], axis=1).reset_index()
cat_df['ret_rate_pct'] = cat_df['n_returns'] / cat_df['n_orders'] * 100

reason_ct = rets['return_reason'].value_counts()

rets_pay2 = rets.merge(pays[['order_id','payment_value']], on='order_id', how='left')
rets_pay2['refund_capped'] = rets_pay2[['refund_amount','payment_value']].min(axis=1)
rets_yr2  = rets_pay2.merge(orders[['order_id','order_date']], on='order_id', how='left')
rets_yr2['year'] = pd.to_datetime(rets_yr2['order_date']).dt.year
items_p2  = items.merge(prods[['product_id','cogs']], on='product_id', how='left')
items_p2['rev_line'] = items_p2['quantity'] * items_p2['unit_price']
yr_rev2   = items_p2.merge(orders[['order_id','order_date']], on='order_id', how='left')
yr_rev2['year'] = yr_rev2['order_date'].dt.year
yr_gross2 = yr_rev2.groupby('year')['rev_line'].sum().rename('gross_rev')
yr_er2    = rets_yr2.groupby('year').agg(refund_capped=('refund_capped','sum')).reset_index()
yr_er2    = yr_er2.merge(yr_gross2, on='year')
yr_er2['erosion_pct'] = yr_er2['refund_capped'] / yr_er2['gross_rev'] * 100

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
cat_sorted = cat_df.sort_values('ret_rate_pct', ascending=True)
colors_cat = ['#27ae60' if r < 6 else '#e74c3c' for r in cat_sorted['ret_rate_pct']]
ax.barh(cat_sorted['category'], cat_sorted['ret_rate_pct'], color=colors_cat)
ax.axvline(cat_df['ret_rate_pct'].mean(), ls='--', color='gray', label='avg')
ax.set_title('Return Rate by Category')
ax.set_xlabel('Return Rate (%)')
ax.legend()

ax = axes[1]
ax.pie(reason_ct.values, labels=reason_ct.index, autopct='%1.1f%%', startangle=90,
       colors=['#3498db','#e74c3c','#f39c12','#2ecc71','#9b59b6'])
ax.set_title('Return Reason Distribution')

ax = axes[2]
ax.bar(yr_er2['year'], yr_er2['erosion_pct'], color='#c0392b', alpha=0.8)
ax.set_title('Refund Erosion as % of Gross Revenue')
ax.set_ylabel('Erosion %')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

plt.suptitle('3.2  Return Analysis  —  Category, Reasons & Revenue Erosion', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_eda09_32_returns.png', dpi=130, bbox_inches='tight')
plt.show()
print('fig_eda09_32_returns.png saved.')
"""


P33_FREQ = r"""
total_snaps = len(inv)
n_stockout  = (inv['stockout_flag']==1).sum()
n_overstock = (inv['overstock_flag']==1).sum()
n_both      = ((inv['stockout_flag']==1) & (inv['overstock_flag']==1)).sum()
n_reorder   = (inv['reorder_flag']==1).sum()

print('=== INVENTORY FLAG SUMMARY ===')
print(f'  Total snapshots       : {total_snaps:,}')
print(f'  stockout_flag=1       : {n_stockout:,}  ({n_stockout/total_snaps*100:.1f}%)')
print(f'  overstock_flag=1      : {n_overstock:,}  ({n_overstock/total_snaps*100:.1f}%)')
print(f'  BOTH flags = 1        : {n_both:,}  ({n_both/total_snaps*100:.1f}%)  ← paradox')
print(f'  reorder_flag=1        : {n_reorder:,}  ({n_reorder/total_snaps*100:.1f}%)')
print()

cat_flags = inv.groupby('category').agg(
    n_snaps     = ('product_id','count'),
    pct_stockout= ('stockout_flag','mean'),
    pct_over    = ('overstock_flag','mean'),
    pct_both    = ('stockout_flag', lambda x:
                   ((x==1) & (inv.loc[x.index,'overstock_flag']==1)).mean()),
    avg_fill    = ('fill_rate','mean'),
).reset_index()
cat_flags['pct_stockout'] *= 100
cat_flags['pct_over']     *= 100
cat_flags['pct_both']     *= 100
pd.set_option('display.float_format','{:.2f}'.format,'display.width',160)
print('=== FLAG % BY CATEGORY ===')
print(cat_flags.to_string(index=False))
"""

P33_FILLRATE = r"""
inv_yr = inv.groupby('year').agg(
    n_snaps     = ('product_id','count'),
    avg_fill    = ('fill_rate','mean'),
    pct_stockout= ('stockout_flag','mean'),
    pct_over    = ('overstock_flag','mean'),
    pct_both    = ('stockout_flag', lambda x:
                   ((x==1) & (inv.loc[x.index,'overstock_flag']==1)).mean()),
    avg_dos     = ('days_of_supply','mean'),
).reset_index()
inv_yr['pct_stockout'] *= 100
inv_yr['pct_over']     *= 100
inv_yr['pct_both']     *= 100

pd.set_option('display.float_format','{:.2f}'.format,'display.width',160)
print('=== INVENTORY HEALTH TREND by YEAR ===')
print(inv_yr.to_string(index=False))
print()

overstock_only = inv[(inv['overstock_flag']==1) & (inv['stockout_flag']==0)]
top_over = (overstock_only.groupby(['product_id','product_name','category'])
            .agg(n_snaps=('snapshot_date','count'),
                 avg_dos=('days_of_supply','mean'),
                 avg_stock=('stock_on_hand','mean'))
            .reset_index()
            .sort_values('avg_dos', ascending=False).head(10))
print('=== TOP OVERSTOCKED PRODUCTS (pure overstock, no stockout) ===')
print(top_over.to_string(index=False))
"""

P33_PARADOX = r"""
both = inv[(inv['stockout_flag']==1) & (inv['overstock_flag']==1)].copy()

print('=== SIMULTANEOUS STOCKOUT + OVERSTOCK ANALYSIS ===')
print(f'Rows with both flags: {len(both):,} / {len(inv):,} ({len(both)/len(inv)*100:.1f}%)')
print()
print('Key metrics for paradox snapshots:')
print(f'  avg stock_on_hand : {both["stock_on_hand"].mean():.1f}  (overall avg: {inv["stock_on_hand"].mean():.1f})')
print(f'  avg days_of_supply: {both["days_of_supply"].mean():.1f}  (overall avg: {inv["days_of_supply"].mean():.1f})')
print(f'  avg stockout_days : {both["stockout_days"].mean():.2f}  (overall avg: {inv["stockout_days"].mean():.2f})')
print(f'  avg fill_rate     : {both["fill_rate"].mean():.4f}  (overall avg: {inv["fill_rate"].mean():.4f})')
print()

print('INTERPRETATION:')
print('  stockout_flag reflects *historical* stockout_days in the period.')
print('  overstock_flag reflects *current* days_of_supply being excessive.')
print('  Both = 1 means: the product ran out, triggered a large reorder,')
print('  and now sits with surplus inventory — classic bull-whip reorder overcompensation.')
print()

print('stockout_days when BOTH=1 :', both['stockout_days'].describe().round(2).to_string())
print()
not_both = inv[~((inv['stockout_flag']==1) & (inv['overstock_flag']==1))]
print('stockout_days when NOT both:', not_both['stockout_days'].describe().round(2).to_string())
print()

paradox_products = (both.groupby(['product_id','product_name','category'])
                    .agg(n_paradox_snaps=('snapshot_date','count'),
                         avg_stockout_days=('stockout_days','mean'),
                         avg_dos=('days_of_supply','mean'),
                         avg_stock=('stock_on_hand','mean'))
                    .reset_index()
                    .sort_values('n_paradox_snaps', ascending=False).head(15))
pd.set_option('display.float_format','{:.2f}'.format,'display.width',160)
print('=== TOP PRODUCTS WITH PARADOX (stockout then overstock) ===')
print(paradox_products.to_string(index=False))
"""

P33_RATING = r"""
reviews = pd.read_csv('reviews.csv', parse_dates=['review_date'])
reviews['year']  = reviews['review_date'].dt.year
reviews['month'] = reviews['review_date'].dt.month

reviews['ym'] = reviews['year'] * 100 + reviews['month']
inv['ym']     = inv['year']    * 100 + inv['month']

inv_pm = inv.groupby(['product_id','ym']).agg(
    stockout_flag  = ('stockout_flag','max'),
    overstock_flag = ('overstock_flag','max'),
    avg_fill       = ('fill_rate','mean'),
    stockout_days  = ('stockout_days','mean'),
).reset_index()

rev_pm = reviews.groupby(['product_id','ym']).agg(
    avg_rating = ('rating','mean'),
    n_reviews  = ('review_id','count'),
).reset_index()

merged = rev_pm.merge(inv_pm, on=['product_id','ym'], how='inner')
print(f'Product-month pairs matched: {len(merged):,}')
print()

for col in ['stockout_flag','stockout_days','avg_fill']:
    r = merged['avg_rating'].corr(merged[col])
    sig = '*** STRONG' if abs(r)>0.3 else ('** moderate' if abs(r)>0.1 else 'weak')
    print(f'  rating vs {col:18s}: r={r:+.4f}  {sig}')
print()

print('Avg rating by stockout_flag:')
print(merged.groupby('stockout_flag')['avg_rating'].agg(['mean','count']).round(3).to_string())
print()

merged['fill_q'] = pd.qcut(merged['avg_fill'], q=5, labels=False, duplicates='drop')
print('Avg rating by fill-rate bucket (0=lowest fill):')
print(merged.groupby('fill_q')['avg_rating'].mean().round(3).to_string())
"""

P33_CHART = r"""
inv_yr = inv.groupby('year').agg(
    avg_fill    = ('fill_rate','mean'),
    pct_stockout= ('stockout_flag','mean'),
    pct_over    = ('overstock_flag','mean'),
    pct_both    = ('stockout_flag', lambda x:
                   ((x==1) & (inv.loc[x.index,'overstock_flag']==1)).mean()),
).reset_index()
inv_yr['pct_stockout'] *= 100
inv_yr['pct_over']     *= 100
inv_yr['pct_both']     *= 100

cat_flags2 = inv.groupby('category').agg(
    pct_stockout= ('stockout_flag','mean'),
    pct_over    = ('overstock_flag','mean'),
    avg_fill    = ('fill_rate','mean'),
).reset_index()
cat_flags2['pct_stockout'] *= 100
cat_flags2['pct_over']     *= 100

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.plot(inv_yr['year'], inv_yr['avg_fill']*100, 'b-o', label='Fill Rate')
ax.set_title('Average Fill Rate by Year')
ax.set_ylabel('Fill Rate (%)')
ax.set_ylim(85, 100)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
ax2 = ax.twinx()
ax2.plot(inv_yr['year'], inv_yr['pct_stockout'], 'r--s', alpha=0.7, label='% Stockout')
ax2.set_ylabel('Stockout %', color='red')
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
lines1, _ = ax.get_legend_handles_labels()
lines2, _ = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, ['Fill Rate','% Stockout'], loc='upper left')

ax = axes[1]
ax.bar(inv_yr['year'], inv_yr['pct_both'], color='#8e44ad', alpha=0.8)
ax.set_title('Simultaneous Stockout+Overstock\n(% of snapshots)')
ax.set_ylabel('% Snapshots')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

ax = axes[2]
x = range(len(cat_flags2))
w = 0.35
ax.bar([i-w/2 for i in x], cat_flags2['pct_stockout'], w, label='Stockout%', color='#e74c3c', alpha=0.8)
ax.bar([i+w/2 for i in x], cat_flags2['pct_over'],     w, label='Overstock%',color='#3498db', alpha=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels(cat_flags2['category'])
ax.set_title('Stockout vs Overstock Rate by Category')
ax.set_ylabel('%')
ax.legend()

plt.suptitle('3.3  Inventory Health  —  Fill Rate, Paradox & Category Breakdown', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_eda09_33_inventory.png', dpi=130, bbox_inches='tight')
plt.show()
print('fig_eda09_33_inventory.png saved.')
"""

cells = [
    md('# EDA-09: Order Fulfilment & SLA | Return Analysis | Inventory Health\n\n**Sections 3.1, 3.2, 3.3**'),
    code(SETUP),

    md('---\n## 3.1  Order Fulfilment & SLA\n### 3.1a  Cancellation & order status mix'),
    code(P31_STATUS),

    md('### 3.1b  Fulfilment time distribution & SLA breach'),
    code(P31_SLA),

    md('### 3.1c  Regional SLA breakdown (geographic bottleneck)'),
    code(P31_REGION),

    md('### 3.1d  Fulfilment trend by year'),
    code(P31_TREND),

    md('### 3.1e  Chart — regional SLA comparison'),
    code(P31_CHART),

    md('---\n## 3.2  Return Analysis\n### 3.2a  Return rate by category & reason'),
    code(P32_RATE),

    md('### 3.2b  Net revenue erosion waterfall'),
    code(P32_REV),

    md('### 3.2c  Chart — return analysis overview'),
    code(P32_CHART),

    md('---\n## 3.3  Inventory Health\n### 3.3a  Stockout & overstock flag frequency'),
    code(P33_FREQ),

    md('### 3.3b  Fill rate trend & top overstocked products'),
    code(P33_FILLRATE),

    md('### 3.3c  Simultaneous stockout + overstock paradox'),
    code(P33_PARADOX),

    md('### 3.3d  Stockout–rating correlation'),
    code(P33_RATING),

    md('### 3.3e  Chart — inventory health summary'),
    code(P33_CHART),
]

nb_json = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.9.0"}
    },
    "cells": cells
}

NB.write_text(json.dumps(nb_json, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'Notebook written: {NB}  ({len(cells)} cells)')

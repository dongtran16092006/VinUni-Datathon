import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')

items  = pd.read_csv('order_items.csv', low_memory=False)
prods  = pd.read_csv('products.csv')
orders = pd.read_csv('orders.csv', parse_dates=['order_date'])
rets   = pd.read_csv('returns.csv')
pays   = pd.read_csv('payments.csv')

items = items.merge(
    prods[['product_id','category','segment','cogs','price']],
    on='product_id', how='left'
)
items['rev_line']  = items['quantity'] * items['unit_price']
items['cogs_line'] = items['quantity'] * items['cogs']

prod_base = items.groupby('product_id').agg(
    price_list=('price','first'), units_sold=('quantity','sum'),
    gross_rev=('rev_line','sum'), cogs_total=('cogs_line','sum'),
    discount_total=('discount_amount','sum'),
).reset_index()
rets_pay = rets.merge(pays[['order_id','payment_value']], on='order_id', how='left')
rets_pay['refund_capped'] = rets_pay[['refund_amount','payment_value']].min(axis=1)
prod_refund = rets_pay.groupby('product_id')['refund_capped'].sum().reset_index()
prod_refund.columns = ['product_id','refund_capped']
prod = prod_base.merge(prod_refund, on='product_id', how='left')
prod['refund_capped'] = prod['refund_capped'].fillna(0)
prod['net_rev']    = prod['gross_rev'] - prod['discount_total'] - prod['refund_capped']
prod['gross_prof'] = prod['gross_rev'] - prod['cogs_total']
prod['gm_pct']     = prod['gross_prof'] / prod['net_rev'] * 100
PORT_GM = prod['gross_prof'].sum() / prod['net_rev'].sum() * 100

pid_tag = {}
for _, r in prod.iterrows():
    if r['gm_pct'] < 0:       pid_tag[r['product_id']] = 'negative'
    elif r['gm_pct'] >= PORT_GM: pid_tag[r['product_id']] = 'above_avg'
    else:                      pid_tag[r['product_id']] = 'mid'

items['margin_tag'] = items['product_id'].map(pid_tag)

tag_dummies = pd.get_dummies(items[['order_id','margin_tag']].drop_duplicates(),
                              columns=['margin_tag'], prefix='', prefix_sep='')
order_tags = tag_dummies.groupby('order_id').max().reset_index()
for col in ['negative','mid','above_avg']:
    if col not in order_tags.columns:
        order_tags[col] = 0

order_tags['only_neg']      = (order_tags['negative']==1) & (order_tags['mid']==0) & (order_tags['above_avg']==0)
order_tags['only_pos']      = (order_tags['negative']==0) & (order_tags['mid']==0) & (order_tags['above_avg']==1)
order_tags['mixed_neg_pos'] = (order_tags['negative']==1) & (order_tags['above_avg']==1)
order_tags['has_neg']       = order_tags['negative'] == 1

print('=== 1. BASKET COMPOSITION ===')
tot = len(order_tags)
print(f'Total orders                   : {tot:,}')
print(f'Orders co san pham margin am   : {order_tags["has_neg"].sum():,}  ({order_tags["has_neg"].mean()*100:.1f}%)')
print(f'  CHI co san pham am           : {order_tags["only_neg"].sum():,}  ({order_tags["only_neg"].mean()*100:.1f}%)')
print(f'  am + above_avg (mixed)       : {order_tags["mixed_neg_pos"].sum():,}  ({order_tags["mixed_neg_pos"].mean()*100:.1f}%)')
print(f'Orders CHI co above_avg        : {order_tags["only_pos"].sum():,}  ({order_tags["only_pos"].mean()*100:.1f}%)')
print()

mixed_oids = set(order_tags[order_tags['mixed_neg_pos']]['order_id'])
mixed_items = items[items['order_id'].isin(mixed_oids)]
gp_by_tag = (mixed_items.groupby('margin_tag')
             .apply(lambda g: (g['rev_line'] - g['cogs_line']).sum()))
print('GP trong mixed orders (am + above_avg):')
for tag in ['negative','mid','above_avg']:
    val = gp_by_tag.get(tag, 0)
    print(f'  {tag:12s}: {val/1e6:+.1f} M VND')
print(f'  Net GP  : {gp_by_tag.sum()/1e6:+.1f} M VND')
print()

ord_cust = orders[['order_id','customer_id','order_date']].merge(
    items[['order_id','product_id','margin_tag','rev_line','cogs_line']],
    on='order_id', how='left'
)

ord_cust_sorted = ord_cust.sort_values('order_date')
first_item = (ord_cust_sorted.groupby('customer_id')[['margin_tag','order_date']]
              .first().reset_index())
first_item.columns = ['customer_id','first_margin','first_date']

lifetime = ord_cust.groupby('customer_id').agg(
    total_rev  = ('rev_line','sum'),
    total_cogs = ('cogs_line','sum'),
    n_orders   = ('order_id','nunique'),
).reset_index()
lifetime['lifetime_gp'] = lifetime['total_rev'] - lifetime['total_cogs']

cust_full = first_item.merge(lifetime, on='customer_id', how='left')

print('=== 2. CUSTOMER JOURNEY: First purchase margin -> Lifetime GP ===')
for tag in ['negative','mid','above_avg']:
    g = cust_full[cust_full['first_margin'] == tag]
    print(f'  first={tag:12s}: {len(g):>6,} custs  '
          f'avg LT-GP={g["lifetime_gp"].mean()/1e3:>7.1f}K  '
          f'avg orders={g["n_orders"].mean():.2f}  '
          f'repeat rate={(g["n_orders"]>1).mean()*100:.1f}%  '
          f'avg LT-rev={g["total_rev"].mean()/1e3:.1f}K')
print()

order_dom = (items.groupby(['order_id','margin_tag'])['rev_line'].sum()
             .reset_index()
             .sort_values('rev_line', ascending=False)
             .groupby('order_id').first()
             .reset_index()[['order_id','margin_tag']])
order_dom.columns = ['order_id','dominant_tag']

ord_ranked = (orders[['order_id','customer_id','order_date']]
              .merge(order_dom, on='order_id')
              .sort_values(['customer_id','order_date']))
ord_ranked['order_rank'] = ord_ranked.groupby('customer_id').cumcount() + 1

neg_first = set(first_item[first_item['first_margin']=='negative']['customer_id'])
neg_journey = ord_ranked[ord_ranked['customer_id'].isin(neg_first)]

print('=== 3. NEG-FIRST CUSTOMERS: subsequent order margin pattern ===')
for rank in [1,2,3,4,5]:
    g = neg_journey[neg_journey['order_rank'] == rank]
    if len(g) == 0: continue
    vc = g['dominant_tag'].value_counts(normalize=True)*100
    print(f'  Order #{rank} (n={len(g):>6,}): '
          + '  '.join([f'{t}={vc.get(t,0):.1f}%' for t in ['negative','mid','above_avg']]))
print()

print('=== 4. PRICE TIER vs MARGIN GROUP ===')
prod['price_tier'] = pd.qcut(
    prod['price_list'], 4,
    labels=['Budget(Q1)','Mid(Q2)','Premium(Q3)','Luxury(Q4)']
)
tier = prod.groupby('price_tier', observed=True).apply(lambda g: pd.Series({
    'n_total'      : len(g),
    'n_neg'        : (g['gm_pct']<0).sum(),
    'pct_neg'      : (g['gm_pct']<0).mean()*100,
    'n_above'      : (g['gm_pct']>=PORT_GM).sum(),
    'pct_above'    : (g['gm_pct']>=PORT_GM).mean()*100,
    'avg_list'     : g['price_list'].mean(),
    'avg_gm'       : g['gm_pct'].mean(),
    'rev_pct'      : g['gross_rev'].sum()/prod['gross_rev'].sum()*100,
})).reset_index()
pd.set_option('display.float_format','{:.1f}'.format,'display.width',160)
print(tier.to_string(index=False))
print()

print('=== 5. ORDER VALUE LIFT: orders with neg product vs without ===')
order_summary = items.groupby('order_id').agg(
    total_rev  = ('rev_line','sum'),
    total_gp   = ('rev_line', lambda x: (x - items.loc[x.index,'cogs_line']).sum()),
    n_skus     = ('product_id','nunique'),
).reset_index().merge(order_tags[['order_id','has_neg','only_neg','mixed_neg_pos']], on='order_id')

for label, mask in [
    ('All orders',               order_summary.index >= 0),
    ('No neg products',          ~order_summary['has_neg']),
    ('Has neg + other products', order_summary['has_neg'] & ~order_summary['only_neg']),
    ('ONLY neg products',        order_summary['only_neg']),
    ('neg + above_avg mixed',    order_summary['mixed_neg_pos']),
]:
    g = order_summary[mask]
    print(f'  {label:30s}: n={len(g):>6,}  avg_rev={g["total_rev"].mean()/1e3:>6.1f}K  '
          f'avg_GP={g["total_gp"].mean()/1e3:>+6.1f}K  avg_SKUs={g["n_skus"].mean():.2f}')

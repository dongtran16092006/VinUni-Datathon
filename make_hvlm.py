"""Generate fig_hvlm_analysis.png - High Volume Low Margin product analysis"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.ticker as mticker
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.dpi':120,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':0.3,'font.size':10})

items   = pd.read_csv('order_items.csv', low_memory=False)
prods   = pd.read_csv('products.csv')
rets    = pd.read_csv('returns.csv')
pays    = pd.read_csv('payments.csv')

items = items.merge(prods[['product_id','product_name','category','segment','cogs','price']], on='product_id', how='left')
items['rev_line']  = items['quantity'] * items['unit_price']
items['cogs_line'] = items['quantity'] * items['cogs']

prod_base = items.groupby('product_id').agg(
    product_name   = ('product_name','first'),
    category       = ('category','first'),
    segment        = ('segment','first'),
    price_list     = ('price','first'),
    units_sold     = ('quantity','sum'),
    gross_rev      = ('rev_line','sum'),
    cogs_total     = ('cogs_line','sum'),
    discount_total = ('discount_amount','sum'),
    n_orders       = ('order_id','nunique'),
).reset_index()

rets_pay = rets.merge(pays[['order_id','payment_value']], on='order_id', how='left')
rets_pay['refund_capped'] = rets_pay[['refund_amount','payment_value']].min(axis=1)
prod_refund = rets_pay.groupby('product_id')['refund_capped'].sum().reset_index()
prod_refund.columns = ['product_id','refund_capped']

prod = prod_base.merge(prod_refund, on='product_id', how='left')
prod['refund_capped']  = prod['refund_capped'].fillna(0)
prod['net_rev']        = prod['gross_rev'] - prod['discount_total'] - prod['refund_capped']
prod['gross_prof']     = prod['gross_rev'] - prod['cogs_total']
prod['net_prof']       = prod['net_rev']   - prod['cogs_total']
prod['gm_pct']         = prod['gross_prof'] / prod['net_rev'] * 100
prod['npm_pct']        = prod['net_prof']   / prod['net_rev'] * 100
prod['discount_rate']  = prod['discount_total'] / prod['gross_rev'] * 100
prod['refund_rate']    = prod['refund_capped']  / prod['gross_rev'] * 100
prod['unit_price_avg'] = prod['gross_rev'] / prod['units_sold']
prod['markup_pct']     = (prod['unit_price_avg'] - prod['price_list']) / prod['price_list'] * 100
prod['cogs_per_unit']  = prod['cogs_total'] / prod['units_sold']

PORT_GM  = prod['gross_prof'].sum() / prod['net_rev'].sum() * 100
PORT_NPM = prod['net_prof'].sum()   / prod['net_rev'].sum() * 100

neg = prod[prod['gm_pct'] < 0].copy()

bucket_order = ['Negative (<0%)', 'Near-zero (0-5%)', 'Below avg (5-15%)', 'Above avg (>=15%)']
BCOLORS = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60']

def classify(row):
    if row['gm_pct'] < 0:        return 'Negative (<0%)'
    elif row['gm_pct'] < 5:      return 'Near-zero (0-5%)'
    elif row['gm_pct'] < PORT_GM: return 'Below avg (5-15%)'
    else:                         return 'Above avg (>=15%)'

prod['margin_bucket'] = prod.apply(classify, axis=1)

cat_colors = {'Streetwear': '#4C72B0', 'Outdoor': '#DD8452', 'Casual': '#55A868', 'GenZ': '#C44E52'}

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

ax = axes[0]
for cat, grp in prod.groupby('category'):
    ax.scatter(grp['gross_rev']/1e6, grp['gm_pct'],
               alpha=0.4, s=grp['units_sold'].clip(0, 50000)/500 + 10,
               color=cat_colors.get(cat, '#aaa'), label=cat)
ax.axhline(0, color='black', lw=1.2, linestyle='--')
ax.axhline(PORT_GM, color='crimson', lw=1.2, linestyle=':', label=f'Portfolio {PORT_GM:.1f}%')
ax.set_xlabel('Gross Revenue (M VND)')
ax.set_ylabel('GM%')
ax.set_title('Revenue vs GM% per Product\n(bubble = units sold)')
ax.legend(fontsize=8)
n_neg = (prod['gm_pct'] < 0).sum()
ax.text(0.97, 0.05, f'Negative margin: {n_neg} products',
        transform=ax.transAxes, ha='right', fontsize=9, color='#e74c3c',
        bbox=dict(boxstyle='round', fc='lightyellow', ec='#e74c3c', alpha=0.9))

ax = axes[1]
bkt = prod.groupby('margin_bucket').agg(
    n_prod=('product_id','count'), rev=('gross_rev','sum')
).reindex(bucket_order)
x = np.arange(len(bucket_order))
bars = ax.bar(x, bkt['n_prod'], color=BCOLORS, alpha=0.85)
ax2 = ax.twinx()
ax2.plot(x, bkt['rev'] / prod['gross_rev'].sum() * 100,
         color='navy', marker='D', ms=6, lw=2, label='% Revenue')
ax2.set_ylabel('% of total revenue', color='navy')
ax2.tick_params(axis='y', colors='navy')
ax.set_xticks(x)
ax.set_xticklabels(bucket_order, fontsize=8, rotation=10, ha='right')
ax.set_ylabel('# Products')
ax.set_title('Margin Bucket: # Products & Revenue Share')
for bar, n in zip(bars, bkt['n_prod']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(int(n)), ha='center', fontsize=9)

ax = axes[2]
cat_neg = neg.groupby('category').agg(
    n_prods   = ('product_id','count'),
    gross_rev = ('gross_rev','sum'),
    gp_loss   = ('gross_prof','sum'),
).reset_index().sort_values('gross_rev')
colors_c = [cat_colors.get(c, '#aaa') for c in cat_neg['category']]
bars2 = ax.barh(cat_neg['category'], cat_neg['gross_rev']/1e6, color=colors_c, alpha=0.85)
ax2c = ax.twiny()
ax2c.barh(cat_neg['category'], cat_neg['gp_loss'].abs()/1e6,
          color=colors_c, alpha=0.3, hatch='///')
ax2c.set_xlabel('GP Loss abs (M VND)', color='#e74c3c')
ax2c.tick_params(axis='x', colors='#e74c3c')
for bar, (_, row) in zip(bars2, cat_neg.iterrows()):
    ax.text(row['gross_rev']/1e6 + 2, bar.get_y() + bar.get_height()/2,
            f"{int(row['n_prods'])} prods | loss {row['gp_loss']/1e6:.1f}M",
            va='center', fontsize=8)
ax.set_xlabel('Gross Revenue (M VND)')
ax.set_title('Negative-Margin Products by Category\n(hatched = GP loss magnitude)')

ax = axes[3]
bins = np.arange(-20, 10, 1)
ok = prod[prod['gm_pct'] >= 0]
ax.hist(ok['markup_pct'].clip(-20, 10), bins=bins,
        alpha=0.6, color='#27ae60', label=f'GM>=0 (n={len(ok)})')
ax.hist(neg['markup_pct'].clip(-20, 10), bins=bins,
        alpha=0.7, color='#e74c3c', label=f'GM<0 (n={len(neg)})')
ax.axvline(0, color='black', lw=1.5, linestyle='--', label='= List price')
ax.set_xlabel('Markup% vs List Price (unit_price_avg / price_list - 1)')
ax.set_ylabel('# Products')
ax.set_title('Root Cause: Actual Sell Price vs List Price\n(negative = sold below list; all 359 neg-margin are here)')
ax.legend(fontsize=9)

ax = axes[4]
top15 = neg.nlargest(15, 'gross_rev').reset_index(drop=True)
y = np.arange(len(top15))
ax.barh(y, top15['gross_rev']/1e6, color='#e74c3c', alpha=0.7, label='Gross Rev (M)')
ax.barh(y, top15['cogs_total']/1e6, color='#c0392b', alpha=0.3, hatch='///', label='COGS (M)')
ax.set_yticks(y)
ax.set_yticklabels([
    f"{r['product_name'][:18]} ({r['category'][:5]})"
    for _, r in top15.iterrows()
], fontsize=7)
ax.set_xlabel('M VND')
ax.set_title('Top 15 Negative-Margin Products\n(hatched bar = COGS exceeds revenue)')
ax.legend(fontsize=8, loc='lower right')
for i, (_, row) in enumerate(top15.iterrows()):
    ax.text(row['gross_rev']/1e6 + 1, i,
            f'GM={row["gm_pct"]:.1f}%  markup={row["markup_pct"]:.1f}%',
            va='center', fontsize=7, color='crimson')

ax = axes[5]
seg_gm = prod.groupby(['category', 'segment']).agg(
    avg_gm   = ('gm_pct', 'mean'),
    pct_neg  = ('gm_pct', lambda x: (x < 0).mean() * 100),
    total_rev= ('gross_rev', 'sum'),
).reset_index().sort_values('avg_gm')
colors_s = [cat_colors.get(c, '#aaa') for c in seg_gm['category']]
labels_s = [f"{r['segment']} ({r['category'][:5]})" for _, r in seg_gm.iterrows()]
bars4 = ax.barh(range(len(seg_gm)), seg_gm['avg_gm'], color=colors_s, alpha=0.8)
ax.axvline(0, color='black', lw=1.2)
ax.axvline(PORT_GM, color='crimson', lw=1.2, linestyle='--', label=f'Portfolio {PORT_GM:.1f}%')
ax.set_yticks(range(len(seg_gm)))
ax.set_yticklabels(labels_s, fontsize=8)
ax.set_xlabel('Avg GM% per product')
ax.set_title('Avg GM% by Category x Segment\n(label = % of products with negative margin)')
ax.legend(fontsize=8)
for bar, (_, row) in zip(bars4, seg_gm.iterrows()):
    xpos = row['avg_gm'] + 0.3 if row['avg_gm'] >= 0 else 0.3
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f'{row["pct_neg"]:.0f}% neg', va='center', fontsize=7.5)

plt.suptitle(
    f'High-Volume / Low-Margin Product Analysis\n'
    f'Portfolio GM={PORT_GM:.1f}%  NPM={PORT_NPM:.1f}%  |  '
    f'359 products (30.5% of revenue) have NEGATIVE gross margin  |  '
    f'Root cause: unit_price < COGS (sold below cost)',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig('fig_hvlm_analysis.png', bbox_inches='tight')
plt.show()
print('Saved: fig_hvlm_analysis.png')

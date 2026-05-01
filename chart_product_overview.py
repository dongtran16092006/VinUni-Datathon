import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

prods = pd.read_csv('products.csv')
items = pd.read_csv('order_items.csv', low_memory=False)

items2 = items.merge(
    prods[['product_id', 'product_name', 'category', 'segment', 'cogs']],
    on='product_id', how='left'
)
items2['revenue']      = items2['quantity'] * items2['unit_price'] - items2['discount_amount']
items2['gross_profit'] = items2['revenue'] - items2['quantity'] * items2['cogs']

prod_agg = items2.groupby(['product_id', 'product_name', 'category', 'segment']).agg(
    revenue      = ('revenue',      'sum'),
    gross_profit = ('gross_profit', 'sum'),
).reset_index()
prod_agg['margin_pct'] = prod_agg['gross_profit'] / prod_agg['revenue'] * 100

total_rev  = prod_agg['revenue'].sum()
avg_margin = items2['gross_profit'].sum() / items2['revenue'].sum() * 100

def group_agg(col):
    agg = items2.groupby(col).agg(
        revenue      = ('revenue',      'sum'),
        gross_profit = ('gross_profit', 'sum'),
    ).reset_index()
    agg['margin_pct'] = agg['gross_profit'] / agg['revenue'] * 100
    agg['rev_share']  = agg['revenue'] / total_rev * 100
    agg = agg.merge(
        prod_agg.groupby(col)['product_id'].count().rename('n_sku'),
        on=col
    )
    return agg

cat_agg = group_agg('category').sort_values('revenue', ascending=False).reset_index(drop=True)
seg_agg = group_agg('segment').sort_values('margin_pct', ascending=False).reset_index(drop=True)

top20 = prod_agg.nlargest(20, 'revenue')[
    ['product_name', 'category', 'segment', 'revenue', 'margin_pct']
].copy()
top20['rev_share'] = top20['revenue'] / total_rev * 100
top20 = top20.sort_values('revenue', ascending=True).reset_index(drop=True)

log_revs = np.log10(prod_agg['revenue'].values / 1e6)
log_med  = np.log10(prod_agg['revenue'].median() / 1e6)
log_mean = np.log10(prod_agg['revenue'].mean() / 1e6)

C_HI2   = '#5db83d'
C_WARN  = '#e07070'
C_WARN2 = '#c0392b'
C_LGRAY = '#e8e8e8'
C_TXT   = '#1a1a1a'
C_MID   = '#666666'
C_MUTED = '#aaaaaa'
C_BG    = '#ffffff'
C_GRID  = '#f0f0f0'

fig = plt.figure(figsize=(16, 12.5), facecolor=C_BG)
fig.patch.set_facecolor(C_BG)

gs = fig.add_gridspec(
    2, 3,
    height_ratios=[1, 1.55],
    width_ratios=[1, 1, 1],
    hspace=0.42, wspace=0.38,
)

ax1 = fig.add_subplot(gs[0, 0])   # category
ax2 = fig.add_subplot(gs[0, 1])   # segment
ax3 = fig.add_subplot(gs[0, 2])   # SKU distribution
ax4 = fig.add_subplot(gs[1, :])   # top 20 SKUs (full width)

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor(C_BG)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#e0e0e0')
    ax.tick_params(colors=C_MID, labelsize=9)

cats    = cat_agg['category'].tolist()
rev_s   = cat_agg['rev_share'].tolist()
mar_c   = cat_agg['margin_pct'].tolist()
n_sku_c = cat_agg['n_sku'].tolist()

bar_colors = [C_LGRAY] * len(cats)   # neutral; margin shown via text
ax1.barh(cats, rev_s, color=bar_colors, height=0.52)
ax1.invert_yaxis()

for i, (rv, mg, n) in enumerate(zip(rev_s, mar_c, n_sku_c)):
    ax1.text(rv + 0.4, i, f'{rv:.1f}%', va='center', fontsize=10.5,
             color=C_MID, fontweight='bold')
    col = C_HI2 if mg >= avg_margin else C_WARN2
    ax1.text(rv + 0.4, i + 0.35, f'Margin {mg:.1f}%  ({n} SKU)',
             va='center', fontsize=7.5, color=col)


ax1.set_xlim(0, 105)
ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax1.grid(axis='x', color=C_GRID, lw=0.8)
ax1.set_xlabel('% doanh thu', fontsize=9, color=C_MID)
ax1.set_title('① Doanh thu & biên theo danh mục',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)
ax1.text(0.5, -0.14,
    f'Avg margin toàn danh mục: {avg_margin:.1f}%',
    transform=ax1.transAxes, ha='center', fontsize=8, color=C_MUTED, style='italic')

segs    = seg_agg['segment'].tolist()
rev_ss  = seg_agg['rev_share'].tolist()
mar_s   = seg_agg['margin_pct'].tolist()
n_sku_s = seg_agg['n_sku'].tolist()

seg_bar_colors = [C_HI2 if m >= avg_margin else C_WARN for m in mar_s]
ax2.barh(segs, rev_ss, color=seg_bar_colors, height=0.52, alpha=0.55)
ax2.invert_yaxis()

for i, (rv, mg, n) in enumerate(zip(rev_ss, mar_s, n_sku_s)):
    col = C_HI2 if mg >= avg_margin else C_WARN2
    ax2.text(rv + 0.2, i, f'{rv:.1f}%  ·  {mg:.1f}%',
             va='center', fontsize=9, color=col)

ax2.set_xlim(0, 55)
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax2.grid(axis='x', color=C_GRID, lw=0.8)
ax2.set_xlabel('% doanh thu', fontsize=9, color=C_MID)
ax2.set_title('② Doanh thu & biên theo phân khúc',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)

from matplotlib.patches import Patch
ax2.legend(
    handles=[Patch(fc=C_HI2, alpha=0.55, label=f'Margin ≥ {avg_margin:.1f}% (avg)'),
             Patch(fc=C_WARN, alpha=0.55, label=f'Margin < {avg_margin:.1f}% (avg)')],
    fontsize=7.5, framealpha=0, loc='lower right'
)

ax3.hist(log_revs, bins=35, color=C_LGRAY, edgecolor='white', linewidth=0.5, zorder=3)
ax3.grid(axis='y', color=C_GRID, lw=0.8, zorder=0)

ax3.axvline(log_med,  color=C_MID,  lw=1.5, ls='--', zorder=4)
ax3.axvline(log_mean, color=C_WARN, lw=1.5, ls='--', zorder=4)

ymax = ax3.get_ylim()[1]
ax3.text(log_med  - 0.05, ymax * 0.82, f'Median\n{10**log_med:.1f}M',
         ha='right', fontsize=8, color=C_MID)
ax3.text(log_mean + 0.05, ymax * 0.82, f'Mean\n{10**log_mean:.1f}M',
         ha='left',  fontsize=8, color=C_WARN)

tick_vals = [-2, -1, 0, 1, 2, 3]
tick_labs = ['0.01M', '0.1M', '1M', '10M', '100M', '1B']
ax3.set_xticks(tick_vals)
ax3.set_xticklabels(tick_labs, fontsize=8)
ax3.set_xlabel('Doanh thu SKU (log scale, VND)', fontsize=9, color=C_MID)
ax3.set_ylabel('Số SKU', fontsize=9, color=C_MID)

skew = float(pd.Series(prod_agg['revenue']).skew())
ax3.text(0.97, 0.96, f'Skewness: {skew:.1f}',
         transform=ax3.transAxes, ha='right', va='top', fontsize=8.5, color=C_MID,
         bbox=dict(fc=C_BG, ec='#e0e0e0', boxstyle='round,pad=0.3'))

ax3.set_title('③ Phân phối doanh thu theo SKU',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)

sku_names = top20['product_name'].tolist()
sku_rev   = top20['rev_share'].tolist()
sku_mar   = top20['margin_pct'].tolist()

sku_colors = [C_HI2 if m >= 0 else C_WARN for m in sku_mar]
bars4 = ax4.barh(range(20), sku_rev, color=sku_colors, height=0.65, alpha=0.7)

for i, (rv, mg) in enumerate(zip(sku_rev, sku_mar)):
    ax4.text(rv + 0.02, i, f'{rv:.2f}%', va='center', fontsize=8.5,
             color=C_MID, fontweight='bold')
    col = C_HI2 if mg >= 0 else C_WARN2
    ax4.text(rv + 0.25, i, f'Margin {mg:+.1f}%', va='center', fontsize=8, color=col)

ax4.set_yticks(range(20))
ax4.set_yticklabels(sku_names, fontsize=8.5)
ax4.invert_yaxis()

ax4.axvline(0, color='#e0e0e0', lw=1)
ax4.set_xlim(0, 3.5)
ax4.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
ax4.grid(axis='x', color=C_GRID, lw=0.8)
ax4.set_xlabel('% tổng doanh thu', fontsize=9, color=C_MID)

from matplotlib.patches import Patch as P
leg4 = ax4.legend(
    handles=[P(fc=C_HI2, alpha=0.7, label='Margin ≥ 0%'),
             P(fc=C_WARN, alpha=0.7, label='Margin < 0%')],
    fontsize=8.5, framealpha=0, loc='lower right'
)

ax4.set_title('④ Top 20 SKU theo doanh thu (tất cả thuộc Streetwear)',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)

fig.text(0.01, 0.962,
    f'1,598 SKU  |  4 danh mục  |  8 phân khúc  |  2012–2022',
    fontsize=8.5, color=C_MUTED, va='top')

plt.savefig('chart_product_overview.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print('Saved.')

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

total_rev = prod_agg['revenue'].sum()
n_total   = len(prod_agg)

prod_s = prod_agg.sort_values('revenue', ascending=False).reset_index(drop=True)
prod_s['cum_rev_pct'] = prod_s['revenue'].cumsum() / total_rev * 100
prod_s['sku_pct']     = (prod_s.index + 1) / n_total * 100
n80      = int((prod_s['cum_rev_pct'] <= 80).sum())
pct80_x  = n80 / n_total * 100          # x-coord of 80% revenue point

prod_agg['bucket'] = pd.cut(
    prod_agg['margin_pct'],
    bins=[-np.inf, 0, 10, 20, np.inf],
    labels=['neg', 'low', 'mid', 'high']
)
bkt_rev = prod_agg.groupby('bucket', observed=True)['revenue'].sum() / total_rev * 100
bkt_cnt = prod_agg.groupby('bucket', observed=True)['product_id'].count() / n_total * 100

prod_agg['rev_q'] = pd.qcut(prod_agg['revenue'], 5, labels=False)
q_labels = ['Q1\n(DT thấp)', 'Q2', 'Q3', 'Q4', 'Q5\n(DT cao)']
q_data   = [prod_agg[prod_agg['rev_q'] == i]['margin_pct'].dropna().values
            for i in range(5)]
q_medians = [np.median(d) for d in q_data]

corr = prod_agg['revenue'].corr(prod_agg['margin_pct'])

C_HI2   = '#5db83d'
C_WARN  = '#e07070'
C_WARN2 = '#c0392b'
C_LGRAY = '#e8e8e8'
C_TXT   = '#1a1a1a'
C_MID   = '#666666'
C_MUTED = '#aaaaaa'
C_BG    = '#ffffff'
C_GRID  = '#f0f0f0'

BKT_COLORS = [C_WARN2, C_WARN, '#c8c8c8', C_HI2]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.8),
                         gridspec_kw={'width_ratios': [1.1, 0.90, 1.25]},
                         facecolor=C_BG)
fig.patch.set_facecolor(C_BG)

for ax in axes:
    ax.set_facecolor(C_BG)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#e0e0e0')
    ax.tick_params(colors=C_MID, labelsize=9)

ax1 = axes[0]

sku_pcts  = prod_s['sku_pct'].values
cum_revs  = prod_s['cum_rev_pct'].values

ax1.plot([0, 100], [0, 100], color=C_MUTED, lw=1.0, ls='--', zorder=1,
         label='Phân phối đều')

ax1.fill_between(sku_pcts, sku_pcts, cum_revs,
                 alpha=0.15, color=C_HI2, zorder=2)

ax1.plot(sku_pcts, cum_revs, color=C_MID, lw=2.0, zorder=3)

ax1.axhline(80, color=C_WARN2, lw=1.0, ls='--', zorder=4)
ax1.axvline(pct80_x, color=C_WARN2, lw=1.0, ls='--', zorder=4)
ax1.scatter([pct80_x], [80], color=C_WARN2, s=55, zorder=6)

ax1.text(pct80_x + 1, 82,
         f'Top {pct80_x:.0f}% SKU ({n80}/{n_total})\n→ 80% doanh thu',
         fontsize=8.5, color=C_WARN2, fontweight='bold', va='bottom', linespacing=1.4)

ax1.text(65, 30,
         f'{n_total - n80} SKU còn lại\n→ 20% doanh thu',
         ha='center', fontsize=8, color=C_MID, linespacing=1.3)

ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)
ax1.set_xlabel('Tỷ lệ SKU tích luỹ (%, sắp xếp DT cao → thấp)', fontsize=9, color=C_MID)
ax1.set_ylabel('Doanh thu tích luỹ (%)', fontsize=9, color=C_MID)
ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax1.grid(color=C_GRID, lw=0.8)
ax1.set_title('① Mức độ tập trung doanh thu theo SKU',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)

leg = ax1.legend(fontsize=8, framealpha=0, loc='upper left')
leg.get_texts()[0].set_color(C_MUTED)

ax2 = axes[1]

bkt_order  = ['neg', 'low', 'mid', 'high']
bkt_labels = ['Âm\n(<0%)', 'Thấp\n(0–10%)', 'TB\n(10–20%)', 'Cao\n(≥20%)']
rev_vals   = [float(bkt_rev[b]) for b in bkt_order]
cnt_vals   = [float(bkt_cnt[b]) for b in bkt_order]

ax2.bar(range(4), rev_vals, color=BKT_COLORS, width=0.62, zorder=3)
ax2.grid(axis='y', color=C_GRID, lw=0.8, zorder=0)

for i, (rv, cv, col) in enumerate(zip(rev_vals, cnt_vals, BKT_COLORS)):
    ax2.text(i, rv + 0.5, f'{rv:.1f}%',
             ha='center', fontsize=11.5, color=col, fontweight='bold', va='bottom')
    ax2.text(i, rv + 5.0, f'{cv:.1f}% SKU',
             ha='center', fontsize=7.5, color=C_MUTED, va='bottom')

ax2.set_xticks(range(4))
ax2.set_xticklabels(bkt_labels, fontsize=9)
ax2.set_ylim(0, 58)
ax2.set_ylabel('% tổng doanh thu', fontsize=9, color=C_MID)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

neg_rev = rev_vals[0]
ax2.text(0.5, -0.15,
    f'33.0% SKU có margin âm, đóng góp {neg_rev:.1f}% doanh thu',
    transform=ax2.transAxes, ha='center', fontsize=8,
    color=C_MID, style='italic', linespacing=1.5)

ax2.set_title('② Doanh thu theo nhóm biên lợi nhuận',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)

ax3 = axes[2]

bp = ax3.boxplot(
    q_data,
    patch_artist=True,
    widths=0.55,
    medianprops=dict(color=C_TXT, lw=2.0),
    whiskerprops=dict(color='#cccccc', lw=1.0),
    capprops=dict(color='#cccccc', lw=1.0),
    flierprops=dict(marker='.', markersize=3, alpha=0.15,
                    markerfacecolor=C_MUTED, markeredgecolor='none'),
)

for patch, med in zip(bp['boxes'], q_medians):
    patch.set_facecolor(C_HI2 if med >= 0 else C_WARN)
    patch.set_alpha(0.35)
    patch.set_edgecolor('#bbbbbb')

for i, med in enumerate(q_medians):
    col = '#2d7d2d' if med >= 0 else C_WARN2
    ax3.text(i + 1, med + 3, f'{med:.1f}%',
             ha='center', fontsize=8.5, color=col, fontweight='bold', va='bottom')

ax3.axhline(0, color=C_TXT, lw=1.2, zorder=4)
ax3.axhspan(-110, 0, alpha=0.04, color=C_WARN, zorder=1)

ax3.set_xticklabels(q_labels, fontsize=9)
ax3.set_ylabel('Biên lợi nhuận gộp (%)', fontsize=9, color=C_MID)
ax3.set_ylim(-110, 65)
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax3.grid(axis='y', color=C_GRID, lw=0.8, zorder=0)
ax3.set_xlabel('Nhóm doanh thu (mỗi nhóm ~320 SKU)', fontsize=9, color=C_MID)

ax3.text(0.97, 0.97, f'r = {corr:.2f}',
         transform=ax3.transAxes, ha='right', va='top', fontsize=9, color=C_MID,
         bbox=dict(fc=C_BG, ec='#e0e0e0', boxstyle='round,pad=0.3'))

ax3.set_title('③ Phân phối biên lợi nhuận theo nhóm doanh thu',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)

fig.text(0.01, 0.958,
    '18% SKU = 80% doanh thu  |  39.5% doanh thu từ SKU có margin âm  |  2012–2022',
    fontsize=8.5, color=C_MUTED, va='top')

plt.tight_layout(rect=[0, 0.05, 1, 0.92])
plt.savefig('chart_portfolio.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print('Saved.')

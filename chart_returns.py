import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

rets  = pd.read_csv('returns.csv')
items = pd.read_csv('order_items.csv', low_memory=False)
prods = pd.read_csv('products.csv')

items_c = items.merge(prods[['product_id','category']], on='product_id', how='left')
rets_c  = rets.merge(prods[['product_id','category']], on='product_id', how='left')

cat_ord = items_c.groupby('category')['order_id'].nunique().rename('n_orders')
cat_ret = rets_c.groupby('category')['return_id'].count().rename('n_returns')
cat_df  = pd.concat([cat_ord, cat_ret], axis=1).reset_index()
cat_df['rate'] = cat_df['n_returns'] / cat_df['n_orders'] * 100
cat_df = cat_df.sort_values('rate', ascending=False)

reason_map = {
    'wrong_size':       'Sai size',
    'defective':        'Hàng lỗi',
    'not_as_described': 'Không đúng mô tả',
    'changed_mind':     'Đổi ý',
    'late_delivery':    'Giao trễ',
}
reason_ct  = rets['return_reason'].value_counts()
reason_pct = reason_ct / reason_ct.sum() * 100
reason_pct.index = [reason_map[r] for r in reason_pct.index]

avg_rate = len(rets) / items_c['order_id'].nunique() * 100

C_HI    = '#98f16d'
C_HI2   = '#5db83d'
C_WARN  = '#e07070'
C_WARN2 = '#c0392b'
C_LGRAY = '#e8e8e8'
C_TXT   = '#1a1a1a'
C_MID   = '#666666'
C_MUTED = '#aaaaaa'
C_BG    = '#ffffff'
C_GRID  = '#f0f0f0'

pie_colors = [C_HI, '#b0b0b0', '#c4c4c4', '#d8d8d8', '#ececec']

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                         gridspec_kw={'width_ratios': [1, 1.2]},
                         facecolor=C_BG)
fig.patch.set_facecolor(C_BG)

ax1 = axes[0]
ax1.set_facecolor(C_BG)
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines[['left', 'bottom']].set_color('#e0e0e0')
ax1.tick_params(colors=C_MID, labelsize=9.5)

cats  = cat_df['category'].tolist()
rates = cat_df['rate'].tolist()

bar_colors = [C_WARN if c == cats[0] else C_LGRAY for c in cats]
ax1.barh(cats, rates, color=bar_colors, height=0.75)
ax1.invert_yaxis()

for i, (r, c) in enumerate(zip(rates, cats)):
    col = C_WARN2 if i == 0 else C_MID
    fw  = 'bold'   if i == 0 else 'normal'
    ax1.text(r + 0.1, i, f'{r:.1f}%', va='center',
             fontsize=10.5, color=col, fontweight=fw)

ax1.axvline(avg_rate, color=C_MUTED, lw=1.2, ls='--')
ax1.text(avg_rate, -0.08, f'Avg {avg_rate:.1f}%',
         transform=ax1.get_xaxis_transform(),
         fontsize=8.5, color=C_MUTED, ha='center')


ax1.set_xlabel('Tỷ lệ hoàn hàng', fontsize=9, color=C_MID)
ax1.set_xlim(0, max(rates) * 1.25)
ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax1.grid(axis='x', color=C_GRID, lw=1)
ax1.set_title('Tỷ lệ hoàn hàng theo danh mục',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)

ax2 = axes[1]
ax2.set_facecolor(C_BG)
ax2.set_aspect('equal')

reasons = reason_pct.index.tolist()
pcts    = reason_pct.values.tolist()

explode = [0.05 if r == 'Sai size' else 0 for r in reasons]

wedges, _ = ax2.pie(
    pcts,
    colors=pie_colors,
    explode=explode,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(edgecolor=C_BG, linewidth=2.5),
    labels=None,
)

r_edge  = 1.04   # điểm bắt đầu (mép bánh)
r_label = 1.28   # tất cả label nằm cùng 1 vòng tròn
r_line  = r_label - r_edge  # độ dài connector = constant

for wedge, reason, pct in zip(wedges, reasons, pcts):
    ang = np.radians((wedge.theta1 + wedge.theta2) / 2)

    xe = r_edge * np.cos(ang)
    ye = r_edge * np.sin(ang)

    lx = r_label * np.cos(ang)
    ly = r_label * np.sin(ang)

    is_key = (reason == 'Sai size')
    col    = C_HI2 if is_key else C_MID
    fw     = 'bold' if is_key else 'normal'
    fs     = 10 if is_key else 9

    ax2.plot([xe, lx], [ye, ly],
             color=col,
             lw=1.2 if is_key else 0.7)

    ha = 'left' if lx > 0 else 'right'
    offset = 0.05 if lx > 0 else -0.05

    ax2.text(lx + offset, ly + 0.08,
             f'{pct:.1f}%',
             ha=ha, va='center',
             fontsize=fs, fontweight=fw, color=col)

    ax2.text(lx + offset, ly - 0.10,
             reason,
             ha=ha, va='center',
             fontsize=8.5, color=col)

ax2.set_xlim(-1.85, 1.85)
ax2.set_ylim(-1.65, 1.50)
ax2.axis('off')
ax2.set_title('Cơ cấu lý do hoàn hàng',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)

fig.text(0.01, 0.955,
    f'Tổng {len(rets):,} lượt hoàn  |  2012–2022',
    fontsize=8.5, color=C_MUTED, va='top')

plt.tight_layout(rect=[0, 0.0, 1, 0.93])
plt.savefig('chart_returns.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print('Saved.')

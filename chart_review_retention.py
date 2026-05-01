import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

orders  = pd.read_csv('orders.csv',  parse_dates=['order_date'])
reviews = pd.read_csv('reviews.csv', parse_dates=['review_date'])

reviews = reviews.merge(
    orders[['order_id', 'order_date']],
    on='order_id', how='left'
)
reviews['review_year'] = reviews['review_date'].dt.year

review_by_year = (
    reviews.groupby('review_year')
    .agg(n_reviews=('review_id', 'count'), avg_rating=('rating', 'mean'))
    .reset_index()
    .sort_values('review_year')
)
peak_n = review_by_year['n_reviews'].max()
peak_yr = int(review_by_year.loc[review_by_year['n_reviews'].idxmax(), 'review_year'])
review_by_year['pct_of_peak'] = review_by_year['n_reviews'] / peak_n * 100

rating_dist = (
    reviews.groupby(['review_year', 'rating'])['review_id']
    .count().unstack(fill_value=0)
    .reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
)
rating_pct = rating_dist.div(rating_dist.sum(axis=1), axis=0) * 100

rev_orders = reviews[['review_id', 'customer_id', 'order_date', 'rating']].dropna(
    subset=['customer_id', 'order_date']
)
all_orders = orders[['customer_id', 'order_date']].sort_values('order_date')
rev_next = rev_orders.merge(
    all_orders.rename(columns={'order_date': 'next_date'}), on='customer_id', how='left'
)
rev_next = rev_next[rev_next['next_date'] > rev_next['order_date']]
repeat_flag = (rev_next.groupby('review_id')['next_date'].count() > 0).rename('has_repeat')
repeat_by_rating = (
    rev_orders[['review_id', 'rating']]
    .merge(repeat_flag, on='review_id', how='left')
)
repeat_by_rating['has_repeat'] = repeat_by_rating['has_repeat'].fillna(False)
repeat_by_rating = (
    repeat_by_rating.groupby('rating')
    .agg(n_reviews=('review_id', 'count'), repeat_rate=('has_repeat', 'mean'))
    .reset_index().sort_values('rating')
)
repeat_by_rating['repeat_rate'] *= 100

C_BG      = '#ffffff'
C_TXT     = '#1a1a1a'
C_MID     = '#555555'
C_GRID    = '#e6e6e6'
C_MAIN    = '#98f16d'
C_ACCENT  = '#4a90d9'
C_DARK    = '#222222'
C_ALERT   = '#d9534f'
C_NEUTRAL = '#999999'
C_GRAY    = '#828282'
C_GRAY2   = '#c8c8c8'

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 10,
    'axes.grid': True, 'grid.color': C_GRID, 'grid.linestyle': '-', 'grid.alpha': 0.35,
})

fig = plt.figure(figsize=(16, 10), facecolor=C_BG)
gs  = gridspec.GridSpec(
    2, 2, figure=fig,
    left=0.07, right=0.97, top=0.91, bottom=0.09,
    hspace=0.34, wspace=0.30,
)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

years   = review_by_year['review_year'].values
n_rev_k = review_by_year['n_reviews'].values / 1000
ratings = review_by_year['avg_rating'].values

bar_clr = [C_ACCENT if y < 2019 else C_GRAY2 for y in years]
ax1.bar(years, n_rev_k, color=bar_clr, edgecolor='white', lw=0.6, width=0.78, zorder=2)

ax1b = ax1.twinx()
ax1b.plot(years, ratings, marker='o', color=C_GRAY, lw=2.0, ls='--', zorder=3)
ax1b.set_ylabel('Avg rating', color=C_GRAY, fontsize=9)
ax1b.set_ylim(3.87, 4.02)
ax1b.set_yticks([3.90, 3.95, 4.00])
ax1b.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax1b.tick_params(axis='y', colors=C_GRAY)

n_2019 = review_by_year.loc[review_by_year['review_year'] == 2019, 'n_reviews'].iat[0] / 1000
drop   = (n_rev_k.max() - n_2019) / n_rev_k.max() * 100
ax1.text(2019.5, n_2019 + 0.4, f'−{drop:.0f}% từ đỉnh',
         fontsize=8.5, color=C_ALERT, fontweight='bold')

p1 = mpatches.Patch(color=C_ACCENT, label='2013–2018 (tương tác cao)')
p2 = mpatches.Patch(color=C_GRAY2,  label='2019–2022')
ax1.legend(handles=[p1, p2], loc='upper left', frameon=False, fontsize=8.5)

ax1.set_xlabel('Năm', color=C_MID)
ax1.set_ylabel('Số lượt đánh giá (nghìn)', color=C_DARK)
ax1.set_title('Lượt đánh giá sụp đổ — chất lượng không đổi',
              fontsize=13, fontweight='bold', color=C_TXT)
ax1.set_facecolor(C_BG)
ax1.set_ylim(0, n_rev_k.max() * 1.28)
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

r_order  = [5, 4, 3, 2, 1]
r_labels = [f'{r}★' for r in r_order]
rates    = [repeat_by_rating.loc[repeat_by_rating['rating'] == r, 'repeat_rate'].iat[0]
            for r in r_order]
ns       = [repeat_by_rating.loc[repeat_by_rating['rating'] == r, 'n_reviews'].iat[0]
            for r in r_order]
h_clrs   = [C_MAIN] + [C_GRAY2] * 4

ax2.barh(r_labels, rates, color=h_clrs, edgecolor=C_DARK, lw=0.7, height=0.55, zorder=2)
ax2.yaxis.grid(False)
ax2.xaxis.grid(True)

for i, (rate, n) in enumerate(zip(rates, ns)):
    ax2.text(rate + 0.08, i, f'{rate:.1f}%   (n = {n/1000:.0f}K)',
             va='center', fontsize=9.5, color=C_DARK)

ax2.set_xlim(83.5, 92.5)
ax2.set_facecolor(C_BG)
ax2.set_xlabel('Tỷ lệ mua lại (%)', color=C_MID)
ax2.set_title('Rating ≠ động lực mua lại',
              fontsize=13, fontweight='bold', color=C_TXT)

ax2.text(
    0.97, 0.15,
    f'Khoảng cách 1★ → 5★: chỉ ~0.1pp\n→ Cải thiện rating KHÔNG giữ được khách',
    transform=ax2.transAxes, fontsize=9.5, color=C_MID, ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.4', fc='#f9f9f9', ec='#ddd', alpha=0.9),
)

yrs3     = review_by_year['review_year'].values
pct_peak = review_by_year['pct_of_peak'].values

mask_pre  = yrs3 <= 2018
mask_post = yrs3 >= 2018

ax3.fill_between(yrs3[mask_pre],  pct_peak[mask_pre],  alpha=0.28, color=C_ACCENT, zorder=1)
ax3.fill_between(yrs3[mask_post], pct_peak[mask_post], alpha=0.38, color=C_GRAY2,  zorder=1)
ax3.plot(yrs3, pct_peak, marker='o', color=C_DARK, lw=2.2, zorder=4)

ax3.axvline(2018.5, color=C_ALERT, ls='--', lw=1.7, zorder=5)
ax3.set_facecolor(C_BG)
ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax3.set_ylim(0, pct_peak.max() * 1.20)
ax3.set_xlabel('Năm', color=C_MID)
ax3.set_ylabel(f'% so với đỉnh {peak_yr} (= 100%)', color=C_MID)
ax3.set_title('Silent churn: nhóm "im lặng" ngày càng đông',
              fontsize=13, fontweight='bold', color=C_TXT)

for yr, val in zip(yrs3, pct_peak):
    if yr in {int(yrs3.min()), peak_yr, 2019, int(yrs3.max())}:
        lbl = f'{val:.0f}%'
        if yr == peak_yr:
            lbl = f'Đỉnh\n{val:.0f}%'
        ax3.text(yr, val + 3.5, lbl, ha='center', fontsize=8.5, color=C_DARK, fontweight='bold')

val_2019_p = review_by_year.loc[review_by_year['review_year'] == 2019, 'pct_of_peak'].iat[0]
ax3.annotate(
    f'Xuống còn {val_2019_p:.0f}% đỉnh\n(giảm {100 - val_2019_p:.0f}% lượt đánh giá)',
    xy=(2019, val_2019_p),
    xytext=(2020.2, val_2019_p + 18),
    arrowprops=dict(arrowstyle='->', color=C_ALERT, lw=1.1),
    fontsize=8.5, color=C_ALERT,
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', alpha=0.95),
)

ax3.text(
    0.02, 0.93,
    '86.7% repeat rate chỉ tính trên người có review.\nKhách "im lặng" rời bỏ không để lại dấu vết.',
    transform=ax3.transAxes, fontsize=9, color=C_MID, ha='left', va='top',
    bbox=dict(boxstyle='round,pad=0.3', fc='#f9f9f9', ec='#ddd', alpha=0.9),
)

p_pre  = mpatches.Patch(color=C_ACCENT, alpha=0.5, label='Trước 2019')
p_post = mpatches.Patch(color=C_GRAY2,  alpha=0.7, label='Từ 2019')
ax3.legend(handles=[p_pre, p_post], loc='upper right', frameon=False, fontsize=8.5)

yrs4    = rating_pct.index.tolist()
bot     = np.zeros(len(yrs4))
stk_clr = {1: '#8b8b8b', 2: '#aaaaaa', 3: '#c0c0c0', 4: '#d7d7d7', 5: C_MAIN}
for rat in [1, 2, 3, 4, 5]:
    ax4.bar(yrs4, rating_pct[rat], bottom=bot,
            color=stk_clr[rat], edgecolor='white', width=0.72, label=f'{rat}★')
    bot += rating_pct[rat].values

ax4.set_title('Phân bố rating ổn định suốt 11 năm',
              fontsize=13, fontweight='bold', color=C_TXT)
ax4.set_xlabel('Năm', color=C_MID)
ax4.set_ylabel('Tỷ trọng (%)', color=C_MID)
ax4.set_ylim(0, 114)
ax4.set_facecolor(C_BG)
ax4.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax4.legend(ncols=5, frameon=False, fontsize=8,
           loc='upper center', bbox_to_anchor=(0.5, -0.16))

for yr in [2012, 2015, 2019, 2022]:
    if yr in rating_pct.index:
        p5 = rating_pct.loc[yr, 5]
        ax4.text(yr, 103, f'{p5:.0f}%\n5★',
                 ha='center', va='bottom', fontsize=7.5, color=C_DARK)


fig.text(
    0.5, 0.978,
    'Bẫy chất lượng: dịch vụ tốt nhưng vẫn mất khách — điểm gãy tương tác & silent churn',
    ha='center', fontsize=15, fontweight='bold', color=C_DARK,
)
fig.text(
    0.5, 0.948,
    'Rating ổn định 3.91–3.98★  |  Tỷ lệ mua lại 86.7% không đổi trên mọi nhóm  |  '
    'Lượt đánh giá giảm >50% từ đỉnh — nhóm "im lặng" ngày càng tăng',
    ha='center', fontsize=10, color=C_MID,
)

plt.savefig('chart_review_retention.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print('Saved chart_review_retention.png')

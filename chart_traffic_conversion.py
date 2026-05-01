import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

web    = pd.read_csv('web_traffic.csv', parse_dates=['date'])
orders = pd.read_csv('orders.csv', parse_dates=['order_date'])
items  = pd.read_csv('order_items.csv', low_memory=False)

web['year'] = web['date'].dt.year
yr_web = web.groupby('year').agg(
    sessions   = ('sessions', 'sum'),
    bounce     = ('bounce_rate', 'mean'),
    duration   = ('avg_session_duration_sec', 'mean'),
    page_views = ('page_views', 'sum'),
).reset_index()

orders['year'] = orders['order_date'].dt.year
ord_yr = orders.groupby('year')['order_id'].count().rename('n_orders')
yr = yr_web.merge(ord_yr, on='year', how='left').fillna(0)
yr['conv']   = yr['n_orders'] / yr['sessions'] * 100
yr['pgs_ps'] = yr['page_views'] / yr['sessions']

base = yr[yr['year'] == 2013].iloc[0]
yr['idx_sess'] = yr['sessions'] / base['sessions'] * 100
yr['idx_ord']  = yr['n_orders']  / base['n_orders']  * 100
yr['idx_conv'] = yr['conv']      / base['conv']       * 100

items['line_rev'] = items['quantity'] * items['unit_price'] - items['discount_amount']
revenue  = items.groupby('order_id')['line_rev'].sum().rename('revenue')
ord2     = orders.merge(revenue, on='order_id')
avg_aov  = ord2['revenue'].mean()

dev_pct = orders['device_type'].value_counts(normalize=True).sort_values(ascending=False) * 100
src_pct = orders['order_source'].value_counts(normalize=True).sort_values(ascending=False) * 100

C_HI2   = '#5db83d'
C_WARN  = '#e07070'
C_WARN2 = '#c0392b'
C_LGRAY = '#e8e8e8'
C_TXT   = '#1a1a1a'
C_MID   = '#666666'
C_MUTED = '#aaaaaa'
C_BG    = '#ffffff'
C_GRID  = '#f0f0f0'

dev_vi = {'mobile': 'Mobile', 'desktop': 'Desktop', 'tablet': 'Tablet'}
src_vi = {
    'organic_search': 'Organic search',
    'paid_search':    'Paid search',
    'social_media':   'Social media',
    'email_campaign': 'Email campaign',
    'referral':       'Referral',
    'direct':         'Direct',
}

fig = plt.figure(figsize=(14, 9.5), facecolor=C_BG)
fig.patch.set_facecolor(C_BG)

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1.25, 1],
    width_ratios=[1, 1],
    hspace=0.32, wspace=0.25,
)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor(C_BG)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#e0e0e0')
    ax.tick_params(colors=C_MID, labelsize=9)

years = yr['year'].tolist()
i_s   = yr['idx_sess'].tolist()
i_o   = yr['idx_ord'].tolist()
i_c   = yr['idx_conv'].tolist()

ax1.axhline(100, color=C_MUTED, lw=0.9, ls='--', zorder=1)

l1, = ax1.plot(years, i_s, color=C_HI2, lw=2.5, marker='o', ms=4.5, zorder=4)
l2, = ax1.plot(years, i_o, color=C_MID,  lw=2.0, marker='s', ms=4.0, ls='--', zorder=3)
l3, = ax1.plot(years, i_c, color=C_WARN, lw=2.5, marker='o', ms=4.5, zorder=5)

last = yr.iloc[-1]
ax1.text(2022.25, last['idx_sess'] + 3,
         f"Sessions  +{last['idx_sess']-100:.0f}%",
         fontsize=9, color=C_HI2, fontweight='bold', va='bottom')
ax1.text(2022.25, last['idx_ord'] + 3,
         f"Đơn hàng  −{100-last['idx_ord']:.0f}%",
         fontsize=9, color=C_MID, fontweight='bold', va='bottom')
ax1.text(2022.25, last['idx_conv'] - 3,
         f"Chuyển đổi  −{100-last['idx_conv']:.0f}%",
         fontsize=9, color=C_WARN2, fontweight='bold', va='top')

leg = ax1.legend(
    [l1, l2, l3],
    ['Sessions', 'Số đơn hàng', 'Tỷ lệ chuyển đổi (đơn/phiên)'],
    loc='upper left', fontsize=9, framealpha=0,
)
for text, col in zip(leg.get_texts(), [C_HI2, C_MID, C_WARN]):
    text.set_color(col)

mean_pgs    = yr['pgs_ps'].mean()
mean_dur    = yr['duration'].mean()
mean_bounce = yr['bounce'].mean() * 100
eng = (f'Chất lượng phiên ổn định (TB 2013–2022)\n'
       f'{mean_pgs:.1f} trang/phiên  ·  {mean_dur:.0f}s  ·  '
       f'bounce {mean_bounce:.2f}%')
ax1.text(0.30, 0.06, eng, transform=ax1.transAxes,
         ha='left', va='bottom', fontsize=7.5, color=C_MID, linespacing=1.65,
         bbox=dict(fc=C_BG, ec='#e0e0e0', boxstyle='round,pad=0.4'))

ax1.set_title(
    'Tăng lưu lượng không bù được suy giảm chuyển đổi — Đơn hàng giảm dài hạn (2013–2022)',
    fontsize=11, fontweight='bold', color=C_TXT, pad=10)

dev_order = dev_pct.index.tolist()
dev_vals  = dev_pct.tolist()

ax2.barh([dev_vi.get(d, d) for d in dev_order], dev_vals,
         color=C_LGRAY, height=0.48)
ax2.invert_yaxis()

for i, (d, p) in enumerate(zip(dev_order, dev_vals)):
    ax2.text(p + 0.8, i, f'{p:.1f}%', va='center', fontsize=10.5, color=C_MID)

ax2.set_xlim(0, 60)
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax2.grid(axis='x', color=C_GRID, lw=0.8)
ax2.set_xlabel('% đơn hàng', fontsize=9, color=C_MID)
ax2.set_title('Cơ cấu thiết bị đặt hàng',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)
ax2.text(0.5, -0.2,
    f'AOV đồng đều: ≈ {avg_aov:,.0f} đ/đơn — không thiết bị nào nổi trội',
    transform=ax2.transAxes, ha='center', fontsize=8.5,
    color=C_MID, style='italic')

src_order = src_pct.index.tolist()
src_vals  = src_pct.tolist()

ax3.barh([src_vi.get(s, s) for s in src_order], src_vals,
         color=C_LGRAY, height=0.52)
ax3.invert_yaxis()

for i, (s, p) in enumerate(zip(src_order, src_vals)):
    ax3.text(p + 0.5, i, f'{p:.1f}%', va='center', fontsize=10.5, color=C_MID)

ax3.set_xlim(0, 60)
ax3.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax3.grid(axis='x', color=C_GRID, lw=0.8)
ax3.set_xlabel('% đơn hàng', fontsize=9, color=C_MID)
ax3.set_title('Cơ cấu nguồn đặt hàng',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)
ax3.text(0.5, -0.2,
    f'AOV đồng đều: ≈ {avg_aov:,.0f} đ/đơn — không kênh nào nổi trội',
    transform=ax3.transAxes, ha='center', fontsize=8.5,
    color=C_MID, style='italic')

fig.text(0.01, 0.960,
    'Proxy chuyển đổi = đơn / phiên  |  Thiết bị & kênh từ orders.csv  |  2013–2022',
    fontsize=8.5, color=C_MUTED, va='top')

plt.savefig('chart_traffic_conversion.png', dpi=150,
            bbox_inches='tight', facecolor=C_BG)
print('Saved.')

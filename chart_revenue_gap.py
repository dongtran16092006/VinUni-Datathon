import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

orders      = pd.read_csv('orders.csv',      parse_dates=['order_date'])
returns     = pd.read_csv('returns.csv',     parse_dates=['return_date'])
order_items = pd.read_csv('order_items.csv', low_memory=False)

order_items['revenue'] = (order_items['quantity'] * order_items['unit_price']
                          - order_items['discount_amount'])
order_rev = order_items.groupby('order_id')['revenue'].sum().reset_index()
orders = orders.merge(order_rev, on='order_id', how='left')
orders['year']  = orders['order_date'].dt.year
orders['month'] = orders['order_date'].dt.month
orders['dow']   = orders['order_date'].dt.dayofweek   # 0=T2 … 6=CN
orders['date']  = orders['order_date'].dt.date

returns = returns.merge(orders[['order_id', 'year']], on='order_id', how='left')

yr = orders.groupby('year').agg(gross_rev=('revenue', 'sum')).reset_index()
rf = returns.groupby('year').agg(refund=('refund_amount', 'sum')).reset_index()
yr = yr.merge(rf, on='year', how='left').fillna(0).sort_values('year')
yr['net_rev'] = yr['gross_rev'] - yr['refund']
yr['gap_pct'] = yr['refund'] / yr['gross_rev'] * 100

sales    = pd.read_csv('sales.csv', parse_dates=['Date'])
sales['year'] = sales['Date'].dt.year
sales_yr = sales.groupby('year').agg(rev_s=('Revenue', 'sum'), cogs_s=('COGS', 'sum')).reset_index()
sales_yr['margin_pct'] = (sales_yr['rev_s'] - sales_yr['cogs_s']) / sales_yr['rev_s'] * 100
yr = yr.merge(sales_yr[['year', 'margin_pct']], on='year', how='left')

years    = yr['year'].values
gross_B  = yr['gross_rev'].values / 1e9
net_B    = yr['net_rev'].values   / 1e9
refund_M   = yr['refund'].values    / 1e6
gap_pct    = yr['gap_pct'].values
margin_pct = yr['margin_pct'].values
idx_peak = int(gross_B.argmax())
yr_peak  = int(years[idx_peak])
mean_gap = gap_pct.mean()

monthly = orders.groupby(['year', 'month'])['revenue'].sum().reset_index()
ann_avg = monthly.groupby('year')['revenue'].sum() / 12
monthly['idx'] = monthly.apply(lambda r: r['revenue'] / ann_avg[r['year']], axis=1)
mo_idx = monthly.groupby('month')['idx'].mean().values * 100   # shape (12,)
mo_std = monthly.groupby('month')['idx'].std().values  * 100

daily = orders.groupby(['date', 'dow', 'year'])['revenue'].sum().reset_index()
ann_day = orders.groupby('year')['revenue'].sum() / 365
daily['idx'] = daily.apply(lambda r: r['revenue'] / ann_day[r['year']], axis=1) * 100
dow_idx = daily.groupby('dow')['idx'].mean().values   # shape (7,)

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
    left=0.06, right=0.97, top=0.91, bottom=0.07,
    hspace=0.26, wspace=0.32,
    width_ratios=[3, 2], height_ratios=[1, 1],
)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

ax1.fill_between(years, net_B, color=C_ACCENT, alpha=0.10, zorder=1)
ax1.fill_between(years, gross_B, net_B,
                 color=C_ALERT, alpha=0.30, label='Phần hoàn trả', zorder=2)
ax1.plot(years, gross_B, marker='o', color=C_MAIN,   lw=2.5, label='Doanh thu gộp',   zorder=4)
ax1.plot(years, net_B,   marker='o', color=C_ACCENT, lw=2.5, label='Doanh thu thuần', zorder=3)

ax1.set_facecolor(C_BG)
ax1.set_xlabel('Năm', color=C_MID)
ax1.set_ylabel('Doanh thu (VND)', color=C_MID)
ax1.set_title('Gap bám sát chu kỳ doanh thu — tỷ lệ hoàn trả ổn định',
              fontsize=13, fontweight='bold', color=C_TXT)
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}B'))
ax1.set_ylim(0, gross_B.max() * 1.24)
ax1.set_xlim(years.min() - 0.4, years.max() + 0.9)

ax1.annotate(
    f'Đỉnh {yr_peak}\nGộp: {gross_B[idx_peak]:.2f}B\nThuần: {net_B[idx_peak]:.2f}B',
    xy=(yr_peak, gross_B[idx_peak]),
    xytext=(yr_peak - 1.8, gross_B[idx_peak] * 1.09),
    arrowprops=dict(arrowstyle='->', color=C_DARK, lw=1.0),
    fontsize=9, color=C_DARK,
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', alpha=0.95),
)
gap_mid = (gross_B[idx_peak] + net_B[idx_peak]) / 2
ax1.text(yr_peak + 0.7, gap_mid, f'{refund_M[idx_peak]:.0f}M hoàn trả',
         fontsize=8.5, color=C_ALERT, fontweight='bold', va='center')
for yr_x, g, n in [(years[0], gross_B[0], net_B[0]),
                   (years[-1], gross_B[-1], net_B[-1])]:
    ax1.text(yr_x, g + 0.05, f'{g:.2f}B', ha='center', fontsize=8, color=C_DARK)
    ax1.text(yr_x, n - 0.09, f'{n:.2f}B', ha='center', fontsize=8, color=C_ACCENT)
ax1.legend(loc='upper left', frameon=False, fontsize=9.5)

highlight = [C_ALERT if y == yr_peak else '#e8a09e' for y in years]
bars2 = ax2.bar(years, refund_M, color=highlight, edgecolor='white', lw=0.6,
                width=0.75, zorder=2)
ax2.set_facecolor(C_BG)
ax2.set_xlabel('Năm', color=C_MID)
ax2.set_ylabel('Giá trị hoàn trả (triệu VND)', color=C_ALERT)
ax2.tick_params(axis='y', colors=C_ALERT)
ax2.set_title('Giá trị & tỷ lệ hoàn trả theo năm',
              fontsize=13, fontweight='bold', color=C_TXT)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.set_ylim(0, refund_M.max() * 1.38)

label_yrs = {int(years[0]), yr_peak, 2019, int(years[-1])}
for bar, y_val, val in zip(bars2, years, refund_M):
    if int(y_val) in label_yrs:
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.7,
                 f'{val:.0f}M', ha='center', va='bottom', fontsize=8.5, color=C_DARK)

ax2b = ax2.twinx()
ax2b.plot(years, gap_pct, marker='o', color=C_DARK, lw=2.0, ls='--', zorder=3)
ax2b.set_ylabel('Tỷ lệ hoàn trả (%)', color=C_GRAY)
ax2b.set_ylim(2.5, 4.2)
ax2b.set_yticks([2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
ax2b.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
ax2b.tick_params(axis='y', colors=C_GRAY)
ax2b.yaxis.grid(False)
ax2b.axhline(mean_gap, color=C_NEUTRAL, ls=':', lw=1.5, zorder=1)
ax2b.text(years[-1] + 0.18, mean_gap + 0.04,
          f'TB {mean_gap:.2f}%', fontsize=8.5, color=C_NEUTRAL, va='bottom')
ax2.text(0.97, 0.97,
         f'Dao động {gap_pct.min():.2f}% – {gap_pct.max():.2f}% trong 11 năm\n'
         '→ Hành vi hoàn trả ổn định, không bất thường',
         transform=ax2.transAxes, fontsize=8.5, color=C_MID, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.35', fc='#f9f9f9', ec='#ddd', alpha=0.92))

months   = np.arange(1, 13)
mo_lbl   = ['Th.1','Th.2','Th.3','Th.4','Th.5','Th.6',
            'Th.7','Th.8','Th.9','Th.10','Th.11','Th.12']

mo_clr = [C_MAIN if v >= 100 else C_GRAY2 for v in mo_idx]
mo_clr[mo_idx.argmax()] = C_MAIN
mo_clr[mo_idx.argmin()] = C_ALERT

bars3 = ax3.bar(months, mo_idx, color=mo_clr, edgecolor='white', lw=0.5,
                width=0.75, zorder=2)

ax3.fill_between(months,
                 np.clip(mo_idx - mo_std, 0, None),
                 mo_idx + mo_std,
                 color=C_NEUTRAL, alpha=0.18, zorder=1, label='±1 std (biến động năm)')

ax3.axhline(100, color=C_DARK, lw=1.4, ls='--', zorder=3)
ax3.text(12.55, 101, 'Trung\nbình', fontsize=8, color=C_DARK, va='bottom')

ax3.set_facecolor(C_BG)
ax3.set_xticks(months)
ax3.set_xticklabels(mo_lbl, fontsize=9)
ax3.set_ylabel('Chỉ số doanh thu (trung bình năm = 100)', color=C_MID)
ax3.set_title('Tính mùa vụ theo tháng — đỉnh quý 2, đáy quý 4',
              fontsize=13, fontweight='bold', color=C_TXT)
ax3.set_xlim(0.3, 13.2)
ax3.set_ylim(0, mo_idx.max() + mo_std.max() + 22)
ax3.set_facecolor(C_BG)

for m, v, s in zip(months, mo_idx, mo_std):
    if m in {int(months[mo_idx.argmax()]), int(months[mo_idx.argmin()]), 1, 3, 9}:
        ax3.text(m, v + s + 3, f'{v:.0f}', ha='center', fontsize=8.5,
                 color=C_DARK, fontweight='bold')

m_peak = int(months[mo_idx.argmax()])
m_low  = int(months[mo_idx.argmin()])
ax3.annotate(f'Cao nhất\nTh.{m_peak}: {mo_idx[m_peak-1]:.0f}',
             xy=(m_peak, mo_idx[m_peak-1]),
             xytext=(m_peak - 2.2, mo_idx[m_peak-1] + mo_std[m_peak-1] + 10),
             arrowprops=dict(arrowstyle='->', color=C_MAIN, lw=1.0),
             fontsize=9, color=C_MAIN,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', alpha=0.95))
ax3.annotate(f'Thấp nhất\nTh.{m_low}: {mo_idx[m_low-1]:.0f}',
             xy=(m_low, mo_idx[m_low-1]),
             xytext=(m_low - 0.5, mo_idx[m_low-1] + mo_std[m_low-1] + 42),
             arrowprops=dict(arrowstyle='->', color=C_ALERT, lw=1.0),
             fontsize=9, color=C_ALERT, ha='right',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', alpha=0.95))

ax3.legend(loc='upper right', frameon=False, fontsize=8.5)

dow_lbl = ['T.Hai', 'T.Ba', 'T.Tư', 'T.Năm', 'T.Sáu', 'T.Bảy', 'CN']
dow_clr = [C_GRAY2] * 7
dow_clr[int(dow_idx.argmax())] = C_MAIN
dow_clr[int(dow_idx.argmin())] = C_ALERT

ax4.bar(range(7), dow_idx, color=dow_clr, edgecolor='white', lw=0.5,
        width=0.65, zorder=2)
ax4.axhline(100, color=C_DARK, lw=1.4, ls='--', zorder=3)
ax4.text(6.55, 100.6, 'TB', fontsize=8, color=C_DARK, va='bottom')

ax4.set_facecolor(C_BG)
ax4.set_xticks(range(7))
ax4.set_xticklabels(dow_lbl, fontsize=9.5)
ax4.set_ylabel('Chỉ số doanh thu (trung bình ngày = 100)', color=C_MID)
ax4.set_title('Tính mùa vụ theo thứ — giữa tuần cao hơn cuối tuần',
              fontsize=13, fontweight='bold', color=C_TXT)
ax4.set_ylim(80, dow_idx.max() + 18)
ax4.set_xlim(-0.6, 6.6)

for i, v in enumerate(dow_idx):
    ax4.text(i, v + 0.8, f'{v:.1f}', ha='center', fontsize=9,
             color=C_DARK, fontweight='bold')

ax4.annotate('',
             xy=(int(dow_idx.argmax()), dow_idx.max() + 10),
             xytext=(int(dow_idx.argmin()), dow_idx.min() + 10),
             arrowprops=dict(arrowstyle='<->', color=C_NEUTRAL, lw=1.2))
spread = dow_idx.max() - dow_idx.min()
ax4.text(3, dow_idx.max() + 11.5,
         f'Chênh lệch T.Tư – T.Bảy: {spread:.1f} điểm',
         ha='center', fontsize=8.5, color=C_MID)

fig.text(0.5, 0.94,
         'Hoàn trả ổn định ~3.2%  |  Đỉnh mùa vụ tháng 4–6 (quý 2)  |  '
         'Giữa tuần giao dịch nhiều hơn cuối tuần',
         ha='center', fontsize=10, color=C_MID)

plt.savefig('chart_revenue_gap.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print('Saved chart_revenue_gap.png')

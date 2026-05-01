import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

orders = pd.read_csv('orders.csv', parse_dates=['order_date'])
ship   = pd.read_csv('shipments.csv', parse_dates=['ship_date','delivery_date'])
rets   = pd.read_csv('returns.csv', parse_dates=['return_date'])

ship_full = ship.merge(orders[['order_id','order_date']], on='order_id', how='left')
ship_full['fulfil_days']  = (ship_full['delivery_date'] - ship_full['order_date']).dt.days
ship_full['time_to_ship'] = (ship_full['ship_date']     - ship_full['order_date']).dt.days

late_ret = rets[rets['return_reason']=='late_delivery'][['order_id']].drop_duplicates()
late_ret['is_late'] = 1
ship_full = ship_full.merge(late_ret, on='order_id', how='left')
ship_full['is_late'] = ship_full['is_late'].fillna(0).astype(int)

total  = len(orders)
cancel = (orders['order_status']=='cancelled').sum()

fd_ct  = ship_full['fulfil_days'].value_counts().sort_index()
tts_ct = ship_full['time_to_ship'].value_counts().sort_index()
by_day = ship_full.groupby('fulfil_days').agg(
    n      = ('order_id','count'),
    n_late = ('is_late','sum')
).reset_index()
by_day['late_pct'] = by_day['n_late'] / by_day['n'] * 100
mean_late = by_day['late_pct'].mean()

C_HI    = '#98f16d'
C_HI2   = '#5db83d'
C_GRAY  = '#d0d0d0'
C_TXT   = '#1a1a1a'
C_MID   = '#666666'
C_MUTED = '#aaaaaa'
C_BG    = '#ffffff'
C_LINE  = '#e5e5e5'

fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor=C_BG)
fig.patch.set_facecolor(C_BG)

for ax in axes:
    ax.set_facecolor(C_BG)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color(C_LINE)
    ax.tick_params(colors=C_MID, labelsize=9)

ax1 = axes[0]
days  = fd_ct.index.tolist()
cnts  = fd_ct.values
pcts  = cnts / cnts.sum() * 100

bar_colors = [C_HI if 5 <= d <= 7 else C_GRAY for d in days]
bars = ax1.bar(days, pcts, color=bar_colors, width=0.7,
               edgecolor=C_BG, linewidth=1.5)

peak_d = days[pcts.argmax()]
peak_p = pcts.max()
ax1.annotate(f'{peak_p:.1f}%',
    xy=(peak_d, peak_p),
    xytext=(peak_d + 1.2, peak_p - 0.5),
    ha='left', fontsize=8.5, fontweight='bold', color=C_HI2)

mean_fd = ship_full['fulfil_days'].mean()
ax1.axvline(mean_fd, color=C_MID, lw=1.2, ls='--', zorder=5)
ax1.text(mean_fd + 0.15, pcts.max() * 0.55,
         f'Mean\n{mean_fd:.1f}d', fontsize=8, color=C_MID, va='center')

ax1.set_xticks(days)
ax1.set_xlabel('Số ngày giao hàng (order → delivery)', fontsize=9, color=C_MID)
ax1.set_ylabel('% đơn', fontsize=9, color=C_MID)
ax1.set_title('Phân phối thời gian giao hàng', fontsize=10.5, fontweight='bold',
              color=C_TXT, pad=10)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

stats_txt = (f'Min  2d  ·  Max  10d\n'
             f'Median  6d  ·  Mean  6.0d\n'
             f'Q1  4d  —  Q3  7d')
ax1.text(0.97, 0.97, stats_txt,
         transform=ax1.transAxes, ha='right', va='top',
         fontsize=7.5, color=C_MID, linespacing=1.7,
         bbox=dict(fc=C_BG, ec=C_LINE, boxstyle='round,pad=0.4'))

ax2 = axes[1]
tts_days = tts_ct.index.tolist()
tts_pcts = tts_ct.values / tts_ct.values.sum() * 100

tts_colors = [C_HI if d == 0 else C_GRAY for d in tts_days]
ax2.bar(tts_days, tts_pcts, color=tts_colors, width=0.55,
        edgecolor=C_BG, linewidth=1.5)

for d, p in zip(tts_days, tts_pcts):
    ax2.text(d, p + 0.4, f'{p:.1f}%', ha='center', fontsize=8.5,
             color=C_HI2 if d == 0 else C_MUTED, fontweight='bold' if d == 0 else 'normal')

ax2.set_xticks(tts_days)
ax2.set_xticklabels([f'{d}d' for d in tts_days])
ax2.set_xlabel('Số ngày từ đặt hàng đến xuất kho', fontsize=9, color=C_MID)
ax2.set_ylabel('% đơn', fontsize=9, color=C_MID)
ax2.set_title('Thời gian chuẩn bị hàng (time-to-ship)', fontsize=10.5,
              fontweight='bold', color=C_TXT, pad=10)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax2.set_ylim(0, tts_pcts.max() * 1.2)

ax2.text(0.5, 0.55,
         '~25% đơn\nxuất kho ngay\ntrong ngày đặt',
         transform=ax2.transAxes, ha='center', va='center',
         fontsize=8.5, color=C_HI2, linespacing=1.6,
         bbox=dict(fc='#f6ffe8', ec=C_HI, boxstyle='round,pad=0.5', lw=1))

ax3 = axes[2]
ax3.bar(by_day['fulfil_days'], by_day['late_pct'],
        color=C_GRAY, width=0.7, edgecolor=C_BG, linewidth=1.5, label='Tỷ lệ hoàn "giao trễ"')

ax3.axhline(mean_late, color=C_HI2, lw=2, ls='-', zorder=5)
ax3.text(5.5, mean_late + 0.03,
         f'~{mean_late:.2f}%  (flat)', fontsize=8, color=C_HI2,
         fontweight='bold', va='bottom')

ax3.set_xticks(by_day['fulfil_days'])
ax3.set_xlabel('Số ngày giao hàng thực tế', fontsize=9, color=C_MID)
ax3.set_ylabel('% đơn hoàn lý do "giao trễ"', fontsize=9, color=C_MID)
ax3.set_title('Hoàn "giao trễ" vs thời gian giao thực tế', fontsize=10.5,
              fontweight='bold', color=C_TXT, pad=10)
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f%%'))
ax3.set_ylim(0, by_day['late_pct'].max() * 2.2)

insight = ('⚠ Tỷ lệ hoàn "giao trễ"\nxấp xỉ nhau ở mọi mức\nthời gian giao')
ax3.text(0.50, 0.97, insight,
         transform=ax3.transAxes, ha='center', va='top',
         fontsize=8, color=C_MID, linespacing=1.2,
         bbox=dict(fc='#fff8e8', ec='#f0d080', boxstyle='round,pad=0.4', lw=1))

cancel_pct = cancel / total * 100
fig.text(0.5, -0.04,
         f'Cancellation rate: {cancel_pct:.1f}%  ({cancel:,}/{total:,} đơn)  |  '
         f'Tổng shipments có dữ liệu: {len(ship_full):,}',
         ha='center', fontsize=9, color=C_MUTED)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('chart_fulfilment.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print('Saved: chart_fulfilment.png')

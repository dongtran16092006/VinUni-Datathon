import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

inv = pd.read_csv('inventory.csv', parse_dates=['snapshot_date'])

inv['group'] = 'none'
inv.loc[(inv['stockout_flag']==1) & (inv['overstock_flag']==0), 'group'] = 'stockout_only'
inv.loc[(inv['stockout_flag']==0) & (inv['overstock_flag']==1), 'group'] = 'overstock_only'
inv.loc[(inv['stockout_flag']==1) & (inv['overstock_flag']==1), 'group'] = 'both'

dos_yr = inv.groupby('year')['days_of_supply'].mean().reset_index()
years  = dos_yr['year'].tolist()
dos    = dos_yr['days_of_supply'].tolist()

sd = inv[inv['stockout_flag']==1]['stockout_days']
sd_ct = sd.value_counts().sort_index()
sd_show = sd_ct[sd_ct.index <= 10]
pct_le2 = (sd <= 2).mean() * 100

group_order  = ['stockout_only', 'none', 'overstock_only', 'both']
group_labels = ['Chỉ\nStockout', 'Không có\ncờ nào', 'Chỉ\nOverstock', 'Cả hai\n(both)']
group_med    = inv.groupby('group')['days_of_supply'].median()
medians      = [group_med[g] for g in group_order]

C_HI    = '#98f16d'
C_HI2   = '#5db83d'
C_HI3   = '#3a7a22'
C_RED   = '#e07070'
C_GRAY  = '#c8c8c8'
C_LGRAY = '#e8e8e8'
C_TXT   = '#1a1a1a'
C_MID   = '#666666'
C_MUTED = '#aaaaaa'
C_BG    = '#ffffff'
C_GRID  = '#f0f0f0'

fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), facecolor=C_BG,
                         gridspec_kw={'width_ratios': [1.1, 0.9, 1.0]})
fig.patch.set_facecolor(C_BG)

for ax in axes:
    ax.set_facecolor(C_BG)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#e0e0e0')
    ax.tick_params(colors=C_MID, labelsize=9)

ax1 = axes[0]

ax1.plot(years, dos, color=C_HI2, lw=2.8, zorder=3)

ax1.scatter([years[0], years[-1]],
            [dos[0], dos[-1]],
            color=C_HI2, zorder=4)

ax1.text(years[0], dos[0] + 80,
         f'{int(dos[0]):,}d',
         fontsize=9, color=C_MID, ha='center')

ax1.text(years[-1], dos[-1] + 80,
         f'{int(dos[-1]):,}d',
         fontsize=9, color=C_HI2, ha='center', fontweight='bold')

ratio = dos[-1] / dos[0]
ax1.text(0.02, 0.92,
         f'↑ {ratio:.1f}x in 10 years',
         transform=ax1.transAxes,
         fontsize=10, color=C_HI2, fontweight='bold')

ax1.text(0.5, -0.18,
    'Tích lũy liên tục, mang tính hệ thống',
    transform=ax1.transAxes, ha='center',
    fontsize=8.5, color=C_MID, style='italic')

ax1.set_xticks(years)
ax1.set_xticklabels(years, rotation=45, ha='right')
ax1.set_ylabel('Days of Supply', fontsize=9, color=C_MID)
ax1.set_ylim(0, 2000)
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}')
)

ax1.grid(axis='y', color=C_GRID, lw=1)
ax1.set_title('① DoS tăng mạnh theo thời gian',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)
ax2 = axes[1]

sd_days = sd_show.index.tolist()

sd_pcts = sd_show.values / sd_show.values.sum() * 100

bars2 = ax2.bar(sd_days, sd_pcts,
                color=[C_RED if d <= 2 else C_LGRAY for d in sd_days],
                width=0.7)

for d, p in zip(sd_days, sd_pcts):
    if d <= 2:
        ax2.text(d, p + 1,
                 f'{p:.0f}%',
                 ha='center', fontsize=9,
                 color=C_RED, fontweight='bold')

tail_pct = sd_pcts[-1]
ax2.text(sd_days[-1], tail_pct + 1,
         f'{tail_pct:.1f}%',
         ha='center', fontsize=8, color=C_MUTED)

ax2.text(0.5, -0.22,
    'Thiếu hàng chủ yếu rất ngắn hạn (1–2 ngày)',
    transform=ax2.transAxes, ha='center',
    fontsize=8.5, color=C_MID, style='italic')

ax2.set_xticks(sd_days)
ax2.set_xticklabels([f'{d}d' for d in sd_days])
ax2.set_xlabel('Stockout duration (days)', fontsize=9, color=C_MID)
ax2.set_ylabel('% of stockout cases', fontsize=9, color=C_MID)

ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax2.set_ylim(0, max(sd_pcts) * 1.4)

ax2.grid(axis='y', color=C_GRID, lw=1)
ax2.set_title('② Thiếu hàng chủ yếu 1–2 ngày',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)
ax3 = axes[2]
ax3.axis('off')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

ax3.set_title('③ Tỷ lệ snapshot theo cờ tồn kho',
              fontsize=10.5, fontweight='bold', color=C_TXT, pad=10)

stats = [
    ('67,3%', 'Có stockout'),
    ('76,3%', 'Có overstock'),
    ('50,6%', 'Có cả hai'),
]

ys = [0.75, 0.45, 0.15]

for (num, lbl), y in zip(stats, ys):
    ax3.text(0.5, y,
             num,
             ha='center', fontsize=30,
             fontweight='bold', color=C_TXT)
    ax3.text(0.5, y - 0.08,
             lbl,
             ha='center', fontsize=10,
             color=C_MID)

ax3.text(0.5, 0.02,
         '50,6% snapshot gặp đồng thời thiếu & dư → vấn đề hệ thống',
         ha='center', fontsize=9,
         color=C_MID, style='italic')

fig.text(0.01, 0.995,
    'Thiếu hàng ngắn hạn kích hoạt nhập hàng quá mức, dẫn đến tích lũy tồn kho dài hạn',
    fontsize=12.5, fontweight='bold', color=C_TXT, va='top')
fig.text(0.01, 0.955,
    'Thiếu hàng ngắn (1–2 ngày) nhưng kích hoạt nhập hàng quá mức  ·  reorder_flag = 0 trên toàn bộ dữ liệu',
    fontsize=8.5, color=C_MUTED, va='top')

plt.tight_layout(rect=[0, 0.08, 1, 0.93])
plt.savefig('chart_inventory_bullwhip.png', dpi=150,
            bbox_inches='tight', facecolor=C_BG)
print('Saved.')

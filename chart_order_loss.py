import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib.pyplot as plt
import numpy as np

total     = 646_945
delivered = 516_716
cancelled =  59_462
returned  =  36_142
inflight  =  34_625

loss_n   = cancelled + returned
loss_pct = loss_n / total * 100

sizes  = [delivered, cancelled, returned, inflight]
labels = ['Giao thành công', 'Huỷ đơn', 'Hoàn trả', 'Đang xử lý']
pcts   = [s / total * 100 for s in sizes]

C_HI    = '#98f16d'
C_HI2   = '#5db83d'
C_GRAY  = '#c8c8c8'
C_LGRAY = '#e0e0e0'
C_TXT   = '#1a1a1a'
C_MID   = '#666666'
C_MUTED = '#aaaaaa'
C_BG    = '#ffffff'
colors  = [C_GRAY, C_HI, C_HI2, C_LGRAY]

fig, ax = plt.subplots(figsize=(8, 6), facecolor=C_BG)
ax.set_facecolor(C_BG)
ax.set_aspect('equal')

wedges, _ = ax.pie(
    sizes,
    colors=colors,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.46, edgecolor=C_BG, linewidth=3),
)

for i in (1, 2):
    ang = np.radians((wedges[i].theta1 + wedges[i].theta2) / 2)
    wedges[i].set_center((0.06 * np.cos(ang), 0.06 * np.sin(ang)))

ax.text(0, 0.15, f'{loss_pct:.1f}%',
        ha='center', va='center', fontsize=32, fontweight='bold', color=C_HI)
ax.text(0, -0.18, 'đơn không tạo\ndoanh thu cuối cùng',
        ha='center', va='center', fontsize=9, color=C_MID, linespacing=1.65)

def mid_ang(w):
    return np.radians((w.theta1 + w.theta2) / 2)

R_INNER = 0.78   # just outside outer edge of donut
R_OUTER = 1.05   # elbow point

text_pos = {
    0: ( 0.68, -0.82, 'center'),   # Delivered  → lower area
    1: (-1.30,  0.42, 'right'),    # Cancelled  → mid-left
    2: (-1.30, -0.22, 'right'),    # Returned   → lower-left
    3: ( 1.30,  0.78, 'left'),     # In-flight  → upper-right
}
bold_set = {1, 2}

for i, wedge in enumerate(wedges):
    ang   = mid_ang(wedge)
    cx, cy = wedge.center   # offset if exploded

    xe = cx + R_INNER * np.cos(ang)
    ye = cy + R_INNER * np.sin(ang)

    xt, yt, ha = text_pos[i]

    bold  = i in bold_set
    col   = {0: C_MUTED, 1: C_HI, 2: C_HI2, 3: C_MUTED}[i]
    fw    = 'bold' if bold else 'normal'
    lw    = 1.2 if bold else 0.8

    xm = xt + (0.08 if xt < 0 else -0.08)   # elbow x (near text)
    ax.annotate('',
        xy=(xe, ye), xytext=(xm, yt),
        arrowprops=dict(arrowstyle='-', color=col, lw=lw,
                        connectionstyle='angle,angleA=90,angleB=0,rad=0'))
    ax.plot([xm, xt], [yt, yt], color=col, lw=lw, solid_capstyle='round')

    ax.text(xt + (0.06 if xt > 0 else -0.06), yt + 0.09,
            f'{pcts[i]:.1f}%',
            ha=ha, va='center', fontsize=11, fontweight=fw, color=col)
    ax.text(xt + (0.06 if xt > 0 else -0.06), yt - 0.09,
            labels[i],
            ha=ha, va='center', fontsize=8.5, color=col)

ax.set_xlim(-1.85, 1.85)
ax.set_ylim(-1.25, 1.30)
ax.axis('off')

ax.text(0, -1.22,
        f'Total mất đơn (huỷ + hoàn trả):  {loss_n:,}  —  {loss_pct:.1f}%',
        ha='center', va='center', fontsize=9.5, color=C_HI, fontweight='bold')

fig.text(0.04, 0.98,
         'Phân bổ trạng thái đơn hàng  —  Tỷ lệ mất đơn thực tế',
         fontsize=12.5, fontweight='bold', color=C_TXT, va='top')
fig.text(0.04, 0.935,
         f'Tổng {total:,} đơn  |  2012 – 2022',
         fontsize=8.5, color=C_MUTED, va='top')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('chart_order_loss.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print('Saved.')

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

orders      = pd.read_csv('orders.csv',      parse_dates=['order_date'])
order_items = pd.read_csv('order_items.csv', low_memory=False)
products    = pd.read_csv('products.csv')
inventory   = pd.read_csv('inventory.csv',  parse_dates=['snapshot_date'])
sales       = pd.read_csv('sales.csv',      parse_dates=['Date'])

sales['year'] = sales['Date'].dt.year
sy = sales.groupby('year').agg(rev=('Revenue','sum'), cogs=('COGS','sum')).reset_index()
sy['margin'] = (sy['rev'] - sy['cogs']) / sy['rev'] * 100
avg_margin   = sy['margin'].mean()

order_items['revenue'] = (order_items['quantity'] * order_items['unit_price']
                          - order_items['discount_amount'])
oi = order_items.merge(products[['product_id','cogs']], on='product_id', how='left')
oi['gp'] = oi['revenue'] - oi['quantity'] * oi['cogs']
sku_gp      = oi.groupby('product_id')['gp'].sum()
n_loss_sku  = (sku_gp < 0).sum()
n_total_sku = len(sku_gp)
loss_amt    = sku_gp[sku_gp < 0].sum() / 1e6

orders['year'] = orders['order_date'].dt.year
by_yr = orders.groupby('year')['customer_id'].apply(set).to_dict()
ret_rows = []
for y in sorted(by_yr)[:-1]:
    cur, nxt = by_yr[y], by_yr.get(y + 1, set())
    ret_rows.append({'year': y, 'ret': len(cur & nxt) / len(cur) * 100})
ret_df   = pd.DataFrame(ret_rows)
ret_2012 = ret_df.loc[ret_df['year'] == 2012, 'ret'].iat[0]
ret_last  = ret_df['ret'].iloc[-1]

ord_rev   = order_items.groupby('order_id')['revenue'].sum().reset_index()
orders    = orders.merge(ord_rev, on='order_id', how='left')
cust_rev  = orders.groupby('customer_id')['revenue'].sum().sort_values(ascending=False)
n_total   = len(cust_rev)
top_n     = int(n_total * 0.253)
champ_pct = cust_rev.iloc[:top_n].sum() / cust_rev.sum() * 100

dos_yr   = inventory.groupby('year')['days_of_supply'].median().reset_index()
dos_2012 = dos_yr.loc[dos_yr['year'] == dos_yr['year'].min(), 'days_of_supply'].iat[0]
dos_peak = dos_yr['days_of_supply'].max()
dos_ratio = dos_peak / dos_2012

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

fig = plt.figure(figsize=(16, 5.5), facecolor=C_BG)
gs  = gridspec.GridSpec(
    2, 3, figure=fig,
    left=0.05, right=0.97, top=0.88, bottom=0.09,
    hspace=0.20, wspace=0.34,
    height_ratios=[1, 2],
)
ax_k1 = fig.add_subplot(gs[0, 0])
ax_k2 = fig.add_subplot(gs[0, 1])
ax_k3 = fig.add_subplot(gs[0, 2])
ax_c1 = fig.add_subplot(gs[1, 0])
ax_c2 = fig.add_subplot(gs[1, 1])
ax_c3 = fig.add_subplot(gs[1, 2])

def kpi_panel(ax, number_txt, number_color, label_top, label_bot, detail):
    ax.set_facecolor(C_BG)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.88, label_top, ha='center', va='top',
            fontsize=10, color=C_MID, fontweight='bold')
    ax.text(0.5, 0.58, number_txt, ha='center', va='center',
            fontsize=30, fontweight='bold', color=number_color)
    ax.text(0.5, 0.28, label_bot, ha='center', va='center',
            fontsize=9, color=C_MID)
    ax.text(0.5, 0.06, detail, ha='center', va='bottom',
            fontsize=8, color=C_NEUTRAL, style='italic')

kpi_panel(ax_k1,
          f'{avg_margin:.2f}%', C_ALERT,
          '① Khủng hoảng biên lợi nhuận',
          f'Biên LN gộp trung bình (ngành TT: 45–60%)',
          f'{n_loss_sku} SKU ({n_loss_sku/n_total_sku*100:.0f}% catalog) bán lỗ  |  Lỗ ròng {abs(loss_amt):.0f}M VND')

kpi_panel(ax_k2,
          f'{ret_2012:.0f}% → {ret_last:.0f}%', C_ALERT,
          '② Suy giảm giữ chân khách hàng',
          f'Tỷ lệ giữ chân YoY (2012 → {int(ret_df["year"].iloc[-1])})',
          f'Top 25.3% khách hàng đóng góp {champ_pct:.0f}% doanh thu')

kpi_panel(ax_k3,
          f'{dos_ratio:.1f}×', C_ALERT,
          '③ Tồn kho mất kiểm soát',
          f'Tăng median days-of-supply ({int(dos_2012)} → {int(dos_peak)} ngày)',
          f'67% snapshot có stockout  |  76% có overstock  |  reorder_flag = 0')

ax_c1.set_facecolor(C_BG)
ax_c1.plot(sy['year'], sy['margin'], marker='o', color=C_ALERT, lw=2.4, zorder=3)
ax_c1.fill_between(sy['year'], sy['margin'], color=C_ALERT, alpha=0.12, zorder=1)

ax_c1.axhline(45, color=C_MAIN, lw=1.6, ls='--', zorder=2)
ax_c1.text(sy['year'].max() + 0.15, 45.5, 'Ngành\n45%', fontsize=8, color=C_MAIN, va='bottom')

ax_c1.axhline(avg_margin, color=C_NEUTRAL, lw=1.2, ls=':', zorder=2)
ax_c1.text(sy['year'].min() - 0.1, avg_margin + 0.5,
           f'TB {avg_margin:.1f}%', fontsize=8, color=C_NEUTRAL, ha='right')

gap_to_industry = 45 - avg_margin
ax_c1.annotate('',
               xy=(2017, avg_margin), xytext=(2017, 45),
               arrowprops=dict(arrowstyle='<->', color=C_GRAY, lw=1.2))
ax_c1.text(2017.2, (avg_margin + 45) / 2, f'−{gap_to_industry:.0f}pp\nvs. ngành',
           fontsize=8, color=C_GRAY, va='center')

for y, m in zip(sy['year'], sy['margin']):
    if y in {2012, 2018, 2021, 2022}:
        ax_c1.text(y, m + 0.7, f'{m:.1f}%', ha='center', fontsize=8, color=C_DARK)

ax_c1.set_ylabel('Biên LN gộp (%)', color=C_MID, fontsize=9)
ax_c1.set_xlabel('Năm', color=C_MID)
ax_c1.set_title('Biên LN gộp theo năm', fontsize=11, fontweight='bold', color=C_TXT)
ax_c1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax_c1.set_ylim(0, 55)
ax_c1.set_xlim(sy['year'].min() - 0.8, sy['year'].max() + 1.7)

ax_c2.set_facecolor(C_BG)
ax_c2.plot(ret_df['year'], ret_df['ret'], marker='o', color=C_ACCENT, lw=2.4, zorder=3)
ax_c2.fill_between(ret_df['year'], ret_df['ret'], color=C_ACCENT, alpha=0.12, zorder=1)

ax_c2.axvspan(2017.5, ret_df['year'].max() + 0.5, color=C_ALERT, alpha=0.06, zorder=0)
ax_c2.text(2018.2, 62, 'Sụt mạnh\ntừ 2018', fontsize=8.5, color=C_ALERT)

ax_c2.text(ret_df['year'].iloc[0], ret_df['ret'].iloc[0] + 1.5,
           f'{ret_df["ret"].iloc[0]:.1f}%', ha='center', fontsize=9,
           fontweight='bold', color=C_ACCENT)
ax_c2.text(ret_df['year'].iloc[-1], ret_df['ret'].iloc[-1] - 3.5,
           f'{ret_df["ret"].iloc[-1]:.1f}%', ha='center', fontsize=9,
           fontweight='bold', color=C_ALERT)

drop_pp = ret_df['ret'].iloc[0] - ret_df['ret'].iloc[-1]
ax_c2.text(0.04, 0.08,
           f'Giảm {drop_pp:.0f}pp trong 9 năm',
           transform=ax_c2.transAxes, fontsize=9, color=C_ALERT,
           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', alpha=0.9))

ax_c2.set_ylabel('Tỷ lệ giữ chân YoY (%)', color=C_MID, fontsize=9)
ax_c2.set_xlabel('Năm', color=C_MID)
ax_c2.set_title('Tỷ lệ giữ chân khách hàng', fontsize=11, fontweight='bold', color=C_TXT)
ax_c2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax_c2.set_ylim(20, 75)
ax_c2.set_xlim(ret_df['year'].min() - 0.5, ret_df['year'].max() + 0.6)

ax_c3.set_facecolor(C_BG)
bar_clr = [C_ALERT if v == dos_yr['days_of_supply'].max() else C_GRAY2
           for v in dos_yr['days_of_supply']]
ax_c3.bar(dos_yr['year'], dos_yr['days_of_supply'],
          color=bar_clr, edgecolor='white', lw=0.5, width=0.75, zorder=2)
ax_c3.plot(dos_yr['year'], dos_yr['days_of_supply'],
           marker='o', color=C_DARK, lw=1.6, ls='--', zorder=3, markersize=4)

ax_c3.text(dos_yr['year'].iloc[0], dos_yr['days_of_supply'].iloc[0] + 8,
           f'{dos_yr["days_of_supply"].iloc[0]:.0f} ngày',
           ha='center', fontsize=8.5, color=C_DARK, fontweight='bold')
idx_max = dos_yr['days_of_supply'].idxmax()
ax_c3.text(dos_yr.loc[idx_max, 'year'], dos_yr['days_of_supply'].max() + 8,
           f'{dos_yr["days_of_supply"].max():.0f} ngày\n(đỉnh)',
           ha='center', fontsize=8.5, color=C_ALERT, fontweight='bold')
ax_c3.text(dos_yr['year'].iloc[-1], dos_yr['days_of_supply'].iloc[-1] + 8,
           f'{dos_yr["days_of_supply"].iloc[-1]:.0f} ngày',
           ha='center', fontsize=8.5, color=C_DARK, fontweight='bold')

ax_c3.text(0.04, 0.92,
           'Không có reorder_flag nào được kích hoạt\n→ Thiếu hệ thống tái đặt hàng tự động',
           transform=ax_c3.transAxes, fontsize=8.5, color=C_MID, va='top',
           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', alpha=0.9))

ax_c3.set_ylabel('Median days-of-supply (ngày)', color=C_MID, fontsize=9)
ax_c3.set_xlabel('Năm', color=C_MID)
ax_c3.set_title('Tồn kho tích lũy theo năm', fontsize=11, fontweight='bold', color=C_TXT)
ax_c3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax_c3.set_ylim(0, dos_yr['days_of_supply'].max() * 1.22)
ax_c3.set_xlim(dos_yr['year'].min() - 0.7, dos_yr['year'].max() + 0.7)

fig.text(0.5, 0.928,
         f'646.945 đơn hàng  |  {n_total:,} khách hàng  |  {n_total_sku:,} SKU sản phẩm',
         ha='center', fontsize=10, color=C_MID)

for x in [0.373, 0.677]:
    fig.add_artist(plt.Line2D([x, x], [0.645, 0.885],
                              color=C_GRID, lw=1.2, transform=fig.transFigure))

plt.savefig('chart_eda_summary.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print('Saved chart_eda_summary.png')

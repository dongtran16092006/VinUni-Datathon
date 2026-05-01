import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

customers = pd.read_csv('customers.csv', parse_dates=['signup_date'])
orders    = pd.read_csv('orders.csv',    parse_dates=['order_date'])
order_items = pd.read_csv('order_items.csv', low_memory=False)
products  = pd.read_csv('products.csv')
returns   = pd.read_csv('returns.csv',   parse_dates=['return_date'])
geo       = pd.read_csv('geography.csv')

order_items['revenue'] = (order_items['quantity'] * order_items['unit_price']
                          - order_items['discount_amount'])
order_items = order_items.merge(products[['product_id', 'cogs']], on='product_id', how='left')
order_items['gross_profit'] = (order_items['revenue']
                               - order_items['quantity'] * order_items['cogs'])

order_rev = order_items.groupby('order_id').agg(
    revenue=('revenue', 'sum'),
    gross_profit=('gross_profit', 'sum'),
).reset_index()

orders = orders.merge(order_rev, on='order_id', how='left')
orders['year'] = orders['order_date'].dt.year

returns = returns.merge(orders[['order_id', 'order_date']], on='order_id', how='left')
returns['year'] = returns['order_date'].dt.year

net_year = orders.groupby('year').agg(order_rev=('revenue', 'sum')).reset_index()
refund_by_year = returns.groupby('year').agg(refund=('refund_amount', 'sum')).reset_index()
net_year = net_year.merge(refund_by_year, on='year', how='left').fillna(0)
net_year['net_rev'] = net_year['order_rev'] - net_year['refund']
net_year = net_year.sort_values('year')

start_year = net_year['year'].iloc[0]
end_year   = net_year['year'].iloc[-1]
start_val  = net_year.loc[net_year['year'] == start_year, 'net_rev'].iat[0]
end_val    = net_year.loc[net_year['year'] == end_year,   'net_rev'].iat[0]
cagr = (end_val / start_val) ** (1 / (end_year - start_year)) - 1

overall_margin = order_items['gross_profit'].sum() / order_items['revenue'].sum()

orders_by_year = orders.groupby('year')['customer_id'].apply(lambda s: set(s)).to_dict()
retention = []
for year in sorted(orders_by_year)[:-1]:
    current   = orders_by_year[year]
    next_year = orders_by_year.get(year + 1, set())
    if current:
        retention.append({
            'year': year,
            'retention_pct': len(current & next_year) / len(current) * 100,
        })
ret_df = pd.DataFrame(retention)
latest_retention = ret_df['retention_pct'].iloc[-1]
avg_retention    = ret_df['retention_pct'].mean()

returned_orders = returns['order_id'].nunique()
return_rate = returned_orders / len(orders) * 100

zip_region    = geo.drop_duplicates('zip')[['zip', 'region']]
customers_geo = customers.merge(zip_region, on='zip', how='left')
orders_geo    = orders.merge(customers_geo[['customer_id', 'region']], on='customer_id', how='left')
region_rev    = orders_geo.groupby('region').agg(
    n_customers=('customer_id', 'nunique'),
    total_rev=('revenue', 'sum'),
).reset_index().dropna(subset=['region'])
region_rev['cust_share']  = region_rev['n_customers'] / region_rev['n_customers'].sum() * 100
region_rev['rev_share']   = region_rev['total_rev'] / region_rev['total_rev'].sum() * 100
region_rev['rev_per_cust']= region_rev['total_rev'] / region_rev['n_customers']
region_rev = region_rev.sort_values('rev_per_cust', ascending=False)

west    = region_rev[region_rev['region'] == 'West'].iloc[0]
east    = region_rev[region_rev['region'] == 'East'].iloc[0]
central = region_rev[region_rev['region'] == 'Central'].iloc[0]
ratio_east    = west['rev_per_cust'] / east['rev_per_cust']
ratio_central = west['rev_per_cust'] / central['rev_per_cust']

C_BG      = '#ffffff'
C_TXT     = '#1a1a1a'
C_MID     = '#555555'
C_GRID    = '#f0f0f0'
C_MAIN    = '#98f16d'
C_SECOND  = '#5f7f44'
C_DARK    = '#222222'
C_ALERT   = '#d9534f'
C_NEUTRAL = '#999999'
C_ACCENT  = '#4a90d9'   # for East bar

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 10, 'axes.grid': True,
    'grid.color': C_GRID, 'grid.linestyle': '-', 'grid.alpha': 0.35,
})

fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
gs  = gridspec.GridSpec(
    2, 3, figure=fig,
    left=0.06, right=0.97, top=0.90, bottom=0.08,
    hspace=0.22, wspace=0.38,
)

ax_kpi    = fig.add_subplot(gs[0, 0])
ax_trend  = fig.add_subplot(gs[0, 1:])
ax_ret    = fig.add_subplot(gs[1, :2])
ax_bar    = fig.add_subplot(gs[1, 2])   # FIX 2: bar chart replaces bubble chart

ax_kpi.axis('off')
ax_kpi.set_xlim(0, 1)
ax_kpi.set_ylim(0, 1)

ax_kpi.set_title('Key Metrics', fontsize=14, fontweight='bold', color=C_TXT, pad=10, loc='center')

kpis = [
    ('CAGR (2012–2022)',  f'{cagr*100:.1f}%'),
    ('Gross margin',      f'{overall_margin*100:.1f}%'),
]
for i, (label, value) in enumerate(kpis):
    ax_kpi.text(0.5, 0.70 - i * 0.24, value,
                fontsize=34, fontweight='bold', color=C_MAIN, ha='center')
    ax_kpi.text(0.5, 0.62 - i * 0.24, label,
                fontsize=12, color=C_MID, ha='center')

ax_kpi.text(
    0.5, 0.08,
    'Trend: tăng đến 2016, sau đó giảm\ntới 2019–2021 rồi phục hồi nhẹ 2022.',
    fontsize=9, color=C_MID, va='bottom', ha='center',
)

ax_trend.plot(net_year['year'], net_year['net_rev'] / 1e9,
              marker='o', lw=2.4, color=C_MAIN)
ax_trend.set_title('Xu hướng doanh thu ròng (2012–2022)',
                   fontsize=14, fontweight='bold', color=C_TXT)
ax_trend.set_xlabel('Năm', color=C_MID)
ax_trend.set_ylabel('Doanh thu ròng (B VND)', color=C_MID)
ax_trend.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax_trend.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}B'))
ax_trend.set_facecolor(C_BG)

highlight_years = {start_year, 2016, 2019, end_year}
for year, val in zip(net_year['year'], net_year['net_rev'] / 1e9):
    if year in highlight_years:
        ax_trend.text(year, val + 0.06, f'{val:.2f}B',
                      ha='center', fontsize=9, color=C_DARK)

peak_2016 = net_year.loc[net_year['year'] == 2016, 'net_rev'].iat[0] / 1e9
ax_trend.annotate(
    'Đỉnh 2016\n1.95B',
    xy=(2016, peak_2016), xytext=(2015.0, peak_2016 + 0.16),
    arrowprops=dict(arrowstyle='->', color=C_DARK), color=C_DARK, fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', alpha=0.95),
)

low_2019 = net_year.loc[net_year['year'] == 2019, 'net_rev'].iat[0] / 1e9
ax_trend.annotate(
    'Giảm tới 2019–2021',
    xy=(2019, low_2019), xytext=(2019.4, low_2019 - 0.12),
    arrowprops=dict(arrowstyle='->', color=C_ALERT), color=C_ALERT, fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', alpha=0.95),
)

ax_trend.set_xlim(start_year - 0.3, end_year + 0.3)
ax_trend.set_ylim(
    (net_year['net_rev'] / 1e9).min() - 0.15,
    (net_year['net_rev'] / 1e9).max() + 0.40,
)

ax_ret.plot(ret_df['year'], ret_df['retention_pct'],
            marker='o', lw=2.2, color=C_MAIN)
ax_ret.set_title('Tỷ lệ giữ chân năm Y+1',
                 fontsize=14, fontweight='bold', color=C_TXT)
ax_ret.set_xlabel('Năm cohort', color=C_MID)
ax_ret.set_ylabel('Tỷ lệ giữ chân năm Y+1 (%)', color=C_MID)
ax_ret.set_facecolor(C_BG)
ax_ret.set_xlim(ret_df['year'].min() - 0.3, ret_df['year'].max() + 1.2)
ax_ret.set_ylim(0, max(ret_df['retention_pct'].max(), avg_retention) * 1.25)
ax_ret.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

for year, val in zip(ret_df['year'], ret_df['retention_pct']):
    if year in {ret_df['year'].min(), 2017, ret_df['year'].max()}:
        ax_ret.text(year, val + 1.2, f'{val:.1f}%',
                    ha='center', fontsize=9, color=C_DARK)

ax_ret.axhline(avg_retention, color=C_NEUTRAL, lw=1.4, ls='--', zorder=1)
ax_ret.fill_between(
    ret_df['year'], ret_df['retention_pct'], avg_retention,
    where=ret_df['retention_pct'] < avg_retention,
    interpolate=True, alpha=0.08, color=C_ALERT,
)
ax_ret.text(
    ret_df['year'].max() + 0.15, avg_retention + 1,
    f'Trung bình nội bộ\n{avg_retention:.1f}%',   # FIX 3: "Benchmark" → "Avg nội bộ"
    fontsize=8.5, color=C_NEUTRAL,
)

if 2017 in ret_df['year'].values:
    val_2016 = ret_df.loc[ret_df['year'] == 2016, 'retention_pct'].iat[0] if 2016 in ret_df['year'].values else None
    val_2017 = ret_df.loc[ret_df['year'] == 2017, 'retention_pct'].iat[0]
    arrow_x = 2017
    arrow_y = val_2017
    ax_ret.annotate(
        'Bắt đầu giảm\ntừ cohort 2017',    # FIX 4: text rõ hơn
        xy=(arrow_x, arrow_y),
        xytext=(2016.2, arrow_y - 10),
        arrowprops=dict(arrowstyle='->', color=C_ALERT), fontsize=8.5, color=C_ALERT,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ddd', alpha=0.95),
    )

regions_ordered = ['West', 'Central', 'East']
bar_colors = {
    'West':    C_MAIN,
    'Central': C_DARK,
    'East':    C_ACCENT,
}

rev_lookup  = region_rev.set_index('region')['rev_per_cust']
cust_lookup = region_rev.set_index('region')['cust_share']
revs_lookup = region_rev.set_index('region')['rev_share']

x      = np.arange(len(regions_ordered))
width  = 0.28

bars_cust = [cust_lookup[r] for r in regions_ordered]
bars_revs = [revs_lookup[r] for r in regions_ordered]

b1 = ax_bar.bar(x - width/2, bars_cust, width, label='% Khách hàng',
                color=[bar_colors[r] for r in regions_ordered], alpha=0.55, edgecolor=C_DARK, lw=0.6)
b2 = ax_bar.bar(x + width/2, bars_revs, width, label='% Doanh thu',
                color=[bar_colors[r] for r in regions_ordered], alpha=0.92, edgecolor=C_DARK, lw=0.6)

for bar in b1:
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8, color=C_MID)
for bar in b2:
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8.5,
                fontweight='bold', color=C_DARK)

for i, r in enumerate(regions_ordered):
    ax_bar.text(i, -3.2, f'{rev_lookup[r] / 1000:.0f}K VND/kh',
                ha='center', fontsize=8, color=C_MID, style='italic')

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(regions_ordered, fontsize=10, fontweight='bold')
ax_bar.set_ylabel('Tỷ lệ (%)', color=C_MID)
ax_bar.set_ylim(-5, max(bars_cust + bars_revs) * 1.28)
ax_bar.set_facecolor(C_BG)
ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax_bar.set_title(
    'Nghịch lý địa lý:\nWest ít khách, doanh thu cao',
    fontsize=12, fontweight='bold', color=C_TXT, pad=8,
)
ax_bar.legend(fontsize=8.5, loc='upper right', framealpha=0.85)
ax_bar.axhline(0, color=C_NEUTRAL, lw=0.6)

w_cust = cust_lookup['West']
w_rev  = revs_lookup['West']
ax_bar.annotate(
    f'+{w_rev - w_cust:.1f}pp\ngap',
    xy=(0 + width/2, w_rev),
    xytext=(0.35, w_rev + 2),
    fontsize=8, color=C_ALERT,
    arrowprops=dict(arrowstyle='->', color=C_ALERT, lw=0.8),
    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#ddd', alpha=0.9),
)

fig.text(
    0.01, 0.975,
    'Tổng quan doanh thu & khách hàng',
    fontsize=15, fontweight='bold', color=C_TXT, va='top',
)
fig.text(
    0.01, 0.95,
    ('Khu vực West chỉ 16.3% khách nhưng đóng góp 23.4% doanh thu; '
     'revenue/khách cao hơn East 1.53× và Central 1.62×.'),
    fontsize=10, color=C_MID, va='top',
)

plt.savefig('chart_overview_fixed.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print('Saved chart_overview_fixed.png')

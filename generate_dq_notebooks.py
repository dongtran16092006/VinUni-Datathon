"""
Generate 11 data quality notebooks for Datathon 2026 Round 1.
Run: python generate_dq_notebooks.py
"""
import nbformat as nbf
import os

def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"}
    }
    return n

def md(src):   return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

IMPORTS = """\
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

issues = []

def flag(label, mask_or_count, df=None, show_sample=True):
    count = int(mask_or_count.sum()) if hasattr(mask_or_count, 'sum') else int(mask_or_count)
    total = len(df) if df is not None else None
    pct   = f' ({count/total*100:.2f}%)' if total else ''
    status = '❌ ISSUE' if count > 0 else '✅ OK'
    print(f'{status}  {label}: {count:,}{pct}')
    if count > 0:
        issues.append(label)
        if show_sample and df is not None and hasattr(mask_or_count, 'sum'):
            sample = df[mask_or_count].head(3)
            print(sample.to_string(index=False))
    return count

def summary():
    print()
    if issues:
        print(f'══ {len(issues)} issue(s) found ══')
        for i in issues: print(f'  • {i}')
    else:
        print('══ All checks passed ══')
"""

nb01 = nb([
    md("# DQ-01 · customers.csv\nChecks: null rate, duplicates, FK → geography, temporal, domain values."),
    code(IMPORTS),
    code("""\
cust = pd.read_csv('customers.csv', parse_dates=['signup_date'])
geo  = pd.read_csv('geography.csv')
print(f'Shape: {cust.shape}')
cust.head(3)"""),

    md("## 1. Null rate"),
    code("""\
null_counts = cust.isnull().sum()
null_pct    = null_counts / len(cust) * 100
print(pd.DataFrame({'null_count': null_counts, 'null_%': null_pct.round(2)}))"""),

    md("## 2. Duplicate customer_id"),
    code("""\
flag('Duplicate customer_id', cust.duplicated(subset='customer_id'), cust)"""),

    md("## 3. FK: zip → geography.zip"),
    code("""\
valid_zips  = set(geo['zip'])
bad_zip     = ~cust['zip'].isin(valid_zips)
flag('customer zip not in geography', bad_zip, cust)"""),

    md("## 4. City consistency with zip"),
    code("""\
zip_city_map = geo.set_index('zip')['city'].to_dict()
cust['expected_city'] = cust['zip'].map(zip_city_map)
mismatch = cust['city'] != cust['expected_city']
flag('city ≠ geography.city for same zip', mismatch, cust)
cust.drop(columns='expected_city', inplace=True)"""),

    md("## 5. Domain values"),
    code("""\
VALID_GENDER  = {'Male','Female','Non-binary'}
VALID_AGE     = {'18-24','25-34','35-44','45-54','55+'}
VALID_CHANNEL = {'organic_search','social_media','paid_search','email_campaign','referral','direct'}

flag('Invalid gender',              ~cust['gender'].isin(VALID_GENDER),  cust)
flag('Invalid age_group',           ~cust['age_group'].isin(VALID_AGE),  cust)
flag('Invalid acquisition_channel', ~cust['acquisition_channel'].isin(VALID_CHANNEL), cust)"""),

    md("## 6. Temporal: signup_date sanity"),
    code("""\
flag('signup_date is null',           cust['signup_date'].isnull(),              cust)
flag('signup_date before 2000',       cust['signup_date'].dt.year < 2000,        cust)
flag('signup_date after 2022-12-31',  cust['signup_date'] > pd.Timestamp('2022-12-31'), cust)"""),

    md("## Summary"),
    code("summary()"),
])

nb02 = nb([
    md("# DQ-02 · orders.csv\nChecks: null rate, duplicates, FK → customers / geography, domain values, temporal."),
    code(IMPORTS),
    code("""\
orders = pd.read_csv('orders.csv', parse_dates=['order_date'])
cust   = pd.read_csv('customers.csv', parse_dates=['signup_date'])
geo    = pd.read_csv('geography.csv')
print(f'Shape: {orders.shape}')
orders.head(3)"""),

    md("## 1. Null rate"),
    code("""\
null_counts = orders.isnull().sum()
print(pd.DataFrame({'null_count': null_counts, 'null_%': (null_counts/len(orders)*100).round(2)}))"""),

    md("## 2. Duplicate order_id"),
    code("flag('Duplicate order_id', orders.duplicated(subset='order_id'), orders)"),

    md("## 3. FK: customer_id → customers"),
    code("""\
valid_cust = set(cust['customer_id'])
flag('customer_id not in customers', ~orders['customer_id'].isin(valid_cust), orders)"""),

    md("## 4. FK: zip → geography"),
    code("""\
valid_zips = set(geo['zip'])
flag('order zip not in geography', ~orders['zip'].isin(valid_zips), orders)"""),

    md("## 5. Domain values"),
    code("""\
VALID_STATUS  = {'delivered','cancelled','returned','shipped','paid','created'}
VALID_PAYMENT = {'credit_card','paypal','cod','apple_pay','bank_transfer'}
VALID_DEVICE  = {'mobile','desktop','tablet'}
VALID_SOURCE  = {'organic_search','paid_search','social_media','email_campaign','referral','direct'}

flag('Invalid order_status',    ~orders['order_status'].isin(VALID_STATUS),  orders)
flag('Invalid payment_method',  ~orders['payment_method'].isin(VALID_PAYMENT), orders)
flag('Invalid device_type',     ~orders['device_type'].isin(VALID_DEVICE),   orders)
flag('Invalid order_source',    ~orders['order_source'].isin(VALID_SOURCE),  orders)"""),

    md("## 6. Temporal: order_date ≥ signup_date"),
    code("""\
merged = orders.merge(cust[['customer_id','signup_date']], on='customer_id', how='left')
early  = merged['order_date'] < merged['signup_date']
flag('order_date < customer signup_date', early, merged)"""),

    md("## 7. Temporal: order_date range"),
    code("""\
flag('order_date before 2012-01-01', orders['order_date'] < pd.Timestamp('2012-01-01'), orders)
flag('order_date after  2022-12-31', orders['order_date'] > pd.Timestamp('2022-12-31'), orders)"""),

    md("## Summary"),
    code("summary()"),
])

nb03 = nb([
    md("# DQ-03 · order_items.csv\nChecks: null rate, duplicates, FK → orders / products / promotions, business rules (price formulas)."),
    code(IMPORTS),
    code("""\
items  = pd.read_csv('order_items.csv', low_memory=False)
orders = pd.read_csv('orders.csv')
prods  = pd.read_csv('products.csv')
promos = pd.read_csv('promotions.csv')
print(f'Shape: {items.shape}')
items.head(3)"""),

    md("## 1. Null rate"),
    code("""\
null_counts = items.isnull().sum()
print(pd.DataFrame({'null_count': null_counts, 'null_%': (null_counts/len(items)*100).round(2)}))"""),

    md("## 2. Duplicate (order_id, product_id)"),
    code("""\
dup = items.duplicated(subset=['order_id','product_id'], keep=False)
flag('Duplicate (order_id, product_id) rows', dup, items)"""),

    md("## 3. FK: order_id → orders"),
    code("""\
valid_orders = set(orders['order_id'])
flag('order_id not in orders', ~items['order_id'].isin(valid_orders), items)"""),

    md("## 4. FK: product_id → products"),
    code("""\
valid_prods = set(prods['product_id'])
flag('product_id not in products', ~items['product_id'].isin(valid_prods), items)"""),

    md("## 5. FK: promo_id → promotions (where not null)"),
    code("""\
valid_promos = set(promos['promo_id'])
has_promo    = items['promo_id'].notna()
flag('promo_id not in promotions', has_promo & ~items['promo_id'].isin(valid_promos), items)"""),

    md("## 6. promo_id_2 — entirely null?"),
    code("""\
print(f'promo_id_2 non-null: {items[\"promo_id_2\"].notna().sum()}')"""),

    md("## 7. Business rules: quantity, unit_price, discount_amount"),
    code("""\
flag('quantity ≤ 0',         items['quantity'] <= 0,         items)
flag('unit_price ≤ 0',       items['unit_price'] <= 0,       items)
flag('discount_amount < 0',  items['discount_amount'] < 0,   items)
flag('discount_amount > quantity × unit_price',
     items['discount_amount'] > items['quantity'] * items['unit_price'], items)"""),

    md("## 8. Business rule: discount formula"),
    code("""\
df = items.merge(promos[['promo_id','promo_type','discount_value']], on='promo_id', how='left')

pct = df[df['promo_type']=='percentage'].copy()
pct['expected'] = pct['quantity'] * pct['unit_price'] * (pct['discount_value']/100)
pct['diff']     = (pct['discount_amount'] - pct['expected']).abs()
flag('percentage promo: discount_amount error > 1.0', pct['diff'] > 1.0, pct)

fix = df[df['promo_type']=='fixed'].copy()
fix['expected'] = fix['quantity'] * fix['discount_value']
fix['diff']     = (fix['discount_amount'] - fix['expected']).abs()
flag('fixed promo: discount_amount error > 0.01', fix['diff'] > 0.01, fix)

no_p = df[df['promo_id'].isna()].copy()
flag('no promo but discount_amount > 0', no_p['discount_amount'] > 0, no_p)"""),

    md("## Summary"),
    code("summary()"),
])

nb04 = nb([
    md("# DQ-04 · payments.csv\nChecks: null rate, duplicates, FK → orders, domain values, business rules (payment ≈ order total)."),
    code(IMPORTS),
    code("""\
pay    = pd.read_csv('payments.csv')
orders = pd.read_csv('orders.csv')
items  = pd.read_csv('order_items.csv', low_memory=False)
print(f'Shape: {pay.shape}')
pay.head(3)"""),

    md("## 1. Null rate"),
    code("""\
null_counts = pay.isnull().sum()
print(pd.DataFrame({'null_count': null_counts, 'null_%': (null_counts/len(pay)*100).round(2)}))"""),

    md("## 2. 1:1 với orders (mỗi order có đúng 1 payment)"),
    code("""\
flag('Duplicate order_id in payments', pay.duplicated(subset='order_id'), pay)
orders_no_pay = ~orders['order_id'].isin(pay['order_id'])
flag('Orders without payment', orders_no_pay, orders)"""),

    md("## 3. FK: order_id → orders"),
    code("""\
valid_orders = set(orders['order_id'])
flag('order_id not in orders', ~pay['order_id'].isin(valid_orders), pay)"""),

    md("## 4. Domain values"),
    code("""\
VALID_PAYMENT = {'credit_card','paypal','cod','apple_pay','bank_transfer'}
VALID_INST    = {1, 2, 3, 6, 12}
flag('Invalid payment_method', ~pay['payment_method'].isin(VALID_PAYMENT), pay)
flag('Invalid installments',   ~pay['installments'].isin(VALID_INST),      pay)"""),

    md("## 5. Business rule: payment_value > 0"),
    code("""\
flag('payment_value ≤ 0', pay['payment_value'] <= 0, pay)"""),

    md("## 6. Business rule: payment_value ≈ net order total (qty×price − discount)"),
    code("""\
net_by_order = (
    items
    .assign(net=lambda d: d['quantity']*d['unit_price'] - d['discount_amount'])
    .groupby('order_id')['net'].sum()
    .reset_index()
    .rename(columns={'net':'calc_total'})
)
df = pay.merge(net_by_order, on='order_id', how='left')
df['diff']    = (df['payment_value'] - df['calc_total']).abs()
df['diff_pct']= df['diff'] / df['calc_total'] * 100

flag('payment_value differs from net order total by > 1 VND', df['diff'] > 1.0, df)
flag('payment_value differs by > 1%',                         df['diff_pct'] > 1.0, df)
print(f"\\nMax diff: {df['diff'].max():,.2f}  |  Mean diff: {df['diff'].mean():,.4f}")"""),

    md("## Summary"),
    code("summary()"),
])

nb05 = nb([
    md("# DQ-05 · products.csv\nChecks: null rate, duplicates, domain values, business rules (price, cogs, margin)."),
    code(IMPORTS),
    code("""\
prods = pd.read_csv('products.csv')
print(f'Shape: {prods.shape}')
prods.head(3)"""),

    md("## 1. Null rate"),
    code("""\
null_counts = prods.isnull().sum()
print(pd.DataFrame({'null_count': null_counts, 'null_%': (null_counts/len(prods)*100).round(2)}))"""),

    md("## 2. Duplicate product_id"),
    code("flag('Duplicate product_id', prods.duplicated(subset='product_id'), prods)"),

    md("## 3. Domain values"),
    code("""\
VALID_CAT     = {'Streetwear','Outdoor','Casual','GenZ'}
VALID_SEG     = {'Activewear','Everyday','Performance','Balanced','Standard','Premium','All-weather','Trendy'}
VALID_SIZE    = {'S','M','L','XL'}

flag('Invalid category', ~prods['category'].isin(VALID_CAT),  prods)
flag('Invalid segment',  ~prods['segment'].isin(VALID_SEG),   prods)
flag('Invalid size',     ~prods['size'].isin(VALID_SIZE),      prods)"""),

    md("## 4. Business rules: price, cogs"),
    code("""\
flag('price ≤ 0',  prods['price'] <= 0,  prods)
flag('cogs ≤ 0',   prods['cogs']  <= 0,  prods)
flag('cogs ≥ price (negative or zero margin)', prods['cogs'] >= prods['price'], prods)"""),

    md("## 5. Margin distribution"),
    code("""\
prods['margin'] = (prods['price'] - prods['cogs']) / prods['price']
print('Margin stats:')
print(prods['margin'].describe().round(4))
flag('Margin < 0',   prods['margin'] < 0,    prods)
flag('Margin > 0.9', prods['margin'] > 0.9,  prods)"""),

    md("## 6. Products không xuất hiện trong order_items"),
    code("""\
items = pd.read_csv('order_items.csv', low_memory=False)
ordered_prods = set(items['product_id'])
not_ordered = ~prods['product_id'].isin(ordered_prods)
flag('Products never ordered', not_ordered, prods)
print(prods[not_ordered][['product_id','product_name','category']].head(5).to_string(index=False))"""),

    md("## Summary"),
    code("summary()"),
])

nb06 = nb([
    md("# DQ-06 · promotions.csv\nChecks: null rate, duplicates, domain values, temporal (start ≤ end), coverage in order_items."),
    code(IMPORTS),
    code("""\
promos = pd.read_csv('promotions.csv', parse_dates=['start_date','end_date'])
print(f'Shape: {promos.shape}')
promos.head(3)"""),

    md("## 1. Null rate"),
    code("""\
null_counts = promos.isnull().sum()
print(pd.DataFrame({'null_count': null_counts, 'null_%': (null_counts/len(promos)*100).round(2)}))"""),

    md("## 2. Duplicate promo_id"),
    code("flag('Duplicate promo_id', promos.duplicated(subset='promo_id'), promos)"),

    md("## 3. Domain values"),
    code("""\
VALID_TYPE    = {'percentage','fixed'}
VALID_CHANNEL = {'all_channels','online','email','social_media','in_store'}

flag('Invalid promo_type',    ~promos['promo_type'].isin(VALID_TYPE),       promos)
flag('Invalid promo_channel', ~promos['promo_channel'].isin(VALID_CHANNEL), promos)
flag('discount_value ≤ 0',    promos['discount_value'] <= 0,                promos)
flag('stackable_flag not in {0,1}', ~promos['stackable_flag'].isin([0,1]),  promos)"""),

    md("## 4. Temporal: start_date ≤ end_date"),
    code("""\
flag('start_date > end_date', promos['start_date'] > promos['end_date'], promos)
flag('end_date before 2012',  promos['end_date'] < pd.Timestamp('2012-01-01'), promos)"""),

    md("## 5. percentage promo: discount_value ≤ 100"),
    code("""\
pct_promos = promos[promos['promo_type']=='percentage']
flag('percentage discount_value > 100', pct_promos['discount_value'] > 100, pct_promos)"""),

    md("## 6. Coverage: promos không được dùng trong order_items"),
    code("""\
items = pd.read_csv('order_items.csv', low_memory=False)
used_promos = set(items['promo_id'].dropna())
not_used = ~promos['promo_id'].isin(used_promos)
flag('Promo defined but never used in order_items', not_used, promos)
print(promos[not_used][['promo_id','promo_name','start_date','end_date']].to_string(index=False))"""),

    md("## 7. Order items có promo_id nằm ngoài thời hạn promo"),
    code("""\
orders = pd.read_csv('orders.csv', parse_dates=['order_date'])
df = items.merge(orders[['order_id','order_date']], on='order_id', how='left')
df = df[df['promo_id'].notna()].merge(
    promos[['promo_id','start_date','end_date']], on='promo_id', how='left')
out_of_range = (df['order_date'] < df['start_date']) | (df['order_date'] > df['end_date'])
flag('Order date outside promo date range', out_of_range, df)"""),

    md("## Summary"),
    code("summary()"),
])

nb07 = nb([
    md("# DQ-07 · returns.csv\nChecks: null rate, duplicates, FK, temporal (return_date ≥ order_date), business rules (refund ≤ payment, qty ≤ ordered)."),
    code(IMPORTS),
    code("""\
ret    = pd.read_csv('returns.csv', parse_dates=['return_date'])
orders = pd.read_csv('orders.csv', parse_dates=['order_date'])
items  = pd.read_csv('order_items.csv', low_memory=False)
pay    = pd.read_csv('payments.csv')
prods  = pd.read_csv('products.csv')
print(f'Shape: {ret.shape}')
ret.head(3)"""),

    md("## 1. Null rate"),
    code("""\
null_counts = ret.isnull().sum()
print(pd.DataFrame({'null_count': null_counts, 'null_%': (null_counts/len(ret)*100).round(2)}))"""),

    md("## 2. Duplicate return_id"),
    code("flag('Duplicate return_id', ret.duplicated(subset='return_id'), ret)"),

    md("## 3. FK: order_id → orders"),
    code("""\
valid_orders = set(orders['order_id'])
flag('order_id not in orders', ~ret['order_id'].isin(valid_orders), ret)"""),

    md("## 4. FK: product_id → products"),
    code("""\
valid_prods = set(prods['product_id'])
flag('product_id not in products', ~ret['product_id'].isin(valid_prods), ret)"""),

    md("## 5. Only 'returned' orders should have returns"),
    code("""\
returned_orders = set(orders[orders['order_status']=='returned']['order_id'])
flag('Return for non-returned order status', ~ret['order_id'].isin(returned_orders), ret)"""),

    md("## 6. Domain values: return_reason"),
    code("""\
VALID_REASON = {'wrong_size','defective','not_as_described','changed_mind','late_delivery'}
flag('Invalid return_reason', ~ret['return_reason'].isin(VALID_REASON), ret)"""),

    md("## 7. Business rule: return_quantity ≥ 1"),
    code("flag('return_quantity ≤ 0', ret['return_quantity'] <= 0, ret)"),

    md("## 8. Business rule: return_quantity ≤ ordered quantity"),
    code("""\
qty_ordered = items.groupby(['order_id','product_id'])['quantity'].sum().reset_index()
qty_ordered.columns = ['order_id','product_id','qty_ordered']
df = ret.merge(qty_ordered, on=['order_id','product_id'], how='left')
flag('return_quantity > ordered quantity', df['return_quantity'] > df['qty_ordered'], df)"""),

    md("## 9. Business rule: refund_amount ≤ payment_value"),
    code("""\
df2 = ret.merge(pay[['order_id','payment_value']], on='order_id', how='left')
flag('refund_amount > payment_value', df2['refund_amount'] > df2['payment_value'], df2)
flag('refund_amount ≤ 0',            df2['refund_amount'] <= 0,                   df2)"""),

    md("## 10. Temporal: return_date ≥ order_date"),
    code("""\
df3 = ret.merge(orders[['order_id','order_date']], on='order_id', how='left')
flag('return_date < order_date', df3['return_date'] < df3['order_date'], df3)

gap = (df3['return_date'] - df3['order_date']).dt.days
print(f'\\nReturn gap (days) — order → return:')
print(gap.describe().round(1))
flag('Return gap > 365 days', gap > 365, df3)
flag('Return gap < 0 days',   gap < 0,   df3)"""),

    md("## Summary"),
    code("summary()"),
])

nb08 = nb([
    md("# DQ-08 · reviews.csv\nChecks: null rate, duplicates, FK, domain values, temporal (review_date ≥ order_date), customer × product consistency."),
    code(IMPORTS),
    code("""\
rev    = pd.read_csv('reviews.csv', parse_dates=['review_date'])
orders = pd.read_csv('orders.csv', parse_dates=['order_date'])
items  = pd.read_csv('order_items.csv', low_memory=False)
cust   = pd.read_csv('customers.csv')
prods  = pd.read_csv('products.csv')
print(f'Shape: {rev.shape}')
rev.head(3)"""),

    md("## 1. Null rate"),
    code("""\
null_counts = rev.isnull().sum()
print(pd.DataFrame({'null_count': null_counts, 'null_%': (null_counts/len(rev)*100).round(2)}))"""),

    md("## 2. Duplicate review_id"),
    code("flag('Duplicate review_id', rev.duplicated(subset='review_id'), rev)"),

    md("## 3. FK checks"),
    code("""\
valid_orders = set(orders['order_id'])
valid_prods  = set(prods['product_id'])
valid_cust   = set(cust['customer_id'])

flag('order_id not in orders',     ~rev['order_id'].isin(valid_orders),  rev)
flag('product_id not in products', ~rev['product_id'].isin(valid_prods), rev)
flag('customer_id not in customers', ~rev['customer_id'].isin(valid_cust), rev)"""),

    md("## 4. Domain values: rating"),
    code("""\
flag('Invalid rating (not in 1-5)', ~rev['rating'].isin([1,2,3,4,5]), rev)"""),

    md("## 5. Consistency: customer_id matches the order"),
    code("""\
order_cust = orders[['order_id','customer_id']].rename(columns={'customer_id':'expected_cust'})
df = rev.merge(order_cust, on='order_id', how='left')
flag('customer_id ≠ order.customer_id', df['customer_id'] != df['expected_cust'], df)"""),

    md("## 6. Consistency: product_id was actually in that order"),
    code("""\
ordered_pairs = set(zip(items['order_id'], items['product_id']))
rev_pairs     = set(zip(rev['order_id'],   rev['product_id']))
bad_pairs     = rev_pairs - ordered_pairs
flag('(order_id, product_id) not in order_items', len(bad_pairs), show_sample=False)
print(f'  Bad pairs (sample): {list(bad_pairs)[:5]}')"""),

    md("## 7. Temporal: review_date ≥ order_date"),
    code("""\
df2 = rev.merge(orders[['order_id','order_date']], on='order_id', how='left')
flag('review_date < order_date', df2['review_date'] < df2['order_date'], df2)

gap = (df2['review_date'] - df2['order_date']).dt.days
print(f'\\nReview gap (days) — order → review:')
print(gap.describe().round(1))
flag('Review gap > 365 days', gap > 365, df2)
flag('Review gap < 0 days',   gap < 0,   df2)"""),

    md("## 8. Only delivered orders should be reviewed"),
    code("""\
delivered_orders = set(orders[orders['order_status']=='delivered']['order_id'])
flag('Review for non-delivered order', ~rev['order_id'].isin(delivered_orders), rev)"""),

    md("## Summary"),
    code("summary()"),
])

nb09 = nb([
    md("# DQ-09 · shipments.csv\nChecks: null rate, duplicates, FK, temporal chain (order → ship → delivery), gap flags."),
    code(IMPORTS),
    code("""\
ship   = pd.read_csv('shipments.csv', parse_dates=['ship_date','delivery_date'])
orders = pd.read_csv('orders.csv',    parse_dates=['order_date'])
print(f'Shape: {ship.shape}')
ship.head(3)"""),

    md("## 1. Null rate"),
    code("""\
null_counts = ship.isnull().sum()
print(pd.DataFrame({'null_count': null_counts, 'null_%': (null_counts/len(ship)*100).round(2)}))"""),

    md("## 2. 1:0-or-1 với orders (mỗi order tối đa 1 shipment)"),
    code("""\
flag('Duplicate order_id in shipments', ship.duplicated(subset='order_id'), ship)"""),

    md("## 3. FK: order_id → orders"),
    code("""\
valid_orders = set(orders['order_id'])
flag('order_id not in orders', ~ship['order_id'].isin(valid_orders), ship)"""),

    md("## 4. Chỉ đơn shipped/delivered/returned mới có shipment"),
    code("""\
shippable = set(orders[orders['order_status'].isin(['shipped','delivered','returned'])]['order_id'])
flag('Shipment for non-shippable order status', ~ship['order_id'].isin(shippable), ship)"""),

    md("## 5. Temporal chain: order_date ≤ ship_date ≤ delivery_date"),
    code("""\
df = ship.merge(orders[['order_id','order_date']], on='order_id', how='left')

flag('ship_date < order_date',    df['ship_date']     < df['order_date'], df)
flag('delivery_date < ship_date', df['delivery_date'] < df['ship_date'],  df)

proc_gap = (df['ship_date']     - df['order_date']).dt.days
del_gap  = (df['delivery_date'] - df['ship_date']).dt.days
tot_gap  = (df['delivery_date'] - df['order_date']).dt.days

print('\\nProcessing gap (order → ship) days:');  print(proc_gap.describe().round(1))
print('\\nDelivery gap  (ship → delivery) days:'); print(del_gap.describe().round(1))
print('\\nTotal gap     (order → delivery) days:'); print(tot_gap.describe().round(1))"""),

    md("## 6. Flag bất thường về gap"),
    code("""\
df['proc_gap'] = (df['ship_date']     - df['order_date']).dt.days
df['del_gap']  = (df['delivery_date'] - df['ship_date']).dt.days

flag('Processing gap > 14 days (order→ship)',     df['proc_gap'] > 14, df)
flag('Processing gap < 0 days',                   df['proc_gap'] < 0,  df)
flag('Delivery gap > 30 days (ship→delivery)',     df['del_gap']  > 30, df)
flag('Delivery gap < 0 days',                     df['del_gap']  < 0,  df)"""),

    md("## 7. Business rule: shipping_fee ≥ 0"),
    code("flag('shipping_fee < 0', ship['shipping_fee'] < 0, ship)"),

    md("## Summary"),
    code("summary()"),
])

nb10 = nb([
    md("# DQ-10 · inventory.csv\nChecks: null rate, duplicates, FK → products, domain values, business rules (flags, rates), temporal."),
    code(IMPORTS),
    code("""\
inv   = pd.read_csv('inventory.csv', parse_dates=['snapshot_date'])
prods = pd.read_csv('products.csv')
print(f'Shape: {inv.shape}')
inv.head(3)"""),

    md("## 1. Null rate"),
    code("""\
null_counts = inv.isnull().sum()
print(pd.DataFrame({'null_count': null_counts, 'null_%': (null_counts/len(inv)*100).round(2)}))"""),

    md("## 2. Duplicate (snapshot_date, product_id)"),
    code("flag('Duplicate (snapshot_date, product_id)', inv.duplicated(subset=['snapshot_date','product_id']), inv)"),

    md("## 3. FK: product_id → products"),
    code("""\
valid_prods = set(prods['product_id'])
flag('product_id not in products', ~inv['product_id'].isin(valid_prods), inv)"""),

    md("## 4. Domain values: flags"),
    code("""\
for col in ['stockout_flag','overstock_flag','reorder_flag']:
    flag(f'{col} not in {{0,1}}', ~inv[col].isin([0,1]), inv)

print(f'\\nreorder_flag unique values: {inv[\"reorder_flag\"].unique()}')"""),

    md("## 5. Business rules: numeric columns"),
    code("""\
flag('stock_on_hand < 0',   inv['stock_on_hand'] < 0,   inv)
flag('units_received < 0',  inv['units_received'] < 0,  inv)
flag('units_sold < 0',      inv['units_sold'] < 0,      inv)
flag('stockout_days < 0',   inv['stockout_days'] < 0,   inv)
flag('days_of_supply < 0',  inv['days_of_supply'] < 0,  inv)
flag('fill_rate < 0',       inv['fill_rate'] < 0,       inv)
flag('fill_rate > 1',       inv['fill_rate'] > 1,       inv)
flag('sell_through_rate < 0', inv['sell_through_rate'] < 0, inv)"""),

    md("## 6. Consistency: stockout_flag vs stockout_days"),
    code("""\
flag('stockout_flag=1 but stockout_days=0',
     (inv['stockout_flag']==1) & (inv['stockout_days']==0), inv)
flag('stockout_flag=0 but stockout_days>0',
     (inv['stockout_flag']==0) & (inv['stockout_days']>0),  inv)"""),

    md("## 7. Temporal: snapshot_date là cuối tháng"),
    code("""\
import pandas as pd
inv['expected_eom'] = inv['snapshot_date'] + pd.offsets.MonthEnd(0)
flag('snapshot_date ≠ end of month', inv['snapshot_date'] != inv['expected_eom'], inv)
print(f'Date range: {inv[\"snapshot_date\"].min().date()} → {inv[\"snapshot_date\"].max().date()}')"""),

    md("## 8. Consistency: product_name, category, segment vs products"),
    code("""\
prod_ref = prods[['product_id','product_name','category','segment']].rename(
    columns={'product_name':'pname_ref','category':'cat_ref','segment':'seg_ref'})
df = inv.merge(prod_ref, on='product_id', how='left')
flag('product_name mismatch', df['product_name'] != df['pname_ref'], df)
flag('category mismatch',     df['category']     != df['cat_ref'],   df)
flag('segment mismatch',      df['segment']      != df['seg_ref'],   df)"""),

    md("## Summary"),
    code("summary()"),
])

nb11 = nb([
    md("# DQ-11 · geography.csv\nChecks: null rate, duplicates, domain values, FK reverse (zip coverage in customers & orders)."),
    code(IMPORTS),
    code("""\
geo    = pd.read_csv('geography.csv')
cust   = pd.read_csv('customers.csv')
orders = pd.read_csv('orders.csv')
print(f'Shape: {geo.shape}')
geo.head(3)"""),

    md("## 1. Null rate"),
    code("""\
null_counts = geo.isnull().sum()
print(pd.DataFrame({'null_count': null_counts, 'null_%': (null_counts/len(geo)*100).round(2)}))"""),

    md("## 2. Duplicate zip"),
    code("flag('Duplicate zip', geo.duplicated(subset='zip'), geo)"),

    md("## 3. Domain values: region"),
    code("""\
VALID_REGION = {'East','Central','West'}
flag('Invalid region', ~geo['region'].isin(VALID_REGION), geo)"""),

    md("## 4. Reverse FK: customer zips covered by geography"),
    code("""\
geo_zips  = set(geo['zip'])
cust_zips = set(cust['zip'])
missing   = cust_zips - geo_zips
flag('Customer zip not in geography', len(missing), show_sample=False)
print(f'  Missing zips (sample): {list(missing)[:10]}')"""),

    md("## 5. Reverse FK: order zips covered by geography"),
    code("""\
order_zips  = set(orders['zip'])
missing_ord = order_zips - geo_zips
flag('Order zip not in geography', len(missing_ord), show_sample=False)
print(f'  Missing zips (sample): {list(missing_ord)[:10]}')"""),

    md("## 6. zip ↔ city consistency (1 zip = 1 city?)"),
    code("""\
cities_per_zip = geo.groupby('zip')['city'].nunique()
flag('zip maps to multiple cities', (cities_per_zip > 1).sum(), show_sample=False)
print(geo.groupby('zip')['city'].nunique().describe())"""),

    md("## 7. zip ↔ region consistency"),
    code("""\
regions_per_zip = geo.groupby('zip')['region'].nunique()
flag('zip maps to multiple regions', (regions_per_zip > 1).sum(), show_sample=False)"""),

    md("## 8. City name consistency across customers & geography"),
    code("""\
zip_city = geo.set_index('zip')['city'].to_dict()
cust['expected_city'] = cust['zip'].map(zip_city)
mismatch = cust['city'].notna() & (cust['city'] != cust['expected_city'])
flag('Customer city ≠ geography city for same zip', mismatch, cust)"""),

    md("## Summary"),
    code("summary()"),
])

eda01 = nb([
    md("# EDA-01 · Catalog Profile\nPhân bổ category/segment, price tier, gross margin distribution."),

    code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})

prods = pd.read_csv('products.csv')
prods['gross_margin_vnd'] = prods['price'] - prods['cogs']
prods['margin_pct']       = prods['gross_margin_vnd'] / prods['price'] * 100

bins   = [0, 1_000, 3_000, 7_000, 15_000, 50_000]
labels = ['<1K', '1K–3K', '3K–7K', '7K–15K', '>15K']
prods['price_tier'] = pd.cut(prods['price'], bins=bins, labels=labels)

print(f'Products: {len(prods):,}  |  Categories: {prods["category"].nunique()}  |  Segments: {prods["segment"].nunique()}')
prods[['product_id','category','segment','price','cogs','gross_margin_vnd','margin_pct','price_tier']].head(5)"""),

    md("## 1. Category Distribution"),
    code("""\
cat_count = prods['category'].value_counts().reset_index()
cat_count.columns = ['category','count']
cat_count['pct'] = cat_count['count'] / len(prods) * 100
print(cat_count.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].barh(cat_count['category'], cat_count['count'], color='steelblue')
axes[0].set_xlabel('Number of products')
axes[0].set_title('Product count by category')
for i, (c, p) in enumerate(zip(cat_count['count'], cat_count['pct'])):
    axes[0].text(c + 5, i, f'{c:,} ({p:.1f}%)', va='center', fontsize=9)

axes[1].pie(cat_count['count'], labels=cat_count['category'],
            autopct='%1.1f%%', startangle=140,
            colors=['#4C72B0','#DD8452','#55A868','#C44E52'])
axes[1].set_title('Category share')

plt.tight_layout()
plt.show()"""),

    md("## 2. Segment Distribution"),
    code("""\
seg_count = prods['segment'].value_counts().reset_index()
seg_count.columns = ['segment','count']
seg_count['pct'] = seg_count['count'] / len(prods) * 100
print(seg_count.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(seg_count['segment'], seg_count['count'],
               color=plt.cm.tab10.colors[:len(seg_count)])
ax.set_xlabel('Number of products')
ax.set_title('Product count by segment')
for bar, (c, p) in zip(bars, zip(seg_count['count'], seg_count['pct'])):
    ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
            f'{c:,} ({p:.1f}%)', va='center', fontsize=9)
plt.tight_layout()
plt.show()"""),

    md("## 3. Category × Segment Cross-tab"),
    code("""\
ct = pd.crosstab(prods['segment'], prods['category'])
print(ct)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(ct.values, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels(ct.columns, rotation=20)
ax.set_yticks(range(len(ct.index)));   ax.set_yticklabels(ct.index)
ax.set_title('Product count: segment × category')
plt.colorbar(im, ax=ax, shrink=0.8)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = ct.values[i, j]
        ax.text(j, i, str(v) if v > 0 else '', ha='center', va='center',
                fontsize=9, color='white' if v > ct.values.max()*0.6 else 'black')
plt.tight_layout()
plt.show()"""),

    md("## 4. Price Distribution"),
    code("""\
print('Price (VND) summary:')
print(prods['price'].describe().apply(lambda x: f'{x:,.0f}').to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(prods['price'], bins=50, color='steelblue', edgecolor='white', linewidth=0.4)
axes[0].set_xlabel('Price (VND)')
axes[0].set_ylabel('Count')
axes[0].set_title('Price distribution')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}K'))

data   = [prods[prods['category']==c]['price'].values for c in prods['category'].unique()]
labels_cat = prods['category'].unique()
bp = axes[1].boxplot(data, labels=labels_cat, patch_artist=True,
                     medianprops={'color':'black','linewidth':1.5})
colors = ['#4C72B0','#DD8452','#55A868','#C44E52']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[1].set_ylabel('Price (VND)')
axes[1].set_title('Price by category (boxplot)')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}K'))
plt.tight_layout()
plt.show()"""),

    md("## 5. Price Tier Analysis"),
    code("""\
tier_count = prods['price_tier'].value_counts().sort_index().reset_index()
tier_count.columns = ['tier','count']
tier_count['pct'] = tier_count['count'] / len(prods) * 100

print('Price tier (VND):')
print(tier_count.to_string(index=False))

ct_tier = pd.crosstab(prods['price_tier'], prods['category'])
print('\\nPrice tier × category:')
print(ct_tier)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].bar(tier_count['tier'].astype(str), tier_count['count'], color='#4C72B0')
axes[0].set_xlabel('Price tier (VND)')
axes[0].set_ylabel('Count')
axes[0].set_title('Products by price tier')
for i, (c, p) in enumerate(zip(tier_count['count'], tier_count['pct'])):
    axes[0].text(i, c + 3, f'{c}\\n({p:.1f}%)', ha='center', fontsize=8)

ct_tier.plot(kind='bar', stacked=True, ax=axes[1],
             color=['#4C72B0','#DD8452','#55A868','#C44E52'], edgecolor='white')
axes[1].set_xlabel('Price tier')
axes[1].set_ylabel('Count')
axes[1].set_title('Price tier × category (stacked)')
axes[1].legend(title='Category', bbox_to_anchor=(1, 1))
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.show()"""),

    md("## 6. Gross Margin — Absolute (price − cogs, VND)"),
    code("""\
print('Gross margin VND (price − cogs) summary:')
print(prods['gross_margin_vnd'].describe().apply(lambda x: f'{x:,.0f}').to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(prods['gross_margin_vnd'], bins=50, color='#55A868', edgecolor='white', linewidth=0.4)
axes[0].set_xlabel('Gross margin (VND)')
axes[0].set_ylabel('Count')
axes[0].set_title('Gross margin distribution (absolute VND)')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}K'))

data_gm   = [prods[prods['category']==c]['gross_margin_vnd'].values for c in prods['category'].unique()]
bp2 = axes[1].boxplot(data_gm, labels=prods['category'].unique(), patch_artist=True,
                      medianprops={'color':'black','linewidth':1.5})
for patch, color in zip(bp2['boxes'], ['#4C72B0','#DD8452','#55A868','#C44E52']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[1].set_ylabel('Gross margin (VND)')
axes[1].set_title('Gross margin by category')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}K'))
plt.tight_layout()
plt.show()"""),

    md("## 7. Gross Margin % Distribution"),
    code("""\
print('Gross margin % summary:')
print(prods['margin_pct'].describe().round(2).to_string())

mbins  = [0, 5.01, 10, 15, 20, 25, 30, 35, 40, 45, 50, 101]
mlbls  = ['≤5%','5-10%','10-15%','15-20%','20-25%','25-30%','30-35%','35-40%','40-45%','45-50%','50%+']
prods['margin_bucket'] = pd.cut(prods['margin_pct'], bins=mbins, labels=mlbls, include_lowest=True)
mb = prods['margin_bucket'].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(mb.index.astype(str), mb.values, color='#C44E52', edgecolor='white', linewidth=0.4)
axes[0].set_xlabel('Margin bucket')
axes[0].set_ylabel('Count')
axes[0].set_title('Margin % distribution (all products)')
axes[0].tick_params(axis='x', rotation=45)

seg_margin = prods.groupby('category')['margin_pct'].mean().sort_values()
axes[1].barh(seg_margin.index, seg_margin.values, color='#C44E52', alpha=0.8)
axes[1].set_xlabel('Mean margin %')
axes[1].set_title('Average margin % by category')
for i, v in enumerate(seg_margin.values):
    axes[1].text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.show()"""),

    md("## 8. Margin % by Segment"),
    code("""\
seg_stats = prods.groupby('segment')['margin_pct'].agg(['mean','median','std','min','max']).round(2)
seg_stats.columns = ['mean%','median%','std%','min%','max%']
seg_stats = seg_stats.sort_values('mean%', ascending=False)
print(seg_stats.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
data_seg   = [prods[prods['segment']==s]['margin_pct'].values for s in seg_stats.index]
bp3 = ax.boxplot(data_seg, labels=seg_stats.index, patch_artist=True,
                 medianprops={'color':'black','linewidth':1.5}, vert=False)
cmap = plt.cm.RdYlGn
for i, patch in enumerate(bp3['boxes']):
    patch.set_facecolor(cmap(i / len(bp3['boxes']))); patch.set_alpha(0.75)
ax.set_xlabel('Gross margin %')
ax.set_title('Margin % distribution by segment')
ax.axvline(prods['margin_pct'].mean(), color='red', linestyle='--', alpha=0.6, label=f'Overall mean ({prods["margin_pct"].mean():.1f}%)')
ax.legend()
plt.tight_layout()
plt.show()"""),

    md("## 9. Price vs Margin % — Scatter by Category"),
    code("""\
fig, ax = plt.subplots(figsize=(10, 5))
color_map = {'Streetwear':'#4C72B0','Outdoor':'#DD8452','Casual':'#55A868','GenZ':'#C44E52'}
for cat, grp in prods.groupby('category'):
    ax.scatter(grp['price'], grp['margin_pct'],
               label=cat, alpha=0.4, s=15, color=color_map.get(cat,'gray'))
ax.set_xlabel('Price (VND)')
ax.set_ylabel('Gross margin %')
ax.set_title('Price vs Gross margin % by category')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}K'))
ax.legend(title='Category')
plt.tight_layout()
plt.show()"""),

    md("## 10. Price Tier × Margin % Summary"),
    code("""\
tier_margin = prods.groupby('price_tier', observed=True)['margin_pct'].agg(
    count='count', mean='mean', median='median', std='std', min='min', max='max'
).round(2)
print('Margin % by price tier:')
print(tier_margin.to_string())

fig, ax = plt.subplots(figsize=(9, 4))
tm = prods.groupby('price_tier', observed=True)['margin_pct'].mean()
ax.bar(tm.index.astype(str), tm.values, color='#4C72B0', edgecolor='white')
ax.set_xlabel('Price tier (VND)')
ax.set_ylabel('Mean margin %')
ax.set_title('Average gross margin % by price tier')
for i, v in enumerate(tm.values):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.show()"""),

    md("## Summary"),
    code("""\
print('=== Catalog Profile Summary ===')
print(f'Total SKUs       : {len(prods):,}')
print(f'Categories       : {prods["category"].nunique()}  ({", ".join(prods["category"].value_counts().index)})')
print(f'Segments         : {prods["segment"].nunique()}')
print()
print('Price (VND):')
for tier, cnt in prods["price_tier"].value_counts().sort_index().items():
    print(f'  {str(tier):>8}  {cnt:>4} SKUs ({cnt/len(prods)*100:.1f}%)')
print()
print(f'Gross margin %:')
print(f'  Overall mean   : {prods["margin_pct"].mean():.1f}%')
print(f'  Overall median : {prods["margin_pct"].median():.1f}%')
print(f'  Range          : {prods["margin_pct"].min():.1f}% – {prods["margin_pct"].max():.1f}%')
print()
print('By category:')
for cat, row in prods.groupby("category")["margin_pct"].agg(["mean","median"]).round(1).iterrows():
    print(f'  {cat:<12}  mean={row["mean"]}%  median={row["median"]}%')"""),
])

SETUP = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})

CAT_COLORS = ['#4C72B0','#DD8452','#55A868','#C44E52']

ret    = pd.read_csv('returns.csv',    parse_dates=['return_date'])
orders = pd.read_csv('orders.csv',     parse_dates=['order_date'])
items  = pd.read_csv('order_items.csv', low_memory=False)
prods  = pd.read_csv('products.csv')
pay    = pd.read_csv('payments.csv')
sales  = pd.read_csv('sales.csv',      parse_dates=['Date'])

items_full = (
    items
    .merge(orders[['order_id','order_date','order_status']], on='order_id', how='left')
    .merge(prods[['product_id','category','segment','price','cogs']], on='product_id', how='left')
)
items_full['revenue_line'] = items_full['quantity'] * items_full['unit_price']
items_full['cogs_line']    = items_full['quantity'] * items_full['cogs']
items_full['year']         = items_full['order_date'].dt.year

ret_full = (
    ret
    .merge(orders[['order_id','order_date']], on='order_id', how='left')
    .merge(prods[['product_id','category','segment']], on='product_id', how='left')
    .merge(pay[['order_id','payment_value']], on='order_id', how='left')
)
ret_full['year'] = ret_full['order_date'].dt.year

print(f'orders  : {len(orders):,}')
print(f'returns : {len(ret_full):,}  (unique orders: {ret_full["order_id"].nunique():,})')
print(f'items   : {len(items_full):,}')
"""

eda02 = nb([
    md("# EDA-02 · Return Analysis\nReturn rate %, top return reasons, category return heatmap, net revenue after returns."),
    code(SETUP),

    md("## 1. Overall Return Rate"),
    code("""\
total_orders   = len(orders)
returned_orders = orders[orders['order_status']=='returned']['order_id'].nunique()
del_ret_orders  = orders[orders['order_status'].isin(['delivered','returned'])]['order_id'].nunique()

total_qty   = items_full['quantity'].sum()
returned_qty = ret_full['return_quantity'].sum()

gross_rev  = items_full['revenue_line'].sum()
total_refund = ret_full['refund_amount'].sum()

print('=== Overall Return Metrics ===')
print(f'Return rate (order-level, returned / all)            : {returned_orders/total_orders*100:.2f}%')
print(f'Return rate (order-level, returned / delivered+ret)  : {returned_orders/del_ret_orders*100:.2f}%')
print(f'Return rate (item quantity, qty_returned / qty_sold) : {returned_qty/total_qty*100:.2f}%')
print(f'Return rate (value, refund / gross revenue)          : {total_refund/gross_rev*100:.2f}%')
print()
print(f'Total refund amount : {total_refund:,.0f} VND')
print(f'Avg refund/return   : {total_refund/len(ret_full):,.0f} VND')"""),

    md("## 2. Return Rate Trend by Year"),
    code("""\
yearly_orders  = orders.groupby(orders['order_date'].dt.year)['order_id'].count().rename('total')
yearly_returned = orders[orders['order_status']=='returned'].groupby(
    orders['order_date'].dt.year)['order_id'].count().rename('returned')
yearly_refund  = ret_full.groupby('year')['refund_amount'].sum()
yearly_rev     = sales.groupby(sales['Date'].dt.year)['Revenue'].sum()

trend = pd.concat([yearly_orders, yearly_returned], axis=1).fillna(0)
trend['return_rate_%'] = (trend['returned'] / trend['total'] * 100).round(2)
trend['refund_VND']    = yearly_refund
trend['gross_rev']     = yearly_rev
trend['refund_rate_%'] = (trend['refund_VND'] / trend['gross_rev'] * 100).round(2)
print(trend.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].bar(trend.index, trend['return_rate_%'], color='#C44E52', alpha=0.85)
axes[0].set_xlabel('Year'); axes[0].set_ylabel('Return rate %')
axes[0].set_title('Order return rate by year')
for i, (yr, v) in enumerate(trend['return_rate_%'].items()):
    axes[0].text(yr, v + 0.05, f'{v:.1f}%', ha='center', fontsize=8)

axes[1].bar(trend.index, trend['refund_rate_%'], color='#DD8452', alpha=0.85)
axes[1].set_xlabel('Year'); axes[1].set_ylabel('Refund / revenue %')
axes[1].set_title('Refund value rate by year (refund / gross revenue)')
for i, (yr, v) in enumerate(trend['refund_rate_%'].items()):
    if not pd.isna(v):
        axes[1].text(yr, v + 0.01, f'{v:.1f}%', ha='center', fontsize=8)
plt.tight_layout(); plt.show()"""),

    md("## 3. Top Return Reasons"),
    code("""\
reason_stats = (
    ret_full.groupby('return_reason')
    .agg(count=('return_id','count'),
         total_qty=('return_quantity','sum'),
         total_refund=('refund_amount','sum'))
    .sort_values('count', ascending=False)
    .reset_index()
)
reason_stats['count_%']  = reason_stats['count']  / len(ret_full) * 100
reason_stats['refund_%'] = reason_stats['total_refund'] / ret_full['refund_amount'].sum() * 100
print(reason_stats.round(2).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

colors_r = ['#C44E52','#DD8452','#4C72B0','#55A868','#9467BD']
axes[0].barh(reason_stats['return_reason'], reason_stats['count'], color=colors_r)
axes[0].set_xlabel('Number of returns')
axes[0].set_title('Return count by reason')
for bar, v, p in zip(axes[0].patches, reason_stats['count'], reason_stats['count_%']):
    axes[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                 f'{v:,} ({p:.1f}%)', va='center', fontsize=8)

axes[1].barh(reason_stats['return_reason'], reason_stats['total_refund']/1e6, color=colors_r)
axes[1].set_xlabel('Refund amount (M VND)')
axes[1].set_title('Refund value by reason')
for bar, v, p in zip(axes[1].patches, reason_stats['total_refund']/1e6, reason_stats['refund_%']):
    axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{v:.0f}M ({p:.1f}%)', va='center', fontsize=8)
plt.tight_layout(); plt.show()"""),

    md("## 4. Return Reason Trend by Year"),
    code("""\
reason_yr = (
    ret_full.groupby(['year','return_reason'])['return_id']
    .count().unstack(fill_value=0)
)
reason_yr_pct = reason_yr.div(reason_yr.sum(axis=1), axis=0) * 100
print('Return reason share by year (%):')
print(reason_yr_pct.round(1).to_string())

fig, ax = plt.subplots(figsize=(11, 4))
reason_yr_pct.plot(kind='bar', stacked=True, ax=ax,
                   color=colors_r, edgecolor='white', linewidth=0.4)
ax.set_xlabel('Year'); ax.set_ylabel('Share %')
ax.set_title('Return reason composition by year (stacked %)')
ax.legend(title='Reason', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.show()"""),

    md("## 5. Return Rate by Category"),
    code("""\
qty_sold = items_full.groupby('category')['quantity'].sum().rename('qty_sold')
qty_ret  = ret_full.groupby('category')['return_quantity'].sum().rename('qty_returned')
cat_stats = pd.concat([qty_sold, qty_ret], axis=1).fillna(0)
cat_stats['return_rate_%'] = (cat_stats['qty_returned'] / cat_stats['qty_sold'] * 100).round(2)

rev_by_cat    = items_full.groupby('category')['revenue_line'].sum().rename('gross_rev')
refund_by_cat = ret_full.groupby('category')['refund_amount'].sum().rename('refund')
cat_stats = cat_stats.join(rev_by_cat).join(refund_by_cat)
cat_stats['refund_rate_%'] = (cat_stats['refund'] / cat_stats['gross_rev'] * 100).round(2)

print(cat_stats.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
cats = cat_stats.index.tolist()

axes[0].bar(cats, cat_stats['return_rate_%'], color=CAT_COLORS[:len(cats)])
axes[0].set_ylabel('Return rate (qty) %')
axes[0].set_title('Return rate by category (quantity-based)')
for i, v in enumerate(cat_stats['return_rate_%']):
    axes[0].text(i, v + 0.05, f'{v:.2f}%', ha='center', fontsize=9)

axes[1].bar(cats, cat_stats['refund_rate_%'], color=CAT_COLORS[:len(cats)])
axes[1].set_ylabel('Refund / gross revenue %')
axes[1].set_title('Refund rate by category (value-based)')
for i, v in enumerate(cat_stats['refund_rate_%']):
    axes[1].text(i, v + 0.05, f'{v:.2f}%', ha='center', fontsize=9)
plt.tight_layout(); plt.show()"""),

    md("## 6. Category × Return Reason Heatmap"),
    code("""\
heat = (
    ret_full.groupby(['category','return_reason'])['return_id']
    .count().unstack(fill_value=0)
)
qty_sold_cat = items_full.groupby('category')['quantity'].sum()
heat_pct = heat.div(qty_sold_cat, axis=0) * 100

print('Return reason rate % (returns / qty sold):')
print(heat_pct.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

im1 = axes[0].imshow(heat.values, cmap='Reds', aspect='auto')
axes[0].set_xticks(range(len(heat.columns))); axes[0].set_xticklabels(heat.columns, rotation=25, ha='right', fontsize=8)
axes[0].set_yticks(range(len(heat.index)));   axes[0].set_yticklabels(heat.index)
axes[0].set_title('Return count: category × reason')
plt.colorbar(im1, ax=axes[0], shrink=0.8)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        axes[0].text(j, i, f'{heat.values[i,j]:,}', ha='center', va='center',
                     fontsize=8, color='white' if heat.values[i,j] > heat.values.max()*0.6 else 'black')

im2 = axes[1].imshow(heat_pct.values, cmap='YlOrRd', aspect='auto')
axes[1].set_xticks(range(len(heat_pct.columns))); axes[1].set_xticklabels(heat_pct.columns, rotation=25, ha='right', fontsize=8)
axes[1].set_yticks(range(len(heat_pct.index)));   axes[1].set_yticklabels(heat_pct.index)
axes[1].set_title('Return rate % (/ qty sold): category × reason')
plt.colorbar(im2, ax=axes[1], shrink=0.8)
for i in range(heat_pct.shape[0]):
    for j in range(heat_pct.shape[1]):
        axes[1].text(j, i, f'{heat_pct.values[i,j]:.2f}%', ha='center', va='center',
                     fontsize=8, color='white' if heat_pct.values[i,j] > heat_pct.values.max()*0.6 else 'black')
plt.tight_layout(); plt.show()"""),

    md("## 7. Category × Segment Return Rate Heatmap"),
    code("""\
qty_sold_seg = items_full.groupby(['category','segment'])['quantity'].sum().rename('qty_sold')
qty_ret_seg  = ret_full.groupby(['category','segment'])['return_quantity'].sum().rename('qty_ret')
seg_heat = pd.concat([qty_sold_seg, qty_ret_seg], axis=1).fillna(0)
seg_heat['rate_%'] = (seg_heat['qty_ret'] / seg_heat['qty_sold'] * 100).round(2)

pivot = seg_heat['rate_%'].unstack(level='segment').fillna(0)
print('Return rate % by category × segment:')
print(pivot.round(2).to_string())

fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_title('Return rate % (qty): category × segment')
plt.colorbar(im, ax=ax, shrink=0.7, label='Return rate %')
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        if v > 0:
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    fontsize=8, color='white' if v > pivot.values.max()*0.65 else 'black')
plt.tight_layout(); plt.show()"""),

    md("## 8. Net Revenue After Returns"),
    code("""\
refund_by_order_date = (
    ret_full.merge(orders[['order_id','order_date']], on='order_id', how='left')
    .groupby(orders['order_date'].dt.to_period('M').rename('month'))['refund_amount']
    .sum()
    .reset_index()
)
refund_by_order_date.columns = ['month','refund']
refund_by_order_date['month_ts'] = refund_by_order_date['month'].dt.to_timestamp()

sales['month'] = sales['Date'].dt.to_period('M')
monthly_rev = sales.groupby('month')[['Revenue','COGS']].sum().reset_index()
monthly_rev['month_ts'] = monthly_rev['month'].dt.to_timestamp()

df_net = monthly_rev.merge(refund_by_order_date[['month','refund']], on='month', how='left').fillna(0)
df_net['net_revenue'] = df_net['Revenue'] - df_net['refund']
df_net['refund_rate_%'] = df_net['refund'] / df_net['Revenue'] * 100

print('Monthly net revenue sample (last 12 months):')
print(df_net[['month','Revenue','refund','net_revenue','refund_rate_%']].tail(12).round(0).to_string(index=False))"""),

    code("""\
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

axes[0].fill_between(df_net['month_ts'], df_net['Revenue']/1e6,
                     alpha=0.3, color='#4C72B0', label='Gross revenue')
axes[0].fill_between(df_net['month_ts'], df_net['net_revenue']/1e6,
                     alpha=0.6, color='#55A868', label='Net revenue (after refund)')
axes[0].set_ylabel('Revenue (M VND)')
axes[0].set_title('Monthly Gross vs Net Revenue')
axes[0].legend(loc='upper left', fontsize=9)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.0f}M'))

axes[1].fill_between(df_net['month_ts'], df_net['refund_rate_%'],
                     alpha=0.7, color='#C44E52', label='Refund / Revenue %')
axes[1].set_ylabel('Refund rate %')
axes[1].set_title('Monthly refund rate (refund / gross revenue)')
axes[1].legend(loc='upper left', fontsize=9)
plt.tight_layout(); plt.show()"""),

    md("## 9. Net Revenue by Category (Annual)"),
    code("""\
gross_cat_yr  = items_full.groupby(['year','category'])['revenue_line'].sum().unstack(fill_value=0)
refund_cat_yr = ret_full.groupby(['year','category'])['refund_amount'].sum().unstack(fill_value=0)
net_cat_yr    = gross_cat_yr.subtract(refund_cat_yr, fill_value=0)

print('Annual gross revenue by category (M VND):')
print((gross_cat_yr/1e6).round(1).to_string())
print()
print('Annual refund by category (M VND):')
print((refund_cat_yr/1e6).round(1).to_string())
print()
print('Annual net revenue by category (M VND):')
print((net_cat_yr/1e6).round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

(gross_cat_yr/1e6).plot(kind='bar', ax=axes[0], color=CAT_COLORS[:gross_cat_yr.shape[1]],
                        edgecolor='white', linewidth=0.4)
axes[0].set_title('Annual gross revenue by category (M VND)')
axes[0].set_xlabel('Year'); axes[0].set_ylabel('M VND')
axes[0].legend(title='Category', fontsize=8)
axes[0].tick_params(axis='x', rotation=0)

(net_cat_yr/1e6).plot(kind='bar', ax=axes[1], color=CAT_COLORS[:net_cat_yr.shape[1]],
                      edgecolor='white', linewidth=0.4)
axes[1].set_title('Annual net revenue by category (M VND)')
axes[1].set_xlabel('Year'); axes[1].set_ylabel('M VND')
axes[1].legend(title='Category', fontsize=8)
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.show()"""),

    md("## Summary"),
    code("""\
total_gross  = items_full['revenue_line'].sum()
total_refund = ret_full['refund_amount'].sum()
total_net    = total_gross - total_refund

print('=== Return Analysis Summary ===')
print(f'Total gross revenue  : {total_gross:>18,.0f} VND')
print(f'Total refunds        : {total_refund:>18,.0f} VND  ({total_refund/total_gross*100:.2f}%)')
print(f'Total net revenue    : {total_net:>18,.0f} VND')
print()
print(f'Order return rate    : {returned_orders/total_orders*100:.2f}%  ({returned_orders:,} / {total_orders:,} orders)')
print(f'Qty return rate      : {returned_qty/total_qty*100:.2f}%  ({int(returned_qty):,} / {int(total_qty):,} units)')
print()
print('Top return reason    :', reason_stats.iloc[0]["return_reason"], f'({reason_stats.iloc[0]["count_%"]:.1f}%)')
print()
print('Return rate by category:')
for cat, row in cat_stats.iterrows():
    print(f'  {cat:<12}  qty_rate={row["return_rate_%"]:.2f}%  refund_rate={row["refund_rate_%"]:.2f}%')"""),
])

INV_SETUP = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})

inv   = pd.read_csv('inventory.csv', parse_dates=['snapshot_date'])
prods = pd.read_csv('products.csv')

inv['year']  = inv['snapshot_date'].dt.year
inv['month'] = inv['snapshot_date'].dt.month

CAT_ORDER = ['Streetwear','Outdoor','Casual','GenZ']
CAT_COLORS = ['#4C72B0','#DD8452','#55A868','#C44E52']

print(f'Inventory snapshots : {len(inv):,}')
print(f'Date range          : {inv["snapshot_date"].min().date()} -> {inv["snapshot_date"].max().date()}')
print(f'Unique products     : {inv["product_id"].nunique():,}')
print(f'Unique months       : {inv["snapshot_date"].nunique()}')
inv.head(3)
"""

eda03 = nb([
    md("# EDA-03 · Inventory Health\nStockout frequency, overstock analysis, fill_rate vs stockout_days, days_of_supply heatmap."),
    code(INV_SETUP),

    md("## 1. Overall Flag Distribution"),
    code("""\
total = len(inv)
so_rate  = inv['stockout_flag'].mean()  * 100
ov_rate  = inv['overstock_flag'].mean() * 100
both     = ((inv['stockout_flag']==1) & (inv['overstock_flag']==1)).mean() * 100
neither  = ((inv['stockout_flag']==0) & (inv['overstock_flag']==0)).mean() * 100

print(f'Total snapshots      : {total:,}')
print(f'Stockout flag = 1    : {inv["stockout_flag"].sum():,}  ({so_rate:.1f}%)')
print(f'Overstock flag = 1   : {inv["overstock_flag"].sum():,}  ({ov_rate:.1f}%)')
print(f'Both flags = 1       : {int(both*total/100):,}  ({both:.1f}%)  <- stock-out AND overstock simultaneously')
print(f'Neither flag         : {int(neither*total/100):,}  ({neither:.1f}%)')
print(f'Reorder flag = 1     : {inv["reorder_flag"].sum():,}  ({inv["reorder_flag"].mean()*100:.1f}%)')

labels = ['Stockout only','Overstock only','Both','Neither']
vals = [
    ((inv['stockout_flag']==1) & (inv['overstock_flag']==0)).sum(),
    ((inv['stockout_flag']==0) & (inv['overstock_flag']==1)).sum(),
    ((inv['stockout_flag']==1) & (inv['overstock_flag']==1)).sum(),
    ((inv['stockout_flag']==0) & (inv['overstock_flag']==0)).sum(),
]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, vals, color=['#C44E52','#DD8452','#9467BD','#55A868'])
ax.set_ylabel('Snapshots'); ax.set_title('Inventory flag distribution')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
            f'{v:,}\\n({v/total*100:.1f}%)', ha='center', fontsize=9)
plt.tight_layout(); plt.show()"""),

    md("## 2. Stockout Frequency"),
    code("""\
cat_so = inv.groupby('category').agg(
    snapshots      = ('stockout_flag','count'),
    stockout_snaps = ('stockout_flag','sum'),
    avg_stockout_days = ('stockout_days','mean'),
    pct_months_stockout = ('stockout_flag','mean'),
).reset_index()
cat_so['pct_months_stockout'] *= 100
print('Stockout by category:')
print(cat_so.round(2).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].bar(cat_so['category'], cat_so['pct_months_stockout'],
            color=CAT_COLORS[:len(cat_so)])
axes[0].set_ylabel('% months in stockout'); axes[0].set_title('Stockout rate by category')
for i, v in enumerate(cat_so['pct_months_stockout']):
    axes[0].text(i, v+0.3, f'{v:.1f}%', ha='center', fontsize=9)

axes[1].bar(cat_so['category'], cat_so['avg_stockout_days'],
            color=CAT_COLORS[:len(cat_so)])
axes[1].set_ylabel('Avg stockout days/month'); axes[1].set_title('Avg stockout days by category')
for i, v in enumerate(cat_so['avg_stockout_days']):
    axes[1].text(i, v+0.02, f'{v:.2f}', ha='center', fontsize=9)
plt.tight_layout(); plt.show()"""),

    md("## 3. Stockout Trend by Year & Month"),
    code("""\
yr_so = inv.groupby('year').agg(
    stockout_rate = ('stockout_flag','mean'),
    avg_days      = ('stockout_days','mean'),
).reset_index()
yr_so['stockout_rate'] *= 100

mo_so = inv.groupby('month').agg(
    stockout_rate = ('stockout_flag','mean'),
    avg_days      = ('stockout_days','mean'),
).reset_index()
mo_so['stockout_rate'] *= 100
MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(yr_so['year'], yr_so['stockout_rate'], marker='o', color='#C44E52')
axes[0].fill_between(yr_so['year'], yr_so['stockout_rate'], alpha=0.15, color='#C44E52')
axes[0].set_xlabel('Year'); axes[0].set_ylabel('Stockout rate %')
axes[0].set_title('Stockout rate trend by year')
for _, row in yr_so.iterrows():
    axes[0].text(row['year'], row['stockout_rate']+0.3, f'{row["stockout_rate"]:.1f}%', ha='center', fontsize=8)

axes[1].bar(range(1,13), mo_so['stockout_rate'], color='#C44E52', alpha=0.8)
axes[1].set_xticks(range(1,13)); axes[1].set_xticklabels(MONTH_LABELS)
axes[1].set_ylabel('Stockout rate %'); axes[1].set_title('Stockout rate by calendar month (avg all years)')
for i, v in enumerate(mo_so['stockout_rate']):
    axes[1].text(i+1, v+0.3, f'{v:.1f}%', ha='center', fontsize=7)
plt.tight_layout(); plt.show()"""),

    md("## 4. Top Products by Stockout Frequency"),
    code("""\
prod_so = inv.groupby(['product_id','product_name','category','segment']).agg(
    n_months          = ('stockout_flag','count'),
    months_stockout   = ('stockout_flag','sum'),
    total_stockout_days = ('stockout_days','sum'),
    avg_fill_rate     = ('fill_rate','mean'),
).reset_index()
prod_so['stockout_rate_%'] = prod_so['months_stockout'] / prod_so['n_months'] * 100

top20_so = prod_so.sort_values('months_stockout', ascending=False).head(20)
print('Top 20 products by stockout months:')
print(top20_so[['product_name','category','segment','months_stockout','stockout_rate_%',
                'total_stockout_days','avg_fill_rate']].to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(top20_so['product_name'].str[:25], top20_so['months_stockout'],
        color=[CAT_COLORS[CAT_ORDER.index(c)] if c in CAT_ORDER else 'grey'
               for c in top20_so['category']])
ax.set_xlabel('Months in stockout'); ax.set_title('Top 20 products: stockout months')
ax.invert_yaxis()
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=CAT_COLORS[i], label=CAT_ORDER[i]) for i in range(len(CAT_ORDER))]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
plt.tight_layout(); plt.show()"""),

    md("## 5. Overstock Analysis"),
    code("""\
cat_ov = inv.groupby('category').agg(
    overstock_rate   = ('overstock_flag','mean'),
    avg_stock_on_hand = ('stock_on_hand','mean'),
    avg_units_sold   = ('units_sold','mean'),
    avg_sell_through = ('sell_through_rate','mean'),
).reset_index()
cat_ov['overstock_rate'] *= 100
print('Overstock by category:')
print(cat_ov.round(3).to_string(index=False))

prod_ov = inv.groupby(['product_id','product_name','category','segment']).agg(
    n_months           = ('overstock_flag','count'),
    months_overstock   = ('overstock_flag','sum'),
    avg_stock_on_hand  = ('stock_on_hand','mean'),
    avg_sell_through   = ('sell_through_rate','mean'),
    avg_days_of_supply = ('days_of_supply','mean'),
).reset_index()
prod_ov['overstock_rate_%'] = prod_ov['months_overstock'] / prod_ov['n_months'] * 100

top20_ov = prod_ov.sort_values('months_overstock', ascending=False).head(20)
print('\\nTop 20 products by overstock months:')
print(top20_ov[['product_name','category','segment','months_overstock',
                'overstock_rate_%','avg_stock_on_hand','avg_sell_through']].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(cat_ov['category'], cat_ov['overstock_rate'],
            color=CAT_COLORS[:len(cat_ov)])
axes[0].set_ylabel('% months overstock'); axes[0].set_title('Overstock rate by category')
for i, v in enumerate(cat_ov['overstock_rate']):
    axes[0].text(i, v+0.3, f'{v:.1f}%', ha='center', fontsize=9)

axes[1].bar(cat_ov['category'], cat_ov['avg_sell_through'],
            color=CAT_COLORS[:len(cat_ov)])
axes[1].set_ylabel('Avg sell-through rate'); axes[1].set_title('Avg sell-through rate by category')
for i, v in enumerate(cat_ov['avg_sell_through']):
    axes[1].text(i, v+0.002, f'{v:.3f}', ha='center', fontsize=9)
plt.tight_layout(); plt.show()"""),

    md("## 6. Fill Rate vs Stockout Days"),
    code("""\
print('fill_rate summary:')
print(inv['fill_rate'].describe().round(4).to_string())
print('\\nstockout_days summary:')
print(inv['stockout_days'].describe().round(2).to_string())

r = inv['fill_rate'].corr(inv['stockout_days'])
print(f'\\nCorrelation fill_rate vs stockout_days: r = {r:.4f}')

sample = inv.sample(min(5000, len(inv)), random_state=42)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for cat, grp in sample.groupby('category'):
    idx = CAT_ORDER.index(cat) if cat in CAT_ORDER else -1
    col = CAT_COLORS[idx] if idx >= 0 else 'grey'
    axes[0].scatter(grp['stockout_days'], grp['fill_rate'],
                    alpha=0.3, s=8, color=col, label=cat)
axes[0].set_xlabel('Stockout days'); axes[0].set_ylabel('Fill rate')
axes[0].set_title(f'Fill rate vs Stockout days (r={r:.3f})')
axes[0].legend(fontsize=7, markerscale=2)

axes[1].hist(inv['fill_rate'], bins=50, color='#4C72B0', edgecolor='white', linewidth=0.3)
axes[1].set_xlabel('Fill rate'); axes[1].set_ylabel('Count')
axes[1].set_title('Fill rate distribution')

so_nonzero = inv[inv['stockout_days'] > 0]['stockout_days']
axes[2].hist(so_nonzero, bins=range(0, int(so_nonzero.max())+2),
             color='#C44E52', edgecolor='white', linewidth=0.3)
axes[2].set_xlabel('Stockout days'); axes[2].set_ylabel('Count')
axes[2].set_title('Stockout days distribution (excl 0)')
plt.tight_layout(); plt.show()"""),

    md("## 7. Fill Rate by Category & Year"),
    code("""\
fr_cat_yr = inv.groupby(['year','category'])['fill_rate'].mean().unstack()
print('Avg fill_rate by category × year:')
print(fr_cat_yr.round(4).to_string())

fig, ax = plt.subplots(figsize=(11, 4))
for i, cat in enumerate(fr_cat_yr.columns):
    ax.plot(fr_cat_yr.index, fr_cat_yr[cat], marker='o',
            label=cat, color=CAT_COLORS[i % len(CAT_COLORS)])
ax.set_xlabel('Year'); ax.set_ylabel('Avg fill rate')
ax.set_title('Fill rate trend by category')
ax.legend(title='Category'); ax.set_ylim(0.9, 1.01)
plt.tight_layout(); plt.show()"""),

    md("## 8. Days of Supply — Heatmap (Category × Month)"),
    code("""\
dos_heat = inv.groupby(['category','month'])['days_of_supply'].mean().unstack()
print('Avg days_of_supply (category × calendar month):')
print(dos_heat.round(1).to_string())

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(dos_heat.values, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_LABELS)
ax.set_yticks(range(len(dos_heat.index))); ax.set_yticklabels(dos_heat.index)
ax.set_title('Avg days of supply: category × month')
plt.colorbar(im, ax=ax, label='Days of supply')
for i in range(dos_heat.shape[0]):
    for j in range(dos_heat.shape[1]):
        v = dos_heat.values[i, j]
        ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=9,
                color='white' if v < dos_heat.values.min() + (dos_heat.values.max()-dos_heat.values.min())*0.35 else 'black')
plt.tight_layout(); plt.show()"""),

    md("## 9. Days of Supply — Heatmap (Year × Month)"),
    code("""\
dos_yr_mo = inv.groupby(['year','month'])['days_of_supply'].mean().unstack()
print('Avg days_of_supply (year × month):')
print(dos_yr_mo.round(1).to_string())

fig, ax = plt.subplots(figsize=(13, 5))
im = ax.imshow(dos_yr_mo.values, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(12)); ax.set_xticklabels(MONTH_LABELS)
ax.set_yticks(range(len(dos_yr_mo.index))); ax.set_yticklabels(dos_yr_mo.index)
ax.set_title('Avg days of supply: year × month')
plt.colorbar(im, ax=ax, label='Days of supply')
for i in range(dos_yr_mo.shape[0]):
    for j in range(dos_yr_mo.shape[1]):
        v = dos_yr_mo.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=8,
                    color='white' if v < dos_yr_mo.values[~np.isnan(dos_yr_mo.values)].min()
                                   + (dos_yr_mo.values[~np.isnan(dos_yr_mo.values)].max()
                                      - dos_yr_mo.values[~np.isnan(dos_yr_mo.values)].min())*0.35
                    else 'black')
plt.tight_layout(); plt.show()"""),

    md("## 10. Simultaneous Stockout & Overstock — Paradox Check"),
    code("""\
paradox = inv[(inv['stockout_flag']==1) & (inv['overstock_flag']==1)].copy()
print(f'Snapshots with BOTH stockout & overstock flags: {len(paradox):,} ({len(paradox)/len(inv)*100:.1f}%)')
print()
print('Stats for paradox snapshots:')
print(paradox[['stock_on_hand','units_sold','stockout_days','days_of_supply',
               'fill_rate','sell_through_rate']].describe().round(3).to_string())

print('\\nParadox by category:')
print(paradox.groupby('category')[['stock_on_hand','stockout_days','days_of_supply','fill_rate']].mean().round(3).to_string())"""),

    md("## Summary"),
    code("""\
print('=== Inventory Health Summary ===')
print(f'Stockout flag rate   : {inv["stockout_flag"].mean()*100:.1f}%  ({inv["stockout_flag"].sum():,} / {len(inv):,} snapshots)')
print(f'Overstock flag rate  : {inv["overstock_flag"].mean()*100:.1f}%  ({inv["overstock_flag"].sum():,} / {len(inv):,} snapshots)')
print(f'Both flags           : {((inv["stockout_flag"]==1)&(inv["overstock_flag"]==1)).sum():,}  ({((inv["stockout_flag"]==1)&(inv["overstock_flag"]==1)).mean()*100:.1f}%)')
print(f'Avg fill_rate        : {inv["fill_rate"].mean():.4f}')
print(f'Avg stockout_days    : {inv["stockout_days"].mean():.2f} days/month')
print(f'Avg days_of_supply   : {inv["days_of_supply"].mean():.1f} days')
print()

worst_so = prod_so.sort_values('months_stockout', ascending=False).iloc[0]
worst_ov = prod_ov.sort_values('months_overstock', ascending=False).iloc[0]
print(f'Most stockout product: {worst_so["product_name"]}  ({worst_so["months_stockout"]} months, {worst_so["stockout_rate_%"]:.1f}%)')
print(f'Most overstock product: {worst_ov["product_name"]}  ({worst_ov["months_overstock"]} months, {worst_ov["overstock_rate_%"]:.1f}%)')
print()
print('Fill rate by category:')
print(inv.groupby('category')['fill_rate'].mean().round(4).to_string())
print()
print('Days of supply by category:')
print(inv.groupby('category')['days_of_supply'].agg(['mean','min','max']).round(1).to_string())"""),
])

WEB_SETUP = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})

MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
SRC_COLORS   = {'organic_search':'#4C72B0','paid_search':'#DD8452','social_media':'#55A868',
                 'email_campaign':'#C44E52','referral':'#9467BD','direct':'#8C564B'}

web    = pd.read_csv('web_traffic.csv',  parse_dates=['date'])
orders = pd.read_csv('orders.csv',       parse_dates=['order_date'])
items  = pd.read_csv('order_items.csv',  low_memory=False)
sales  = pd.read_csv('sales.csv',        parse_dates=['Date'])

daily_orders = orders.groupby('order_date').agg(
    n_orders = ('order_id','count')).reset_index().rename(columns={'order_date':'date'})

daily_rev = (
    items.merge(orders[['order_id','order_date']], on='order_id', how='left')
    .groupby('order_date')
    .agg(revenue=('unit_price', lambda x: (x * items.loc[x.index,'quantity']).sum()))
    .reset_index().rename(columns={'order_date':'date'})
)

daily_src_orders = (
    orders.groupby(['order_date','order_source'])['order_id']
    .count().reset_index()
    .rename(columns={'order_date':'date','order_id':'n_orders','order_source':'traffic_source'})
)

df = web.merge(daily_orders, on='date', how='left')
df['year']  = df['date'].dt.year
df['month'] = df['date'].dt.month
df['conv_rate'] = df['n_orders'] / df['sessions']

df_src = web.merge(daily_src_orders, on=['date','traffic_source'], how='left')
df_src['conv_rate'] = df_src['n_orders'] / df_src['sessions']
df_src['year']      = df_src['date'].dt.year
df_src['month']     = df_src['date'].dt.month

print(f'web_traffic rows  : {len(web):,}  ({web["date"].min().date()} -> {web["date"].max().date()})')
print(f'Overlap with orders: {df["n_orders"].notna().sum():,} days')
print(f'Traffic sources    : {web["traffic_source"].unique().tolist()}')
df.head(3)
"""

eda04 = nb([
    md("# EDA-04 · Web Traffic vs Revenue\nSessions→orders funnel, bounce_rate trend, traffic_source contribution, seasonality."),
    code(WEB_SETUP),

    md("## 1. Overview: Web Traffic & Orders"),
    code("""\
print('=== Web Traffic Summary ===')
print(web[['sessions','unique_visitors','page_views','bounce_rate','avg_session_duration_sec']].describe().round(2).to_string())

print()
print(f'Total sessions (2013-2022)       : {web["sessions"].sum():,.0f}')
print(f'Total unique_visitors            : {web["unique_visitors"].sum():,.0f}')
print(f'Avg daily sessions               : {web["sessions"].mean():,.0f}')
print(f'Avg daily orders (overlap period): {df["n_orders"].mean():.1f}')
print(f'Overall conversion rate          : {df["n_orders"].sum()/df["sessions"].sum()*100:.4f}%')"""),

    md("## 2. Sessions → Orders Funnel"),
    code("""\
funnel = df_src.groupby('traffic_source').agg(
    total_sessions  = ('sessions','sum'),
    total_visitors  = ('unique_visitors','sum'),
    total_orders    = ('n_orders','sum'),
).fillna(0).reset_index()
funnel['conv_rate_%']      = funnel['total_orders'] / funnel['total_sessions'] * 100
funnel['visitor_rate_%']   = funnel['total_visitors'] / funnel['total_sessions'] * 100
funnel = funnel.sort_values('conv_rate_%', ascending=False)
print('Funnel by traffic source:')
print(funnel.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

colors = [SRC_COLORS.get(s,'grey') for s in funnel['traffic_source']]
bars = axes[0].barh(funnel['traffic_source'], funnel['conv_rate_%'], color=colors)
axes[0].set_xlabel('Conversion rate %  (orders / sessions)')
axes[0].set_title('Conversion rate by traffic source')
for bar, v in zip(bars, funnel['conv_rate_%']):
    axes[0].text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                 f'{v:.4f}%', va='center', fontsize=8)

x = range(len(funnel))
w = 0.25
axes[1].bar([i-w for i in x], funnel['total_sessions']/1e6,  width=w, label='Sessions (M)',  color='#4C72B0', alpha=0.85)
axes[1].bar([i   for i in x], funnel['total_visitors']/1e6,  width=w, label='Visitors (M)',  color='#55A868', alpha=0.85)
axes[1].bar([i+w for i in x], funnel['total_orders']/1e3,    width=w, label='Orders (K)',    color='#C44E52', alpha=0.85)
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(funnel['traffic_source'], rotation=20, ha='right', fontsize=8)
axes[1].set_ylabel('Volume (M/K)')
axes[1].set_title('Sessions vs Visitors vs Orders by source')
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()"""),

    md("## 3. Conversion Rate Trend by Year"),
    code("""\
yr_conv = df.groupby('year').agg(
    sessions = ('sessions','sum'),
    orders   = ('n_orders','sum'),
).reset_index()
yr_conv['conv_rate_%'] = yr_conv['orders'] / yr_conv['sessions'] * 100

yr_src = df_src.groupby(['year','traffic_source']).agg(
    sessions = ('sessions','sum'),
    orders   = ('n_orders','sum'),
).reset_index().fillna(0)
yr_src['conv_rate_%'] = yr_src['orders'] / yr_src['sessions'] * 100

print('Overall conversion rate by year:')
print(yr_conv[['year','sessions','orders','conv_rate_%']].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(yr_conv['year'], yr_conv['conv_rate_%'], marker='o', color='#4C72B0', lw=2)
axes[0].fill_between(yr_conv['year'], yr_conv['conv_rate_%'], alpha=0.15, color='#4C72B0')
axes[0].set_xlabel('Year'); axes[0].set_ylabel('Conversion rate %')
axes[0].set_title('Overall conversion rate by year')
for _, row in yr_conv.iterrows():
    axes[0].text(row['year'], row['conv_rate_%']+0.001, f'{row["conv_rate_%"]:.4f}%', ha='center', fontsize=7)

for src, grp in yr_src.groupby('traffic_source'):
    axes[1].plot(grp['year'], grp['conv_rate_%'], marker='o', ms=4,
                 label=src, color=SRC_COLORS.get(src,'grey'), lw=1.5)
axes[1].set_xlabel('Year'); axes[1].set_ylabel('Conversion rate %')
axes[1].set_title('Conversion rate by source & year')
axes[1].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.show()"""),

    md("## 4. Bounce Rate Trend"),
    code("""\
df['month_period'] = df['date'].dt.to_period('M')
monthly_bounce = df.groupby('month_period').agg(
    avg_bounce = ('bounce_rate','mean'),
    avg_sessions = ('sessions','mean'),
    avg_orders = ('n_orders','mean'),
).reset_index()
monthly_bounce['month_ts'] = monthly_bounce['month_period'].dt.to_timestamp()

src_bounce = web.groupby('traffic_source')['bounce_rate'].agg(['mean','std','min','max']).round(6)
print('Bounce rate by traffic source:')
print(src_bounce.to_string())

r_bounce_orders = df['bounce_rate'].corr(df['n_orders'])
r_bounce_conv   = df['bounce_rate'].corr(df['conv_rate'])
print(f'\\nCorr(bounce_rate, n_orders)   : {r_bounce_orders:.4f}')
print(f'Corr(bounce_rate, conv_rate)  : {r_bounce_conv:.4f}')

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

axes[0].plot(monthly_bounce['month_ts'], monthly_bounce['avg_bounce']*100,
             color='#C44E52', lw=1.2)
axes[0].set_ylabel('Avg bounce rate %'); axes[0].set_title('Monthly bounce rate trend')

axes[1].plot(monthly_bounce['month_ts'], monthly_bounce['avg_sessions'],
             color='#4C72B0', lw=1.2, label='Sessions')
ax2 = axes[1].twinx()
ax2.plot(monthly_bounce['month_ts'], monthly_bounce['avg_orders'],
         color='#55A868', lw=1.2, linestyle='--', label='Orders')
axes[1].set_ylabel('Avg sessions'); ax2.set_ylabel('Avg orders')
axes[1].set_title('Monthly sessions vs orders')
lines1, lbl1 = axes[1].get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1+lines2, lbl1+lbl2, loc='upper left', fontsize=8)
plt.tight_layout(); plt.show()

fig2, ax3 = plt.subplots(figsize=(9, 4))
for src, grp in web.groupby('traffic_source'):
    mo = grp.groupby(grp['date'].dt.to_period('M'))['bounce_rate'].mean().reset_index()
    mo['ts'] = mo['date'].dt.to_timestamp()
    ax3.plot(mo['ts'], mo['bounce_rate']*100, lw=1, alpha=0.8,
             label=src, color=SRC_COLORS.get(src,'grey'))
ax3.set_ylabel('Bounce rate %'); ax3.set_title('Bounce rate by traffic source over time')
ax3.legend(fontsize=8); plt.tight_layout(); plt.show()"""),

    md("## 5. Traffic Source Contribution (Sessions vs Revenue)"),
    code("""\
src_sess = web.groupby('traffic_source')['sessions'].sum()
src_sess_pct = src_sess / src_sess.sum() * 100

src_ord  = orders.groupby('order_source')['order_id'].count().rename(index=lambda x: x)
src_ord_pct = src_ord / src_ord.sum() * 100

rev_by_src = (
    items.merge(orders[['order_id','order_source']], on='order_id', how='left')
    .assign(rev_line=lambda d: d['quantity']*d['unit_price'])
    .groupby('order_source')['rev_line'].sum()
)
rev_by_src_pct = rev_by_src / rev_by_src.sum() * 100

contrib = pd.DataFrame({
    'sessions_%' : src_sess_pct,
    'orders_%'   : src_ord_pct,
    'revenue_%'  : rev_by_src_pct,
}).fillna(0).sort_values('revenue_%', ascending=False)
print('Traffic source contribution:')
print(contrib.round(2).to_string())

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title in zip(axes,
    ['sessions_%','orders_%','revenue_%'],
    ['Sessions share %','Orders share %','Revenue share %']):
    colors = [SRC_COLORS.get(s,'grey') for s in contrib.index]
    wedges, texts, autotexts = ax.pie(
        contrib[col], labels=contrib.index, autopct='%1.1f%%',
        colors=colors, startangle=140,
        textprops={'fontsize':7})
    ax.set_title(title)
plt.tight_layout(); plt.show()"""),

    md("## 6. Source Share Trend Over Time"),
    code("""\
yr_src_sess = (
    web.groupby(['year','traffic_source'])['sessions'].sum()
    .unstack(fill_value=0)
)
yr_src_sess_pct = yr_src_sess.div(yr_src_sess.sum(axis=1), axis=0) * 100

orders['year'] = orders['order_date'].dt.year
yr_src_ord = (
    orders.groupby(['year','order_source'])['order_id'].count()
    .unstack(fill_value=0)
)
yr_src_ord_pct = yr_src_ord.div(yr_src_ord.sum(axis=1), axis=0) * 100

print('Yearly sessions share %:')
print(yr_src_sess_pct.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

src_list = yr_src_sess_pct.columns.tolist()
colors_list = [SRC_COLORS.get(s,'grey') for s in src_list]

yr_src_sess_pct.plot.area(ax=axes[0], color=colors_list, alpha=0.8, linewidth=0)
axes[0].set_xlabel('Year'); axes[0].set_ylabel('Share %')
axes[0].set_title('Web sessions share by source (stacked area)')
axes[0].legend(fontsize=7, loc='upper left')

src_list2 = yr_src_ord_pct.columns.tolist()
colors_list2 = [SRC_COLORS.get(s,'grey') for s in src_list2]
yr_src_ord_pct.plot.area(ax=axes[1], color=colors_list2, alpha=0.8, linewidth=0)
axes[1].set_xlabel('Year'); axes[1].set_ylabel('Share %')
axes[1].set_title('Orders share by source (stacked area)')
axes[1].legend(fontsize=7, loc='upper left')
plt.tight_layout(); plt.show()"""),

    md("## 7. Seasonality — Web Traffic vs Sales Revenue"),
    code("""\
df['month'] = df['date'].dt.month
sales['month'] = sales['Date'].dt.month

monthly_web = df.groupby('month').agg(
    avg_sessions  = ('sessions','mean'),
    avg_visitors  = ('unique_visitors','mean'),
    avg_orders    = ('n_orders','mean'),
    avg_bounce    = ('bounce_rate','mean'),
    avg_dur       = ('avg_session_duration_sec','mean'),
).reset_index()

monthly_sales = sales.groupby('month').agg(
    avg_revenue = ('Revenue','mean'),
    avg_cogs    = ('COGS','mean'),
).reset_index()

sea = monthly_web.merge(monthly_sales, on='month')

def norm(s): return (s - s.min()) / (s.max() - s.min())

print('Monthly seasonal profile (normalized):')
sea_show = sea[['month','avg_sessions','avg_orders','avg_revenue']].copy()
for c in ['avg_sessions','avg_orders','avg_revenue']:
    sea_show[c+'_norm'] = norm(sea[c])
print(sea_show.round(3).to_string(index=False))

print('\\nPearson correlations (12 monthly avg points):')
for col in ['avg_sessions','avg_visitors','avg_bounce','avg_dur','avg_orders']:
    r = sea[col].corr(sea['avg_revenue'])
    print(f'  {col:<30}: r = {r:.4f}')

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

for metric, color, label in [
    ('avg_sessions','#4C72B0','Sessions'),
    ('avg_orders',  '#55A868','Orders'),
]:
    axes[0].plot(sea['month'], norm(sea[metric]), marker='o', label=label, color=color)
axes[0].plot(sea['month'], norm(sea['avg_revenue']), marker='s', lw=2,
             label='Revenue', color='#C44E52', linestyle='--')
axes[0].set_ylabel('Normalized value (0–1)')
axes[0].set_title('Seasonal pattern: Sessions / Orders / Revenue (normalized)')
axes[0].set_xticks(range(1,13)); axes[0].set_xticklabels(MONTH_LABELS)
axes[0].legend(fontsize=9)

axes[1].plot(sea['month'], sea['avg_bounce']*100, marker='o', color='#9467BD', label='Bounce rate %')
ax2b = axes[1].twinx()
ax2b.plot(sea['month'], sea['avg_dur'], marker='s', color='#8C564B',
          linestyle='--', label='Avg session duration (s)')
axes[1].set_ylabel('Bounce rate %'); ax2b.set_ylabel('Duration (s)')
axes[1].set_title('Seasonal bounce rate & session duration')
axes[1].set_xticks(range(1,13)); axes[1].set_xticklabels(MONTH_LABELS)
l1,lb1 = axes[1].get_legend_handles_labels()
l2,lb2 = ax2b.get_legend_handles_labels()
axes[1].legend(l1+l2, lb1+lb2, fontsize=8)
plt.tight_layout(); plt.show()"""),

    md("## 8. Year-over-Year: Sessions Growth vs Revenue Growth"),
    code("""\
yr_web = web.groupby('year').agg(
    total_sessions  = ('sessions','sum'),
    total_visitors  = ('unique_visitors','sum'),
    avg_bounce      = ('bounce_rate','mean'),
    avg_duration    = ('avg_session_duration_sec','mean'),
).reset_index()

yr_sales = sales.groupby(sales['Date'].dt.year)[['Revenue','COGS']].sum().reset_index()
yr_sales.columns = ['year','Revenue','COGS']

yr = yr_web.merge(yr_sales, on='year', how='inner')
yr['yoy_sessions_%'] = yr['total_sessions'].pct_change() * 100
yr['yoy_revenue_%']  = yr['Revenue'].pct_change() * 100

print('YoY growth — sessions vs revenue:')
print(yr[['year','total_sessions','Revenue','yoy_sessions_%','yoy_revenue_%']].round(1).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ax_r = axes[0].twinx() if True else None
axes[0].bar(yr['year']-0.2, yr['total_sessions']/1e6, width=0.4,
            label='Sessions (M)', color='#4C72B0', alpha=0.8)
ax_r = axes[0].twinx()
ax_r.bar(yr['year']+0.2, yr['Revenue']/1e9, width=0.4,
         label='Revenue (B VND)', color='#C44E52', alpha=0.8)
axes[0].set_ylabel('Sessions (M)'); ax_r.set_ylabel('Revenue (B VND)')
axes[0].set_title('Annual sessions vs revenue')
axes[0].set_xlabel('Year')
h1,l1 = axes[0].get_legend_handles_labels()
h2,l2 = ax_r.get_legend_handles_labels()
axes[0].legend(h1+h2, l1+l2, fontsize=8, loc='upper left')

yr2 = yr.dropna(subset=['yoy_sessions_%','yoy_revenue_%'])
axes[1].plot(yr2['year'], yr2['yoy_sessions_%'], marker='o', label='Sessions YoY %', color='#4C72B0')
axes[1].plot(yr2['year'], yr2['yoy_revenue_%'],  marker='s', label='Revenue YoY %',  color='#C44E52', linestyle='--')
axes[1].axhline(0, color='black', lw=0.8, linestyle=':')
axes[1].set_xlabel('Year'); axes[1].set_ylabel('YoY growth %')
axes[1].set_title('YoY growth: sessions vs revenue')
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()"""),

    md("## 9. Daily Scatter: Sessions vs Revenue"),
    code("""\
df_rev = df.merge(sales.rename(columns={'Date':'date'}), on='date', how='inner')
df_rev['year'] = df_rev['date'].dt.year

r_daily = df_rev['sessions'].corr(df_rev['Revenue'])
print(f'Daily Pearson correlation sessions vs Revenue: r = {r_daily:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

sc = axes[0].scatter(df_rev['sessions']/1e3, df_rev['Revenue']/1e6,
                     c=df_rev['year'], cmap='viridis', alpha=0.3, s=8)
plt.colorbar(sc, ax=axes[0], label='Year')
axes[0].set_xlabel('Daily sessions (K)'); axes[0].set_ylabel('Daily Revenue (M VND)')
axes[0].set_title(f'Daily sessions vs Revenue  (r={r_daily:.3f})')

monthly_both = df_rev.groupby(df_rev['date'].dt.to_period('M')).agg(
    sessions=('sessions','sum'), Revenue=('Revenue','sum')).reset_index()
r_mo = monthly_both['sessions'].corr(monthly_both['Revenue'])
axes[1].scatter(monthly_both['sessions']/1e6, monthly_both['Revenue']/1e6,
                alpha=0.5, s=25, color='#4C72B0')
axes[1].set_xlabel('Monthly sessions (M)'); axes[1].set_ylabel('Monthly Revenue (M VND)')
axes[1].set_title(f'Monthly sessions vs Revenue  (r={r_mo:.3f})')
plt.tight_layout(); plt.show()"""),

    md("## Summary"),
    code("""\
print('=== Web Traffic vs Revenue Summary ===')
print(f'Period          : {web["date"].min().date()} -> {web["date"].max().date()}')
print(f'Avg sessions/day: {web["sessions"].mean():,.0f}')
print(f'Overall conv.   : {df["n_orders"].sum()/df["sessions"].sum()*100:.4f}%  (orders/sessions)')
print()
print('Conversion rate by source:')
for _, row in funnel.iterrows():
    print(f'  {row["traffic_source"]:<18}: {row["conv_rate_%"]:.4f}%')
print()
print(f'Seasonal corr sessions-revenue (monthly): r = {sea["avg_sessions"].corr(sea["avg_revenue"]):.4f}')
print(f'Daily   corr sessions-revenue           : r = {r_daily:.4f}')
print()
print('Sessions share vs Revenue share (top 3 sources):')
for src in contrib.head(3).index:
    print(f'  {src:<18}: sessions={contrib.loc[src,"sessions_%"]:.1f}%  revenue={contrib.loc[src,"revenue_%"]:.1f}%')"""),
])

notebooks = [
    ('eda_01_catalog_profile.ipynb', eda01),
    ('eda_02_return_analysis.ipynb', eda02),
    ('eda_03_inventory_health.ipynb', eda03),
    ('eda_04_web_traffic_revenue.ipynb', eda04),
    ('dq_01_customers.ipynb',   nb01),
    ('dq_02_orders.ipynb',      nb02),
    ('dq_03_order_items.ipynb', nb03),
    ('dq_04_payments.ipynb',    nb04),
    ('dq_05_products.ipynb',    nb05),
    ('dq_06_promotions.ipynb',  nb06),
    ('dq_07_returns.ipynb',     nb07),
    ('dq_08_reviews.ipynb',     nb08),
    ('dq_09_shipments.ipynb',   nb09),
    ('dq_10_inventory.ipynb',   nb10),
    ('dq_11_geography.ipynb',   nb11),
]

for fname, notebook in notebooks:
    with open(fname, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    print(f'Created: {fname}  ({len(notebook.cells)} cells)')

print('\nDone.')

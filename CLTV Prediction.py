### BGNBD & GG ile CLTV Tahmini ###

# !pip install lifetimes

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
pd.set_option('display.float_format',lambda x:'%.4f' %x)

from sklearn.preprocessing import MinMaxScaler

today_date = dt.datetime(2011,12,11)

df = pd.read_excel("online_retail_II.xlsx",sheet_name="Year 2010-2011")


### GÖREV 1 ###

# Veri Ön İşleme

df.dropna(inplace = True)
df = df[~df['Invoice'].str.contains("C",na = False)]
df = df[df['Quantity']>0]
df = df[df['Price']>0]
df['TotalPrice'] = df['Quantity'] * df['Price']

# United Kingdom müşterilerini seçme
print(df)
print("\n")
df = df[df['Country'].str.contains('United Kingdom',na = False)]

### Lifetime Veri Yapısının Hazırlanması ###

"""
recency: Son satın alma üzerinden geçen zaman. Haftalık.
T: Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış. Haftalık.
frequency : Tekrar eden toplam satın alma sayısı (frequency > 1).
monetary_value: Satın alma başına ortalama kazanç.
"""

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate':[lambda date:(date.max() - date.min()).days,
                                                        lambda date:(today_date-date.min()).days],
                                      'Invoice':lambda Invoice:Invoice.nunique(),
                                      'TotalPrice':lambda TotalPrice:TotalPrice.sum()})



cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# Satın alma başına ortalama kazanç

cltv_df['monetary'] = cltv_df['monetary'] / cltv_df['frequency']

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df = cltv_df[(cltv_df['monetary'] > 0)]

# Recency ve T'yi haftalık değere çevirme

cltv_df['recency'] = cltv_df['recency'] / 7
cltv_df['T'] = cltv_df['T'] / 7

cltv_df['frequency'] = cltv_df['frequency'].astype(int)


# BG/NBD Modelinin Kurulması

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])

# GAMMA-GAMMA Modelinin Kurulması

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

# BG/NBD ve GAMMA-GAMMA Modeli ile CLTV'nin Hesaplanması

cltv_altiay = ggf.customer_lifetime_value(bgf,
                                          cltv_df['frequency'],
                                          cltv_df['recency'],
                                          cltv_df['T'],
                                          cltv_df['monetary'],
                                          time = 6,
                                          freq = "W",
                                          discount_rate=0.01)
print(cltv_altiay.head())
print("\n")
cltv_altiay = cltv_altiay.reset_index()

print(cltv_altiay.sort_values(by="clv", ascending=False).head(20))

# cltv ve cltv_df merge ile birleştirme

cltv_altiay_final = cltv_df.merge(cltv_altiay, on ='Customer ID', how ='left')

print(cltv_altiay_final.head())

print("\n")
## Değerleri scale ederek daha kolay anlaşılabilir hale getirme

#Scale nesnesi yaratma

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(cltv_altiay_final[["clv"]])

cltv_altiay_final['scaled_clv'] = scaler.transform(cltv_altiay_final[["clv"]])

print("2010 - 2011 UK Müşterileri için 6 aylık CLTV Prediction")
print("-------------------------------------------------------")
print(cltv_altiay_final.sort_values(by="scaled_clv", ascending= False).head())
print("\n")


### GÖREV 2 ###

# 1 aylık CLTV tahmini
cltv_biray = ggf.customer_lifetime_value(bgf,
                                          cltv_df['frequency'],
                                          cltv_df['recency'],
                                          cltv_df['T'],
                                          cltv_df['monetary'],
                                          time = 1,
                                          freq = "W",
                                          discount_rate=0.01)


print(cltv_biray.head())
print("\n")
cltv_biray = cltv_biray.reset_index()

# cltv ve cltv_df merge ile birleştirme

cltv_biray_final = cltv_df.merge(cltv_biray, on ='Customer ID', how ='left')

print(cltv_biray_final.head())

print("\n")
## Değerleri scale ederek daha kolay anlaşılabilir hale getirme

#Scale nesnesi yaratma

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(cltv_biray_final[["clv"]])

cltv_biray_final['scaled_clv'] = scaler.transform(cltv_biray_final[["clv"]])

print("2010 - 2011 UK Müşterileri için 1 aylık CLTV Prediction")
print("-------------------------------------------------------")
print(cltv_biray_final.sort_values(by="scaled_clv", ascending= False).head(10))
print("\n")


# 12 aylık CLTV tahmini


cltv_onikiay = ggf.customer_lifetime_value(bgf,
                                          cltv_df['frequency'],
                                          cltv_df['recency'],
                                          cltv_df['T'],
                                          cltv_df['monetary'],
                                          time = 12,
                                          freq = "W",
                                          discount_rate=0.01)

print("\n")
cltv_onikiay = cltv_onikiay.reset_index()

# cltv ve cltv_df merge ile birleştirme

cltv_onikiay_final = cltv_df.merge(cltv_onikiay, on ='Customer ID', how ='left')


print("\n")

## Değerleri scale ederek daha kolay anlaşılabilir hale getirme

#Scale nesnesi yaratma
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(cltv_onikiay_final[["clv"]])

cltv_onikiay_final['scaled_clv'] = scaler.transform(cltv_onikiay_final[["clv"]])

print("2010 - 2011 UK Müşterileri için 12 aylık CLTV Prediction")
print("-------------------------------------------------------")
print(cltv_onikiay_final.sort_values(by="scaled_clv", ascending= False).head(10))
print("\n")

"""
1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık CLTV'de en yüksek olan 10 kişi aynı kişilerdir.
Bu da bu müşterilerin şirketin sadık ve daimi müşterileri olduğunu gösterir.
"""

### GÖREV 3 ###

#Scale edilmiş değerleri 4 segmente ayırma ve grup isimlerini veri setine ekleme
cltv_altiay_final["segment"] = pd.qcut(cltv_altiay_final['scaled_clv'],4,labels=["D","C","B","A"])

print("2010 - 2011 UK Müşterileri için 6 aylık CLTV Prediction Değerleri ve Segmentleri")
print("--------------------------------------------------------------------------------")
print(cltv_altiay_final)

"""
A ve B Segmentindeki müşteriler sadık müşteriler oldukları için ödüllendirilebilir. Kampanyalar ve özel fırsatlardan

haberdar edilip satın alma alışkanlıklarını aynı şekilde devam ettirmeleri sağlanabilir.
"""



### GÖREV 4 ###

final_tablo = cltv_altiay_final


# 1 haftalık expected purchase tahmini
final_tablo["expected_purc_1_week"] = bgf.predict(1,
                                                  final_tablo['frequency'],
                                                  final_tablo['recency'],
                                                  final_tablo['T']
                                                  )

# 1 aylık expected purchase tahmini
final_tablo["expected_purc_1_month"] = bgf.predict(4,
                                                  final_tablo['frequency'],
                                                  final_tablo['recency'],
                                                  final_tablo['T']
                                                  )

# Gamma-Gamma Submodelini kullanarak expected average profit bulma
final_tablo['expected_average_profit'] = ggf.conditional_expected_average_profit(final_tablo['frequency'],
                                                                                 final_tablo['monetary']
                                                                                )



print("\nFinal Tablo\n")
print(final_tablo.head(10))


# -*- coding: utf-8 -*-
"""
Created on Feb 2020

@author: Tugba Ozkan
"""

import numpy as np 
import pandas as pd 


x= pd.read_csv('Test.csv')
y= pd.read_csv('Train.csv')
#tek bir dosyada kombine etmek
y['source']='train'
x['source']='test'
data = pd.concat([y,x],ignore_index= True)
#concat eklemek birlestirmek baglanti kurmak demektir 
#pandas concat ile ilgili daha fazla bilgi icin : 
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html


print(y.shape, x.shape, data.shape) 
#shape kaç satır ve kaç sutun oldugunu gosterir
a=data.head()
data.head()
 #nesnede ilk n tane verinin dondurulmesi 
#default degeri 5 tir 0 dan 5 e kadar olan veriler doğrulugu kontrol etmek acisindan gösterilir
data.describe()
b=data.describe()
#verilerin kac tane oldugu ortalamsı ,min degeri max deger; istatiksel tanımı 
c=data.apply(lambda z: sum(z.isnull()))
data.apply(lambda z: sum(z.isnull()))
#eksik satır saysını saymanın en iyi yolu 
#sum topla demek isnull bos ise demek lambda=definition
#lambda fonk tanimlaka gibi
data.Item_Outlet_Sales = data.Item_Outlet_Sales.fillna(data.Item_Outlet_Sales.mean())
data.Item_Weight = data.Item_Weight.fillna(data.Item_Weight.mean())
#bosluklari doldurmak icin bu kalıppları kullandık
#o sutunun ortalama degerine gore degerler atadı 
#dafa fazla bilgi icin : 
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html

print(data['Outlet_Size'].value_counts())
#outlet size sutunundaki verilerin türlerinden kac tane oldugunu sayip gosterir
data.Outlet_Size = data.Outlet_Size.fillna('Medium')
t=data.Outlet_Size
#outlet size bosluklarini medium olarak doldur bolumundeki bosluklari doldur fillna=fillnan fill=doldurmak nan=bosluk
 
print(data.Outlet_Size)
#tekrar bos olanları kontrol edelim 
o=data.apply(lambda z: sum(z.isnull()))
print(o)
print(data.info())
#dataula ilgili bilgilerhangi datada kac tane float var gibi
g=data['Item_Identifier'].value_counts()
f=data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
h=data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
cc=data['Item_Type_Combined'].value_counts()
print(cc)
'''
yiyecek mi icecek mi gibi turlerin hangisinden kac tane oldugunu anlamak icin
oncelikle item combined ile item identifier nerdeyse aynı anlama geliyor basit bir
hale indirgemek amacıyla aynı degerler birbirlerine esitlendi ve kisaltmalar verildi 
en sonunda bu kısaltmalara karsilik gelen combinelerden kac tane oldugu gosterildi 
i bunun ornek dosyasıdır

''' 
'''
Oncelikle info attigimiz kisimda objectler ve floatlar gorundu
ve makine diline gore bazilarını basite indirgememiz gerekiyor 
data dosyasina bakacak olursak bunların hangisi olması gerektigine dair
tahminlerde bulunabilirz
bu datalari one hot encoder ve label encoder ile sayisal diziye cevirebiliriz 
genellikle diziye benzeyen string ve sayisal degerleri dizyle kategorize etmek icin
kullanilan kutuphanelerdir
bu kategorizasyonu elle  veya otomatik olusturabiliriz 
daha fazlası icin : 
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
label encoder da numeric array olusturur fakat bu numericin bir kısıtlaması vardır
0-1 arasında bool type false/true  seklinde kodlar orn:erkek 0 kadın 1 gibi .
daha fazlasi icin : 
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
kutuphanemizi cagiralim : 
'''
#from sklearn.prepocessing import OneHotEncoder
#ohe = OneHotEncoder()
from sklearn.preprocessing import LabelEncoder
lb= LabelEncoder()
#simdi outlet degerlerimizi egitip donusturelim : fit-transform
data['Outlet']= lb.fit_transform(data['Outlet_Identifier'])
var_mod= ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
for i in var_mod:
    data[i]= lb.fit_transform(data[i])
    print(data [i])
son_hal=data
#burada sunu yaptik outlet identirifer datalarini egitip degistirp outlet dosyasina aktardık 
    #var_mod kisaktmasiyla sayisallastirmak istedigimiz belgeleri siraladik 
    #for dongusuyla tek tek hepsinin 14204 satirdaki degerlerin sayisal verilere gore kodlanisini yaptik 
son_hal = pd.get_dummies(son_hal, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])
#get dummies birbirne benzeyen degiskenleri tek sınıfa indirgemek icin kullanilir ornek verecek olursak yas ve dogum yili gibi ikisini tek bir degere indirger 
#dummies bu vakalar icin kullanilan bir terimdir machine learningte
print(son_hal.dtypes) #☺son haldeki datalarin tipleri: integer olmus durumda
#dummy ile daha fazla bilgi icin :
#https://www.w3resource.com/pandas/get_dummies.php
ff=son_hal.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
#item_type ve kurulus yillari( establishment year) gereksiz bilgidir ve tahminde ise yaramaz farkli turdedir o yuzden drop ettik yani kaldirdik
#buradaki drop=remove anlaminda kullanilmaktadir.
y = son_hal.loc[son_hal['source']=="train"]
x = son_hal.loc[son_hal['source']=="test"]
#son hali test ve train kisimlarina ayirdik
x.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
#tahmin edilecek deger outlet sales oldugu icin kaldirildi 
y.drop(['source'],axis=1,inplace=True)
#gerekli olmayanlari kaldirdik 
'''
Eger en son hali csv dosyasi olarak gormek istiyorsak su komutlari kullanabiliriz : 
    y.to_csv("train_modified.csv",index=False)
    x.to_csv("test_modified.csv",index=False)
ve bunlari tekrar okutmak icin de 
train2 = pd.read_csv("train_modified.csv")
test2 = pd.read_csv("test_modified.csv")
bu komutlari kullanabiliriz 
'''
print(y.head())
#verilerin dogrulugunu check ettik 
#verilerin sutunlarina baktigimizda yine fazlalik yapan bilgiler var 
#item kimlikleri(identifier) farklı kodlamalar icermekte ve cok fazla one hot yapılamayacak adar bu yuzden kaldırılmalı
#outlet kimlikleri de ayni sekilde oldugu icin drop edilmeli 
#item outlet sales kismi da tahmin ettirecegimiz deger oldugu icin x train kismindan kaldirilmali cunku tahmin yapilacak degerin olmaması gerrekiyor
X_train = y.drop(['Item_Outlet_Sales', 'Outlet_Identifier','Item_Identifier'], axis=1)
Y_train= y.Item_Outlet_Sales
X_test = x.drop(['Outlet_Identifier','Item_Identifier'], axis=1)
Y_test= data.Item_Outlet_Sales
#verileri aynı dunyaya indirdik bu sekilde yaparak yani standardize ettik
#train dosyalari tahmin icin egitilecek dosyalardir 
''' 
Veri on isleme bitti istedigimiz ayara getirdik.
Verilerimiz sayisal oldugu icin 
Regresyonla tahmin yontemleri kullanilir 
Classification daha cok sayisal olmayan veriler icin kullanilan yontemdir
Regresyon yontemi olan Multiple Linear//dogrusal regresyonla baslayalim:
'''
from sklearn.linear_model import  LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train) #modeli insa ediyor;tahmin modelini
#bu model x trainden y traini tahmin edecek,dogrusal bir baglanti olusacak aralarında
tahmin= lr.predict(X_test) 
#predict=tahmin et demek verilen x_test bilgilerinden verilmeyen item_outlet_sales bilgilerini tahmin et
#x trainden y traini ogrendi x testten de kendi tahmin sonuclarini cikartti 
#yuzde kac dogru tahmin edebildigine bakalim: 
lr_accuracy = round(lr.score(X_train,Y_train) * 100,8)
print('Accuracy: (%)')
print(lr_accuracy)
#0.56 ile basladigi icin yuzdesel oranını gormek amacıyla *100 yazdık, 8 ise virgulden sonraki gormek istediginiz kadar girebileceginiz rakamdır.

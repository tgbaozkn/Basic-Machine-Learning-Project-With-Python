# -*- coding: utf-8 -*-
"""
Created on Feb 2020

@author: Tugba Ozkan
"""

import numpy as np 
#numpy kütüphanesini sayısal işlemler için kullaniliriz;örn. array(dizi) oluşturmak gibi.
import pandas as pd 
#dosyaya dair işlemler(okuma,yazma,bölme,birleştirme vs.) için bu kütüphane kullanilir.

#dosyalar pandas kutuphanesiyle okunuyor:
x= pd.read_csv('Test.csv')
y= pd.read_csv('Train.csv')
#tek bir dosyada kombine etmek
y['source']='train'
x['source']='test'
data = pd.concat([y,x],ignore_index= True)
#concat eklemek birlestirmek baglanti kurmak demektir 
#pandas concat ile ilgili daha fazla bilgi icin : 
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
#ignore index= indexlemeyi yok say niye boyle yaptık indexlemek o kadar onem arzetmedigi icin


print(y.shape, x.shape, data.shape) 
#shape kaç satır ve kaç sutun oldugunu gosterir
a=data.head()
data.head()
 #.head () komutu : nesnede ilk n tane verinin dondurulmesi 
#.head() komutunun parantez iiçindeki bolsuk:default degeri 5 tir 0 dan 5 e kadar olan veriler doğrulugu kontrol etmek acisindan gösterilir
data.describe()
b=data.describe()
#verilerin kac tane oldugu ortalamsı ,min degeri max deger; istatiksel tanımı 
c=data.apply(lambda z: sum(z.isnull()))
data.apply(lambda z: sum(z.isnull()))

''
bu dosyalari incelersek satırlarda ve sutunlarda eksik veriler "nan" oldugunu goruruz bu da tahmin ederken yanlıs tahminlere ,
veriler eğitilirken(train kısmı) ,işlenirken yanlış bağlantı kurulmasına yol açar eksik ve fazla olan şeyler hiçbir zaman 
iyi değildir.
''
#eksik satır saysını saymanın en iyi yolu : apply(lambda z: sum(z.isnull())) 
#bir z tanimla ve bu z ile sutunu satırlari dondur bos mu degil mi: bos olanlarin sayisini belirt
#sum topla demek isnull bos ise demek lambda=definition
#lambda fonk tanimlamak gibi
data.Item_Outlet_Sales = data.Item_Outlet_Sales.fillna(data.Item_Outlet_Sales.mean())
data.Item_Weight = data.Item_Weight.fillna(data.Item_Weight.mean())
#bosluklari doldurmak icin bu kalıpları kullandık
''
yukaridaki kodlari turkce aciklayacak olursak:
data.dosyasinin_icinden_su_sutunu_al= bu_sutunu .doldur(data.dosyasinin_bu_sutunundaki_verilerin.ortalaması ile())
''
#o sutunun ortalama degerine gore degerler atadı 
#dafa fazla bilgi icin : 
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html

print(data['Outlet_Size'].value_counts())
#outlet size sutunundaki verilerin türlerinden kac tane oldugunu sayip gosterir
data.Outlet_Size = data.Outlet_Size.fillna('Medium')
#outlet size bosluklarini medium olarak doldur bolumundeki bosluklari doldur fillna=fillnan fill=doldurmak nan=bosluk
#fillna(parantez icine ne yazarsaniz onunla doldurmus olursunuz : string ifadeyse " veya ' ile yazmaniz gerekir)
t=data.Outlet_Size

#t dosyasi ile Outlet Size bosluklari medium ile dolmus mu dolmamis mi onu kontrol ettim.
 
#tekrar bos olanları kontrol edelim 
o=data.apply(lambda z: sum(z.isnull()))
print(o)
print(data.info())
#data.info = datayla ilgili bilgilerhangi datada kac tane float var gibi
g=data['Item_Identifier'].value_counts()
f=data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
h=data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})

'''
g dosyasinda Item kimliklerinden hangisinden kac tane oldugu gosterir 
Identifier ile Type Combined neredeyse aynı anlama geldigi icin (FD54=Food ,DR54=Drink gibi ) birbirine esitlendi ve sadece
identifierin ilk iki harfi alındı x[0:2] bu anlama gelir : FD,DR,NC geriye kalan sayisal karmasik ifadeler atildi.
H dosyasi bu ifadelerin atilmis halidir.
F dosyasi da bu donguye(lambda x:x[0:2]) sahiplik yapan dosyadir.
yiyecek mi icecek mi gibi turlerin hangisinden kac tane oldugunu anlamak icin
oncelikle item combined ile item identifier nerdeyse aynı anlama geliyor basit bir
hale indirgemek amacıyla aynı degerler birbirlerine esitlendi ve kisaltmalar verildi 
en sonunda bu kısaltmalara karsilik gelen combinelerden kac tane oldugu gosterildi 
cc bunun ornek dosyasıdır

''' 
cc=data['Item_Type_Combined'].value_counts()
print(cc)
'''
value.counts ile hangi type dan ne kadar var gorebiliriz :object mi int mi float mi string mi 
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
#label encodere göre outlet identifier stununu eğit ve donustur neye : data outlet arrayine.
var_mod= ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
for i in var_mod:
    data[i]= lb.fit_transform(data[i])
    print(data [i])
son_hal=data
#burada sunu yaptik outlet identirifer datalarini egitip degistirip outlet dosyasina aktardık 
    #var_mod kisaktmasiyla sayisallastirmak istedigimiz belgeleri siraladik 
    #for dongusuyla tek tek hepsinin 14204 satirdaki degerlerin sayisal verilere gore kodlanisini yaptik 
  --i var_mod dizisindeki tüm elemanlarda donsun ve label encodere gore eğitilip donusturulsun --
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
    y.to_csv("train_modified.csv",index=False)  burada da calistigimiz klasörün icine bu dosyalari yazdirir ve sonra,
    x.to_csv("test_modified.csv",index=False)
***************
train2 = pd.read_csv("train_modified.csv") pandas ile okuyabiliriz bu dosyalari.
test2 = pd.read_csv("test_modified.csv")
bu komutlari kullanabiliriz 
'''
print(y.head())
#verilerin dogrulugunu check ettik head komutu ile ilk beş ifadeyi gorerek
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

'''
Gordugunuz uzere Çokllu Dogrusal Regresyon ile dogruluk sadece yuzde 50 cikti.
Baska prediction yontemleriyle neler cikabilecegine bakalim.
Decision Tree yontemi: 
Genellikle classification kısmında kullanilir fakat regresyonda da iyi sonuclar verir 
isleyisi su sekildediler train verilerinden su sekilde ogrenir : 
    ornegin boyu kilosu verilen insanların yaslarini tahmin etmek gorevi olsun ,
    oncelikle verileri boler ve ortalamalarını alır yani mesele boyu 1.70 üstü ve altı olanlar boyu 1.70
    altında olup kilosu 50 altında olanlar ve olmayanlar bunların ortalaması vs vs gibi kısımlara boler.
    1.80 boyunda ve kilosu 80 olan birinin ortalamasına gore yasini tahmin eder.
daha fazla bilgi ve ornekler icin : 
https://scikit-learn.org/stable/modules/tree.html#tree    
'''
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=15,min_samples_leaf=300)
dt.fit(X_train, Y_train)
#max_depth orneklerin kontenjanını kasteder,ne kadar fazla olursa bolunen sayilarin dogrulugu daha iyi olur ve sample leaf ornekleme sayisi ,belli bir ozellige gore partlara ayirip ortalama alma
#dt.fit :predict datası, x ve y ye gore baglantiyi ogrenip kendini egitiyor
tahmin=dt.predict(X_test)
dt_accuracy= round(dt.score(X_train,Y_train)*100,8)
print("Decision Tree Accuracy (%): ")
print(dt_accuracy)
#%58.84050822
'''
Çok fazla veri oldugu icin dogruluk sayisi da cok iyi cikmiyor
ve decision tree ezbere gidiyor yani belli aralıktaki deger sabittir 
farklı veri farklı degere gelse de ezbere gittigi icin bilemiyor. 
Bir de Random Forest(Rassal Agaclar) tahminini deneyelim : 
    Birden fazla decision treenin tahmin etmek icin bir araya gelmesidir 
    İyi bir dogruluk almak icin birden fazla Decision Tree lerin birleşmesi;
    her parçayı tekrar kucuk parcalara bolup onların ortalamasını almasıdır.
    cok verilerde kullanilmasi daha iyidir.ensemble öğrenme yani kollektif öğrenme olarak geçer.
    birden fazla decision treedan olusan anlamına gelir.
    daha fazla bilgi icin : 
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        
'''

from sklearn.ensemble import RandomForestRegressor 
rfr= RandomForestRegressor(n_estimators=250,max_depth=6, min_samples_leaf=150,n_jobs=4)
#n_estimators : ne kadar decision tree kullanilacak
#ornekleme kontenjanı =max_depth
#ornekleme sayisi= min_samples_leaf
#☺n_jobs : paralel calisan is sayisi ;fit ederken ,predict yaparken ,agaclarin hepsini calistirirken vs aynı anda yapilan is sayisi
rfr.fit(X_train,Y_train)
tahmin=rfr.predict(X_test)
#dogruluk oranına bakalim : 
rfr.accuracy= round(rfr.score(X_train,Y_train)*100,5)
print( "Random Forest Accuracy (%): ")
print(rfr.accuracy)
#♣Yaklasik %60 oranında dogruluk verdi

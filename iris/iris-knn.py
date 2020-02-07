# -*- coding: utf-8 -*-
"""
Created on Thu Feb 6 2020

@author: Tugba Ozkan
"""
import numpy as np #sayisal islemler icin kullanilir
import pandas as pd  #xls,csv gibi dosyalari okumak ve islem yapmak icin kullanilir

from mpl_toolkits.mplot3d import Axes3D #3 boyutlu graphic cizmek icin 

from sklearn import datasets
iris= datasets.load_iris() #bulunan iris datasetinden satirlarca olan ozelliklerini toplamak   

type(iris)
print(iris.data) #iris datasını yazdır 
print(iris.feature_names) #baslıklakları yazdirdi ,yani bu sayiların ne olduguna dair short information 
print(iris.target)  #tahmin icin kullanilacak verilerin integera donusmesi 
print(iris.target_names) #tahmin edilecek kisimlarin adları : setosa,versicolor,virginica
#0= setosa, 1=versicolor, 2= virginica
#genel olalarak kategorize edilen ozellikler vara o classificvation problemidir
#burada 0-1-2 olarak convert ettigimiz uc kategorik deger var
print type(iris.data)
print type(iris.target)
#satırları yani ozellik sayilarını(kac satir var) kontrol etmek ve basliklari(kac sutun var) kontrol etmek icin kullanilan kod 
print(iris.data.shape)

x= iris.data
y= iris.target 
#plt.plot(x,y) birdenbire bu sekilde grafik cizerseniz neyin neye gore cizildigi anlasilmaz
#bu yüzden once bir algoritma kullanilmasi gerekir istediginiz classification algoritmasını cekebilirsiniz
#ben knn kullanacagim bu sayfadan sizde karar verebilirsiniz accuracy oranlarına gore 
#sayfa url = "https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html"
from sklearn.neighbors import KNeighborsClassifier
#kolaylikla kullanabilmek icin kisaltma yapiyorum 
knc= KNeighborsClassifier(n_neighbors= 5,metric='minkowski') #komsu tanimla 5 (default deger de budur) yaptim ve mesafeye gore tahmin yapması icin buna gore uygun bir algoritma secilmeli 
#for example : oklid ama oklid genel olarak ilkel seviyeye kacar ve tam dogru sonuc vermez
#websitesinden distance(mesafe) algoritmalarına bakabilirsiniz 
#url = https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
#problemimiz lineer olmadiginda knn kullanilabilir
#knn de koordinat sisteminde kısımlara komsuluklara ayrılır ve tahmin edilecek nokta hangi noktaya daha yakinsa ondan sayilir 
#bu tahmin  mesafe algoritmalarla degisebilir
#lazy ve eager learning olarak ikiye ayrilir 
#eager  : once ogrenip sonra siniflandirir ,kendisi bolge olusturur
#lazy : girilen bilgileri tutar butun verilere bakar ve tahmini belirlemeye calisir,bolge olusturmadan mesafeye bakar 
print(knc) #default atanan degerleri gosterir
'''knc.fit(x,y) '''
#makine ogrenmesinde veriyi test ve egitmek icin bolmemiz gerekir 
#oranı ayarlamasını kendinize göre deneyebilirsiniz ,verinin ne kadarını makineye vrmek bunun uzerınden ogrenmesini saglamak istiyorsanız o kadar oranda egitime ayırın 
#test kisminda da uzerinden tahmin yurutmesi ve test yapması icin ayrilan kisim 
#benim mantigima gore ne kadar cok egitirsem yani fazla oranda egitime vermek istiyorum ki accuracy o kadar yuksek ciksin 
#bunu skilearn librarysinden model selection'dan train test split(bolmesi)  import edilir
#x=egitim ,y=tahmin etmesini istedigimiz degerler
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.40, random_state= 0)
#4 ozelligimiz vardı hepsi birbirinden farklı ,farklı ortalamaya sahip o yüzden hepsini orta noktada bulusturmak icin standartlastirma yapacagiz = istatistik  
#bunu standardscaler ile yapabiliriz 
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(x_train) #degistir_uygula
X_test = sc.fit_transform(x_test)

#verileri aldi kendi icinde standartlastirildi -sonsuzla +sonsuz arasında degerler alır fkat ortalama olarak 0 a cekmeye calisir
knc.fit(X_train,y_train)
y_predict= knc.predict(X_test)
#♣dogruluk degerini gormek istiyorsak bu kodu kullanabiliriz 
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_predict))

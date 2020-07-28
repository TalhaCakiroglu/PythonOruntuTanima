from matplotlib import pyplot as plt
import kutuphane
from sklearn.feature_selection import SelectKBest, f_classif

import warnings
warnings.filterwarnings("ignore")

#1- Verileri yükle
giris, cikis, CustomerID =  kutuphane.dosya_oku('data/Credit_Card_Applications.csv')
#CustomerID['Country'] = CustomerID['Country'].replace([":",','],"").astype(int)


#2 Ölçeklendir
olcekli_giris = kutuphane.olceklendir(giris)

#Parametre Optimizasyonu
sonuclar = kutuphane.parametre_optimizasyonu(giris,cikis, CustomerID)

#Özellik Seçimi
dogruluk_chi = []

#Chi-square 
for k in range(1,16,1):
    ozellikler  = kutuphane.chi2_ozellik_cikar(giris,cikis,k)
    azaltilmis_olcekli_giris = olcekli_giris[:,ozellikler]
    dogruluk,f1skor = kutuphane.basari_hesaplaCV(azaltilmis_olcekli_giris, cikis, CustomerID,10)
    dogruluk_chi.append(dogruluk)
    print("k="+str(k) + " acc="+str(dogruluk))

dogruluk_anova = []
for k in range(1,16,1):
    azaltilmis_olcekli_giris = SelectKBest(f_classif, k=k).fit_transform(X=olcekli_giris,y=cikis)
    dogruluk,f1skor = kutuphane.basari_hesapla(azaltilmis_olcekli_giris, cikis, CustomerID)
    print("k="+str(k) + " acc="+str(dogruluk))
    dogruluk_anova.append(dogruluk)
import numpy as np
x_ekseni = np.arange(1,16,1)
fig,plots = plt.subplots(2,1)
plots[0].plot(x_ekseni,dogruluk_chi)
plots[0].set_xlabel('Chi-square Özellik Sayısı')
plots[0].set_ylabel('Doğruluk')
plots[1].plot(dogruluk_anova)
plots[1].set_xlabel('Anova-F Özellik Sayısı')
plots[1].set_ylabel('Doğruluk')

plt.show()




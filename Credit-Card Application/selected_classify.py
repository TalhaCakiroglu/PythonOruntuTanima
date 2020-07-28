import kutuphane
from sklearn.feature_selection import SelectKBest, f_classif
#1- Verileri yükle
giris, cikis,CustomerID =  kutuphane.dosya_oku('data/Credit_Card_Applications.csv')
#kisi_bilgisi['Country'] = kisi_bilgisi['Country'].replace([":",','],"").astype(int)
olcekli_giris = kutuphane.olceklendir(giris)
dogruluk,f1skor = kutuphane.basari_hesapla(olcekli_giris, cikis, CustomerID)

#Chi-square 
for k in range(1,9,1):
    ozellikler  = kutuphane.chi2_ozellik_cikar(giris,cikis,k)
    azaltilmis_olcekli_giris = olcekli_giris[:,ozellikler]
    dogruluk,f1skor = kutuphane.basari_hesaplaCV(azaltilmis_olcekli_giris, cikis, CustomerID,10)
    print("k="+str(k) + " acc="+str(dogruluk))
 
for k in range(1,9,1):
    azaltilmis_olcekli_giris = SelectKBest(f_classif, k=k).fit_transform(X=olcekli_giris,y=cikis)
    dogruluk,f1skor = kutuphane.basari_hesaplaCV(azaltilmis_olcekli_giris, cikis, CustomerID)
    print("k="+str(k) + " acc="+str(dogruluk))
ozellikler  = kutuphane.chi2_ozellik_cikar(giris,cikis,140)



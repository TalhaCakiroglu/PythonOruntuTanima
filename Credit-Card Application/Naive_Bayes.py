import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


#Dosyayi Yukle
veri = pd.read_csv('data/Credit_Card_Applications.csv')
ozellik_sayisi = 16


#giris cikis belirle
giris_verileri = veri.iloc[:,1:ozellik_sayisi+1]
cikis = veri.iloc[:,-1]



#Egitim ve test verilerini ayir
egitim_giris, test_giris,egitim_cikis, test_cikis = train_test_split(giris_verileri,cikis, test_size=0.15, random_state=0)



#Gaussian NB
olasilik_modeli = GaussianNB()
olasilik_modeli.fit(egitim_giris, egitim_cikis)
cikis_tahmin = olasilik_modeli.predict(test_giris)



#Başarıyı belirle
basari = accuracy_score(test_cikis, cikis_tahmin)
fSkor =  f1_score(test_cikis, cikis_tahmin, labels=None, pos_label=1, average='binary', sample_weight=None)

## Setting Kaggle
```
!mkdir ~p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!pip install kaggle
!kaggle datasets download -d uciml/breast-cancer-wisconsin-data
```

## Pandas
```
df.info() = melihat isi semua kolom beserta jenisnya
df.describe() = melihat std mean di dataset --- termasuk eksplorasi data (EDA)
df.nunique() = melihat jenis unik atau jenis berapa saja yang beda beda seperti 1 laki 0 perempuan
df['kolom'].unique() = jenis kolom tertentu
df.isnull().sum() = menampilkan missing value
df.duplicated().sum() = menampilkan duplikat
print(df[df.duplicated()]) = melihat apa saja yang duplikat
df.dropna() = hapus missing value
df.drop_duplicates() = hapus diplicated
Y = df['diagnosis']
X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1) = drop target biasanya target kolom
df.iloc[0] = menampilkan 1 data saja index berapa
```

## Catatan df.describe()
```
Std kecil → data ngumpul rapat di sekitar mean
Std besar → data nyebar jauh dari mean

25% (Q1) → 25% data berada di bawah nilai ini
50% (Median) → titik tengah data (setengah di bawah, setengah di atas)
75% (Q3) → 75% data berada di bawah nilai ini
Q1–Q3 = IQR → 50% data paling representatif
Dipakai untuk:
melihat sebaran data tengah
deteksi outlier
lebih tahan outlier dibanding mean/std
```

## Encoding
```
y = y.map({'M': 1, 'B': 0})
```
jika semisal targetnya itu masih bentuk huruf harus di encoding

## Standarisasi
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # BELAJAR DARI DATA 
X_test = scaler.fit_transform(X_test) # DIANGGAP DATA BARU
```

## Split Test
```
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

print(X_train.shape) (455, 30) = 30 itu kolomnya
print(X_test.shape) (114, 30)
print(y_train.shape) (455,)
print(y_test.shape) (114,)

455 + 144 = keseluruhan dataset, 80% dan 20 %
y bukan bagian dari X
y hanya berasal dari target
```

## Menampilkan Akurasi
```
from sklearn.metrics import accuracy_score
y_pred = rf.predict(X_test)
akurasi = accuracy_score(y_test,y_pred)
print(f"{akurasi*100:.2f}%")
```
y_pred itu mengubah X_test yang tadi 114,30, kan X_test gak ada Y nya,
dia itu diprediksi dan diubah menjadi Y_pred atau Y itu sendiri,
terus Ypred dibandingkan dengan y_test karena y_test itu data yang asli setelah itu, munculah akurasi

## Menampilkan Confusion Matrix
```
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
```

## Classification Report
```
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## Prediksi Machine Learning
```
data_baru = [[
    14.2, 20.1, 92.3, 654.1, 0.096,
    0.104, 0.088, 0.048, 0.18, 0.062,
    0.42, 1.12, 2.88, 35.2, 0.006,
    0.021, 0.028, 0.01, 0.02, 0.003,
    16.3, 25.4, 107.2, 880.5, 0.132,
    0.251, 0.31, 0.12, 0.29, 0.082
]]
data_baru = scaler.transform(data_baru) DI STANDARISASI DULU
prediksi = rf.predict(data_baru)
print(prediksi)
output = [2]

if prediksi[0] == 1:
    print("Prediksi: Kanker GANAS")
else:
    print("Prediksi: Kanker JINAK")
```
## Urutan Machine Learning
```
1. siapkan kaggle
2. bersihkan data
3. pilih Y dan X (yang tanpa Y) 
4. encoding
5. split
6. standariasi
7 model fit
8. menampilkan akurasi
9. confusion matrix
10. prediksi
```

## Jenis Model
- RandomForest (tidak butuh scaler):
  
```
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier() atau RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train,y_train)
```
Logistic Regression = Klasifikasi

Linear Regression = Regresi

- KNN
  ```
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors=5)
  knn.fit(X_train,y_train)
  ```
  
  KNN itu bisa ngelakuin jika datasetnya itu tidak ada kategori, atau semisal gender itu kan ada laki laki dan perempuan nah sedangkan umur itu cuma angka
jadi KNN bisa ngatasi masalah yang cuma angka itu bukan seperti gender laki laki perempuan


## DATASET YANG PERNAH DIPAKAI
- Breast Cancer
- Heart Disease
- Edible Posion Musroom

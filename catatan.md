=== Setting Kaggle ===
!mkdir ~p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!pip install kaggle
!kaggle datasets download -d uciml/breast-cancer-wisconsin-data

=== Pandas ===
df.info = melihat isi semua kolom beserta jenisnya
df.isnull().sum() = menampilkan missing value
df.duplicated().sum() = menampilkan duplikat
df.dropna() = hapus missing value
df.drop_duplicates() = hapus diplicated
X = df.drop(columns=['']) = drop target biasanya target kolom

=== Encoding ===
y = y.map({'M': 1, 'B': 0})

=== Standarisasi ===
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

=== Split Test ===
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print(X_train.shape) (455, 30) = 30 itu kolomnya
print(X_test.shape) (114, 30)
print(y_train.shape) (455,)
print(y_test.shape) (114,)
455 + 144 = keseluruhan dataset, 80% dan 20 %

=== Menampilkan Akurasi ===
from sklearn.metrics import accuracy_score
y_pred = rf.predict(X_test)
akurasi = accuracy_score(y_test,y_pred)
print(f"{akurasi*100:.2f}%")


=== Urutan Machine Learning ===
1. siapkan kaggle
2. bersihkan data
3. pilih Y dan X (yang tanpa Y) 
4. encoding
5. split
6. standariasi
7 model fit
8. menampilkan akurasi

=== Jenis Model ===
RandomForest:
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)

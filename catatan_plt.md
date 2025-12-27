# Grafik Batang
```
import matplotlib.pyplot as plt

plt.hist(df['age'])
plt.title("Distribusi Age")
plt.xlabel('Age')
plt.ylabel("jumlah")
plt.show()
```

## plt.hist(df['age'],bins=10,color='skyblue',edgecolor='black')
- bins = jumlah “kotak” atau “ember” untuk mengelompokkan data di histogram.
- color = warna bar
- edgecolor = warna tepi bar
- alpha = makin kecil makin transparan
- label = untuk memperlihatkan di pojok kiri atau kanan penjelasan semisal age warna apa gitu

## plt.legend() lokasi (harus dikasih label di plt hist nya
```
plt.legend(loc='upper right')   # kanan atas
plt.legend(loc='upper left')    # kiri atas
plt.legend(loc='lower right')   # kanan bawah
plt.legend(loc='lower left')    # kiri bawah
plt.legend(loc='center')        # tengah
```

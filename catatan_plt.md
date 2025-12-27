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
- rwidth = semakin dikit semakin pisah, defaultnya 1.0 jadi jika 0.9 bar nya bakal pisah

## plt.xticks(np.arange(30,75,5))
untuk menampilkan jarak dari plt.xlabel berarti itu awalnya 30, akhirnya 75, kelipatannya 5

## plt.legend() lokasi (harus dikasih label di plt hist nya
```
plt.legend(loc='upper right')   # kanan atas
plt.legend(loc='upper left')    # kiri atas
plt.legend(loc='lower right')   # kanan bawah
plt.legend(loc='lower left')    # kiri bawah
plt.legend(loc='center')        # tengah
```

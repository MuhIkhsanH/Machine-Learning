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
- ```bins=[-0.5,0.5,1.5]``` = biar di tengah untuk laki laki perempuan 
- color = warna bar
- edgecolor = warna tepi bar
- alpha = makin kecil makin transparan
- label = untuk memperlihatkan di pojok kiri atau kanan penjelasan semisal age warna apa gitu
- rwidth = semakin dikit semakin pisah, defaultnya 1.0 jadi jika 0.9 bar nya bakal pisah

## hist biner rapi
```
import matplotlib.pyplot as plt
import numpy as np

plt.hist(df['sex'],bins=[-0.5,0.5,1.5],rwidth=0.9,edgecolor='black')
plt.title("KELAMIN")
plt.xticks([0,1],["Laki Laki","Perempuan"])
plt.yticks(np.arange(100,1000,100))
plt.xlabel('Gender')
plt.ylabel("Jumlah")
plt.show()
```
## hist untuk lebih dari 2, semisal 3
```
pakai bins = [-0.5, 0.5, 1.5, 2.5]
```
## hist tapi ada angka diatasnya
```
import matplotlib.pyplot as plt
import numpy as np

counts, bins, patches = plt.hist(df['Outcome'], bins=[-0.5, 0.5, 1.5], rwidth=0.8)
plt.xticks([0,1], ["Not Diabetes", "Diabetes"])
plt.xlabel("Outcome")
plt.ylabel("Total")

for count, patch in zip(counts, patches):
    plt.text(patch.get_x() + patch.get_width()/2, count + 0.5,  # posisi x dan y
             int(count), ha='center', va='bottom')

plt.show()

```

## plt.xticks(np.arange(30,75,5))
untuk menampilkan jarak dari plt.xlabel berarti itu awalnya 30, akhirnya 75, kelipatannya 5
 - rotation=90 = untuk memutar teks dari xlabel jadi agak kebawah
 - ```plt.xticks([0,1],["Laki Laki","Perempuan"])``` membuat ditengah dan ada namanya

## plt.legend() lokasi (harus dikasih label di plt hist nya
```
plt.legend(loc='upper right')   # kanan atas
plt.legend(loc='upper left')    # kiri atas
plt.legend(loc='lower right')   # kanan bawah
plt.legend(loc='lower left')    # kiri bawah
plt.legend(loc='center')        # tengah
```

## CONFUSION MATRIX
```
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_test = [0,0,1,1,0,1,0,1,1,0]
y_pred = [0,0,1,0,0,1,0,1,0,0]

cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Laki Laki", "Perempuan"],
            yticklabels=["Laki Laki", "Perempuan"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

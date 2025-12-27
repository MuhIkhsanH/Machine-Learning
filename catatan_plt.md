## Grafik Batang
```
import matplotlib.pyplot as plt

plt.hist(df['age'])
plt.title("Distribusi Age")
plt.xlabel('Age')
plt.ylabel("jumlah")
plt.show()
```

```
plt.hist(df['age'],bins=10,color='skyblue',edgecolor='black')
```
- color = warna bar
- edgecolor = warna tepi bar

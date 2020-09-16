# soal no 1

Bagaimana proses algoritma KNN?
1. menghitung jarak antar data poin
2. mengambil K terdekat
3. melakukan voting

Jelaskan minimal 2 kekurangan dari algoritma KNN?
1. sensitif terhadap data pencilan
2. jika jumlah K nya genap dan kelas label tetangganya sama, ia akan mengambil kelas secara random

# soal no 2

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

wine = load_wine()
df = pd.DataFrame(np.c_[wine['data'], wine['target']], columns= wine['feature_names'] + ['class'])

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

ks = np.arange(2, 21)
weights = ['uniform', 'distance']

def knn_predict(k, weight):
    model_knn = KNeighborsClassifier(n_neighbors = k, weights = weight)
    model_knn.fit(X_train, y_train)
    score = model_knn.score(X_test, y_test)
    return score

list_uniform_score = []
list_distance_score = []

for k, weight in product(ks, weights):
    score = knn_predict(k, weight)
    if weight == 'uniform':
        list_uniform_score.append(score)
    elif weight == 'distance':
        list_distance_score.append(score)
                        
fig, ax = plt.subplots(figsize = (18, 8))

ax.plot(ks, list_uniform_score, c = 'b', label = 'Uniform Score', marker = 'o')
ax.plot(ks, list_distance_score, c = 'orange', label = 'Distance Score', marker = 'o')

fig.legend()
ax.set_xlabel('K-value')
ax.set_ylabel('Scores')
ax.set_title('Accuracy : Uniform Scores vs Distance Scores')
print('Uniform Scores :', list_uniform_score)
print('Distance Scores :', list_distance_score)
plt.show()

# soal no 3

from sklearn.linear_model import LinearRegression
import numpy as np

rng = np.random.RandomState(1)

X = 10 * rng.rand(50, 4)
y = np.array([  0.9826564 ,  49.40390035,  85.76013175,  29.17254633,
        11.270054  ,  61.53852735, -19.52503854,  10.10230867,
        82.03058206,  45.72660678,  82.09252575,  90.78872391,
        67.94178098,  39.97492762, 124.78866966,  85.24615819,
        85.68491086,  80.39481211,  14.54591581, 137.19722354,
        85.04063428,  94.76681927,  84.64289989,  38.76421156,
        47.33953927,  94.50853335,  77.3276567 ,  -9.77846805,
        61.13434468,  42.12124052,  26.65634335, 120.41593333,
       138.08132504, 103.33745675, 145.64447692,  19.54777986,
        87.40631024, 111.47327389,  95.94874761, 146.34817502,
       118.10333528,  56.85956484, 108.57660235, 114.30790247,
        40.27107041,  73.83871008,  45.52217182,  -7.57752547,
       147.81772162, 139.40285349])

model_lr = LinearRegression(fit_intercept = True)

model_lr.fit(X, y)

y_predict = model_lr.predict(X)

print('Parameter Koefisien :', model_lr.coef_)
print('Paramater Intercept :', model_lr.intercept_)
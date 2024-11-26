# Logistic Regression

# Kutubxonalarni Import qilish
import numpy as np            # NumPy kutubxonasini import qilish - matematik hisoblar uchun
import matplotlib.pyplot as plt  # Matplotlib kutubxonasini import qilish - grafiklar yaratish uchun
import pandas as pd           # Pandas kutubxonasini import qilish - ma'lumotlarni boshqarish uchun

# Datasetni Import qilish
dataset = pd.read_csv('Social_Network_Ads.csv')  # Ma'lumotlar faylini yuklaymiz
X = dataset.iloc[:, :-1].values                  # X o'zgaruvchiga ma'lumotlar tanlanadi (so'nggi ustunsiz)
y = dataset.iloc[:, -1].values                   # y o'zgaruvchiga chiqish ma'lumotlari tanlanadi (so'nggi ustun)

# Datasetni Trening va Test to'plamlariga bo'lish
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# ma'lumotlarni 75% trening va 25% test uchun bo'lish

# Feature Scaling - Ma'lumotlarni normallashtirish
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()         # Standart skeyler yaratish
X_train = sc.fit_transform(X_train)  # Trening ma'lumotlarni normallashtirish
X_test = sc.transform(X_test)        # Test ma'lumotlarini normallashtirish

# Logistic Regression modelini Trening to'plamida o'rgatish
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)  # Logistic Regression model yaratish
classifier.fit(X_train, y_train)                   # Modelni trening to'plamiga o'rgatish

# Yangi natijani taxmin qilish
print(classifier.predict(sc.transform([[30,87000]])))  # 30 yosh va 87000 maosh uchun taxmin qilish

# Test to'plamidagi natijalarni taxmin qilish
y_pred = classifier.predict(X_test)                   # Test to'plamidagi qiymatlarni taxmin qilish
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# Haqiqiy va taxmin qilingan qiymatlarni yonma-yon ko'rsatish

# Confusion Matrix yaratish
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)                # Test uchun to'g'ri va noto'g'ri natijalar jadvali
print(cm)                                            # Confusion Matrixni chop qilish
accuracy_score(y_test, y_pred)                       # Modellni aniqchilik foizini hisoblash

# Trening to'plami natijalarini vizuallashtirish
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train   # Skalingdan oldingi X qiymatlarni olish
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))  # Klasslar orasidagi chegara
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Trening to\'plami)')
plt.xlabel('Yosh')
plt.ylabel('Baholanadigan Ish Haqi')
plt.legend()
plt.show()

# Test to'plami natijalarini vizuallashtirish
X_set, y_set = sc.inverse_transform(X_test), y_test   # Test to'plami uchun skalingdan oldingi X qiymatlarni olish
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))  # Klasslar orasidagi chegara
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test to\'plami)')
plt.xlabel('Yosh')
plt.ylabel('Baholanadigan Ish Haqi')
plt.legend()
plt.show()

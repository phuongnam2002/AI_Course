import numpy as np

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170]]).T  # Chiều cao
y = np.array([[49, 50, 51, 52, 53, 54, 55, 56]]).T  # Label: cân nặng

one = np.ones((X.shape[0], 1))
X_new = np.concatenate((one, X), axis=1)

A = np.dot(X_new.T, X_new)
b = np.dot(X_new.T, y)

w = np.dot(np.linalg.pinv(A), b)

print(f'w = {w}')

w_0 = w[0][0]
w_1 = w[1][0]

from sklearn import datasets, linear_model

model = linear_model.LinearRegression(fit_intercept=False)  # Không sử dụng bias, tức là ko chèn thêm số 1 vào mỗi biến
model.fit(X_new, y)

print(model.coef_)  # Nghiệm của bài toán

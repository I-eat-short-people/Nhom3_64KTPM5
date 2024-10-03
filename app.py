import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 1. Đọc và tiền xử lý dữ liệu
df = pd.read_csv('student-mat.csv', sep=';')

# Chỉ lấy các cột cần thiết
df = df[['sex', 'studytime', 'failures', 'G3']]

# Biến đổi cột 'sex' thành nhãn số (Label Encoding)
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])

# Tách biến đầu vào và biến mục tiêu
X = df[['sex', 'studytime', 'failures']]
y = df['G3']

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Xây dựng các mô hình
# 2.1 Hồi quy tuyến tính
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
r2_linear = r2_score(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)

# 2.2 Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
r2_lasso = r2_score(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)

# 2.3 Neural Network - MLPRegressor
mlp_model = MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)
r2_mlp = r2_score(y_test, y_pred_mlp)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
# Tạo mô hình Stacking từ các mô hình hồi quy trước đó
estimators = [
    ('linear', linear_model),
    ('lasso', lasso_model),
    ('mlp', mlp_model)
]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacking_model.fit(X_train, y_train)

# Dự đoán với Stacking model
y_pred_stacking = stacking_model.predict(X_test)
r2_stacking = r2_score(y_test, y_pred_stacking)
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
rmse_stacking = np.sqrt(mse_stacking)

# 3. Giao diện Streamlit
st.title("Dự đoán kết quả học tập")

# Nhập thông tin từ người dùng
sex = st.selectbox("Giới tính", ("Nam", "Nữ"))
studytime = st.slider("Thời gian học tập (1-4)", 1, 4, 2)
failures = st.slider("Số lần trượt môn", 0, 3, 0)

# Chuyển đổi giới tính
sex = 1 if sex == "Nam" else 0

# Khi người dùng nhấn nút "Dự đoán"
if st.button("Dự đoán"):
    # Chuẩn bị dữ liệu để dự đoán
    features = np.array([[sex, studytime, failures]])

    # Dự đoán với các mô hình
    pred_linear = linear_model.predict(features)[0]
    pred_lasso = lasso_model.predict(features)[0]
    pred_mlp = mlp_model.predict(features)[0]
    pred_stacking = stacking_model.predict(features)[0]

    # Hiển thị kết quả
    st.subheader("Kết quả dự đoán:")
    
    # Linear Regression
    st.write("### Hồi quy tuyến tính")
    st.write(f"Dự đoán: {pred_linear:.2f}")
    st.write(f"R²: {r2_linear:.2f}, MSE: {mse_linear:.2f}, RMSE: {rmse_linear:.2f}")

    # Lasso Regression
    st.write("### Lasso")
    st.write(f"Dự đoán: {pred_lasso:.2f}")
    st.write(f"R²: {r2_lasso:.2f}, MSE: {mse_lasso:.2f}, RMSE: {rmse_lasso:.2f}")

    # Neural Network (MLP)
    st.write("### Neural Network (MLP)")
    st.write(f"Dự đoán: {pred_mlp:.2f}")
    st.write(f"R²: {r2_mlp:.2f}, MSE: {mse_mlp:.2f}, RMSE: {rmse_mlp:.2f}")
    
    # Stacking
    st.write("### Stacking")
    st.write(f"Dự đoán: {pred_stacking:.2f}")
    st.write(f"R²: {r2_stacking:.2f}, MSE: {mse_stacking:.2f}, RMSE: {rmse_stacking:.2f}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 1. Đọc và tiền xử lý dữ liệu
df = pd.read_csv('student-mat.csv', sep=';')

# Chỉ lấy các cột cần thiết
df = df[['sex', 'studytime', 'failures','absences', 'freetime', 'G3']]

# Biến đổi cột 'sex' thành nhãn số (Label Encoding)
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])

# Tách biến đầu vào và biến mục tiêu
X = df[['sex', 'studytime', 'failures', 'absences', 'freetime']]
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
absences = st.slider("Số buổi vắng học", 0, 93, 5)
freetime = st.slider("Thời gian rảnh (1-5)", 1, 5, 3)


# Chuyển đổi giới tính
sex = 1 if sex == "Nam" else 0

# Lựa chọn mô hình dự đoán
model_choice = st.selectbox("Chọn phương pháp dự đoán", ("Linear Regression", "Lasso Regression", "Neural Network", "Stacking"))


# Khi người dùng nhấn nút "Dự đoán"
if st.button("Dự đoán"):
    # Chuẩn bị dữ liệu để dự đoán
    features = np.array([[sex, studytime, failures, absences, freetime]])

    # Dự đoán dựa trên mô hình đã chọn
    if model_choice == "Linear Regression":
        prediction = linear_model.predict(features)[0]
        r2 = r2_linear
        mse = mse_linear
        rmse = rmse_linear
    elif model_choice == "Lasso Regression":
        prediction = lasso_model.predict(features)[0]
        r2 = r2_lasso
        mse = mse_lasso
        rmse = rmse_lasso
    elif model_choice == "Neural Network":
        prediction = mlp_model.predict(features)[0]
        r2 = r2_mlp
        mse = mse_mlp
        rmse = rmse_mlp
    else:
        prediction = stacking_model.predict(features)[0]
        r2 = r2_stacking
        mse = mse_stacking
        rmse = rmse_stacking

    # Hiển thị kết quả dự đoán
    st.subheader("Kết quả dự đoán:")
    st.write(f"Phương pháp: {model_choice}")
    st.write(f"Dự đoán: {prediction:.2f}")
    st.write(f"R²: {r2:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

    # Vẽ biểu đồ so sánh giá trị thực và dự đoán
    y_test_pred = stacking_model.predict(X_test) if model_choice == "Stacking" else None
    if y_test_pred is not None:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_test_pred, edgecolors=(0, 0, 0))
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
        ax.set_xlabel('Giá trị thực tế')
        ax.set_ylabel('Giá trị dự đoán')
        ax.set_title(f'So sánh giá trị thực tế và dự đoán - {model_choice}')
        st.pyplot(fig)

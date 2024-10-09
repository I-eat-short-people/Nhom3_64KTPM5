import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

# 1. Đọc và tiền xử lý dữ liệu
df = pd.read_csv('student-mat.csv', sep=';')

# Chỉ lấy các cột cần thiết 
df = df[['sex', 'studytime', 'failures', 'absences', 'freetime', 'nursery', 'G1', 'G2', 'G3']]

# Biến đổi cột 'sex' thành nhãn số (Label Encoding)
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['nursery'] = le.fit_transform(df['nursery'])  # Mã hóa nursery

# Tách biến đầu vào và biến mục tiêu
X = df[['sex', 'studytime', 'failures', 'absences', 'freetime', 'nursery', 'G1', 'G2']]  # Chỉ lấy 8 thuộc tính
y = df['G3']

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Xây dựng các mô hình
# 2.1 Hồi quy tuyến tính (Chỉ với 8 thuộc tính)
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_linear_train = linear_model.predict(X_train_scaled)  # Dự đoán trên tập huấn luyện
y_pred_linear_test = linear_model.predict(X_test_scaled)    # Dự đoán trên tập kiểm tra
r2_linear = r2_score(y_test, y_pred_linear_test)
mse_linear = mean_squared_error(y_test, y_pred_linear_test)
rmse_linear = np.sqrt(mse_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear_test)

# 2.2 Lasso Regression
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(Lasso(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

best_lasso = grid_search.best_estimator_

# 2.3 Neural Network - MLPRegressor
mlp_model = MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=1000, random_state=42, learning_rate='adaptive', alpha=0.0001)
mlp_model.fit(X_train_scaled, y_train)

# Tạo mô hình Stacking từ các mô hình hồi quy trước đó (base models)
estimators = [
    ('linear', linear_model),
    ('lasso', best_lasso),
    ('mlp', mlp_model)
]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
stacking_model.fit(X_train_scaled, y_train)

# Dự đoán cho tập kiểm tra với các mô hình
y_pred_lasso_test = best_lasso.predict(X_test_scaled)
y_pred_mlp_test = mlp_model.predict(X_test_scaled)
y_pred_stacking_test = stacking_model.predict(X_test_scaled)

# 3. Giao diện Streamlit
st.title("Dự đoán kết quả học tập")

# Nhập thông tin từ người dùng
sex = st.selectbox("Giới tính", ("Nam", "Nữ"))
studytime = st.slider("Thời gian học tập (1-4)", 1, 4, 2)
failures = st.slider("Số lần trượt môn", 0, 3, 0)
absences = st.slider("Số buổi nghỉ học", 0, 93, 5)
freetime = st.slider("Thời gian rảnh (1-5)", 1, 5, 3)
nursery = st.selectbox("Có đi học thêm không", ("Có", "Không"))
g1 = st.slider("Điểm kiểm Tra lần 1", 0, 20)
g2 = st.slider("Điểm kiểm tra lần 2", 0, 20)

# Chuyển đổi giới tính và nursery
sex = 1 if sex == "Nam" else 0
nursery = 1 if nursery == "Có" else 0

# Lựa chọn mô hình dự đoán
model_choice = st.selectbox("Chọn phương pháp dự đoán", ("Linear Regression", "Lasso Regression", "Neural Network", "Stacking"))

# Khi người dùng nhấn nút "Dự đoán"
if st.button("Dự đoán"):
    # Chuẩn bị dữ liệu để dự đoán (chuẩn hóa dữ liệu)
    features = np.array([[sex, studytime, failures, absences, freetime, nursery, g1, g2]])
    features_scaled = scaler.transform(features)

    # Dự đoán dựa trên mô hình đã chọn
    if model_choice == "Linear Regression":
        prediction = linear_model.predict(features_scaled)[0]
    elif model_choice == "Lasso Regression":
        prediction = best_lasso.predict(features_scaled)[0]
    elif model_choice == "Neural Network":
        prediction = mlp_model.predict(features_scaled)[0]
    else:
        prediction = stacking_model.predict(features_scaled)[0]

    # Hiển thị kết quả dự đoán
    st.subheader("Kết quả dự đoán:")
    st.write(f"Phương pháp: {model_choice}")
    st.write(f"Dự đoán: {prediction:.2f}")

    # Vẽ biểu đồ cho tập huấn luyện
    fig_train, ax_train = plt.subplots()
    ax_train.scatter(y_train, y_pred_linear_train, edgecolors=(0, 0, 0))
    ax_train.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--', lw=2)
    ax_train.set_xlabel('Giá trị thực tế (Tập huấn luyện)')
    ax_train.set_ylabel('Giá trị dự đoán (Tập huấn luyện)')
    ax_train.set_title('So sánh giá trị thực tế và dự đoán - Tập huấn luyện')

    st.subheader("Biểu đồ Tập Huấn Luyện")
    plt.tight_layout()
    st.pyplot(fig_train)

    # Vẽ biểu đồ cho tập xác thực
    fig_val, ax_val = plt.subplots()
    ax_val.scatter(y_test, y_pred_linear_test, edgecolors=(0, 0, 0))
    ax_val.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    ax_val.set_xlabel('Giá trị thực tế (Tập xác thực)')
    ax_val.set_ylabel('Giá trị dự đoán (Tập xác thực)')
    ax_val.set_title('So sánh giá trị thực tế và dự đoán - Tập xác thực')

    st.subheader("Biểu đồ Tập Xác Thực")
    plt.tight_layout()
    st.pyplot(fig_val)

    # Vẽ biểu đồ cho tập kiểm tra
    fig_test, ax_test = plt.subplots()
    ax_test.scatter(y_test, y_pred_linear_test, edgecolors=(0, 0, 0))
    ax_test.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    ax_test.set_xlabel('Giá trị thực tế (Tập kiểm tra)')
    ax_test.set_ylabel('Giá trị dự đoán (Tập kiểm tra)')
    ax_test.set_title('So sánh giá trị thực tế và dự đoán - Tập kiểm tra')

    st.subheader("Biểu đồ Tập Kiểm Tra")
    plt.tight_layout()
    st.pyplot(fig_test)

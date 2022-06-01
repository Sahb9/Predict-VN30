
# In[0]: IMPORT AND FUNCTIONS
# Importing libraries for data handelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime
from turtle import color
import joblib
import pandas as pd
import numpy as np
from numpy import math

# Importing libraries for Data Visulization
import matplotlib.pyplot as plt
import seaborn as sns

# Importing Machine Learning libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings('ignore')

# Load dữ liệu vào
df = pd.read_csv("dataVN30_new.csv")

# In[1]: Data Preprocessing
# Data Preprocessing
# 5 dòng đầu tiên của dữ liệu
print(df.head(5))  # first 5 rows
# 5 dòng cuối cùng của dữ liệu
print(df.tail(5))
# Số lượng sample và feature của dữu liệu.
print(df.shape)
# chi tiết về dataset
df.info()
# print the unique value
print(df.nunique())
# Summary of dataset
print(df.describe())
# %% Chuyển đổi kiểu dữ liệu ở cột DATE thành datetime format
# Đổi từ chuỗi %m/%d/%Y thành kiểu datetime (với %m là tháng, %d là ngày và %Y là năm có 4 chữ số)
df['DATE'] = df['DATE'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
df.info()
# In[2]:Phân tích biến phụ thuộc
# Biến phụ thuộc là biến mà giá trị của nó sẽ thay đổi phụ thuộc vào giá trị của một biến khác.

# In ra sự thay đổi của giá đóng cửa theo thời gian
plt.figure(figsize=(12, 6))
plt.grid(True)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Close Price', fontsize=14)
plt.plot(df['DATE'], df['CLOSE'], color="red")
plt.title('VN30 closing price', fontsize=18)
plt.show()
# Ở đây ta thấy khoảng đầu năm 2018 giá cổ phiếu rớt xuống bởi vì vào thời điểm này
# xảy ra vụ tranh chấp ở Hoàng Sa và Trường Sa
# Và ngoài ra khoảng đầu năm 2020 lại tiếp tục giảm bởi vì đại dịch Covid19 diễn ra
# khiến cho các NDT rút vốn ra nhiều.

# %% Hiển thị ra tất cả giá mở cửa, đóng cửa, cao nhất, thấp nhất khối lượng của VN30 theo thời gian
plt.figure(figsize=(14, 7))
plt.grid(True)
plt.xlabel('Year', fontsize=14)
plt.ylabel(' Price (Volume)', fontsize=14)
plt.plot(df['DATE'], df['CLOSE'])
plt.plot(df['DATE'], df['OPEN'])
plt.plot(df['DATE'], df['LOW'])
plt.plot(df['DATE'], df['HIGH'])
plt.plot(df['DATE'], df['VOLUME'])
plt.legend(['CLOSE', 'OPEN', 'LOW', 'HIGH', 'VOLUME'])
plt.title('Combined Plot: Close, Open, Low, High Stock Price & Volume', fontsize=18)
plt.show()
# Ở đây chúng em chọn giá đóng cửa ('CLOSE' price) làm biến phụ thuộc
# Bởi vì giá đóng cửa là giá cuối cùng của một phiên.
# Và ngoài ra giá cao nhất và thấp nhất đôi khi không khách quan lắm vì một số thời điểm 2 giá này rất chênh lệch.

# %% Hiển thị sự phân phối của giá đóng cửa của VN30
plt.figure(figsize=(6, 5))
sns.distplot(df['CLOSE'], color='green')
plt.title('Distribution of Close Price', fontsize=16)
plt.xlabel('Closing Price', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.show()

# distribution plot of Close price by applying log transformation
plt.figure(figsize=(6, 5))
sns.distplot(np.log10(df['CLOSE']), color='green')
plt.title('Distribution of Close Price', fontsize=16)
plt.xlabel('Log of Closing Price', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.show()
# %% Phân tích Numerical variables
# Xóa các feature không sử dụng
df.drop(columns=["STT"], inplace=True)
# In ra tất cả các feature của dữ liệu
df.describe().columns

# Danh sách các biến độc lập (tất cả trừ giá đóng cửa)
numerical_features = list(set(df.describe().columns)-{'CLOSE'})
numerical_features

# Hiển thị ra độ phân phối của các numerical features.
for col in numerical_features:
    plt.figure(figsize=(6, 5))
    sns.distplot(df[col], color='green')
    plt.title("Distribution", fontsize=16)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Density', fontsize=12)
plt.show


# %% Regression plot
# Regression plot sẽ tạo ra một đường thẳng giữa hai tham số
# Và dùng nó để hiển thị ra quan hệ tuyến tính giữa chúng
# Ở đây chúng em sẽ hiển thị ra quan hệ giữa giá đóng cửa và các giá trị của các numerical features
for col in numerical_features:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=df[col], y=df['CLOSE'], scatter_kws={
                "color": 'orange'}, line_kws={"color": "black"})

# Hiển thị ra số điểm tương quan giữa giá đóng cửa và các numerical features
for col in numerical_features:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.gca()
    feature = df[col]
    label = df['CLOSE']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Closing Price', fontsize=12)
    ax.set_title(col + ' Vs. Close' + '         Correlation: ' +
                 str(round(correlation, 4)), fontsize=16)
    z = np.polyfit(df[col], df['CLOSE'], 1)
    y_hat = np.poly1d(z)(df[col])

    plt.plot(df[col], y_hat, "r--", lw=1)
# Ta thấy số điểm tương quan giữa giá đóng cửa và giá mở cửa, cao nhất, thấp nhất khá cao
# Điều này có nghĩa là khi giá của các numerical feature này tăng thì giá đóng cửa cũng tăng theo.

# %%Heatmap
# we check correletion betweeen variables using Correlation heatmap, it is graphical representation of correlation matrix representing correlation between different variables
# plot the Correlation matrix
plt.figure(figsize=(20, 8))
correlation = round(df.corr(), 4)
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap((correlation), mask=mask, annot=True, cmap='coolwarm')
# Lets find correlation with heatmap
plt.figure(figsize=(8, 5))
correlation = df.corr()
sns.heatmap(abs(correlation), annot=True, cmap='coolwarm')
plt.title("Correlation Map", fontsize=16)
# VIF


def calc_vif(df):
    vif = pd.DataFrame()
    vif['Variables'] = df.columns
    vif['VIF'] = [variance_inflation_factor(
        df.values, i) for i in range(df.shape[1])]
    return (vif)


calc_vif(df[[i for i in df.describe().columns if i not in ['DATE', 'CLOSE']]])
# heatmap for correlation
plt.figure(figsize=(10, 5))
sns.heatmap(df[['LOW', 'HIGH', 'OPEN', 'CLOSE']].corr(),
            annot=True, cmap='coolwarm')
plt.show()
# In[3]: Training
# Chia dữ liệu thành 2 tập
X = df.drop(columns=['CLOSE', 'DATE', 'HIGH', 'LOW'])
# truyền vào giá mở cửa
# dự đoán ra giá cao nhất, thấp nhất, đóng cửa
y = df.drop(columns=['OPEN', 'DATE', 'VOLUME'])

# Data transformation
X = X.apply(zscore)
y = np.log10(y)

# Chia dữ liệu thành 2 tập train và test với tỉ lệ 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=False)
# Hiển thị ra mô tả của 2 tập
print('-------------Tập train--------------')
print(X_train.shape)
print('-------------Tập test--------------')
print(X_test.shape)
# %%LINEAR REGRESSION
print("LINEAR REGRESSION")
# import the packages
reg = LinearRegression().fit(X_train, y_train)
# check the score
reg.score(X_train, y_train)
reg.score(X_test, y_test)
# check the coefficeint
# reg.coef_
# get the X_train and X-test value
y_pred_train = reg.predict(X_train)
y_pred_test = reg.predict(X_test)
# import the packages
# %% Tính toán RMSE trên tập train
MSE_lr = mean_squared_error((y_train), (y_pred_train))
print("MSE :", 10**MSE_lr)

# calculate RMSE
RMSE_lr = np.sqrt(MSE_lr)
print("RMSE :", 10**RMSE_lr)

# import the packages
# calculate r2 and adjusted r2
r2_lr = r2_score(y_train, y_pred_train)
print("R2 :", r2_lr)
# Lưu lại các điểm để so sánh
dict1 = {'Model': 'Linear regression ',
         'MSE': round((10**MSE_lr), 3),
         'RMSE': round((10**RMSE_lr), 3),
         'R2_score': round((r2_lr), 3),
         }
training_df = pd.DataFrame(dict1, index=[1])

# %% Tính toán RMSE và R2 ở tập test
# calculate MSE
MSE_lr = mean_squared_error(y_test, y_pred_test)
print("MSE :", 10**MSE_lr)

# calculate RMSE
RMSE_lr = np.sqrt(MSE_lr)
print("RMSE :", 10**RMSE_lr)

# import the packages
# calculate r2 and adjusted r2
r2_lr = r2_score((y_test), (y_pred_test))
print("R2 :", r2_lr)

dict2 = {'Model': 'Linear regression ',
         'MSE': round((10**MSE_lr), 3),
         'RMSE': round((10**RMSE_lr), 3),
         'R2_score': round((r2_lr), 3),
         }
test_df = pd.DataFrame(dict2, index=[1])
# %% Heteroscadacity
plt.scatter(10**(y_pred_test), 10**((y_test)-(y_pred_test)))
# %%Actual Price vs. Predicted Price for Linear Regression Plot
plt.figure(figsize=(10, 5))
plt.plot(10**(np.array(y_test['CLOSE'])))
plt.plot(10**(y_pred_test[:, 0]))
plt.suptitle('Actual Vs. Predicted Close Price: Linear Regression', fontsize=16)
plt.legend(['Actual', 'Predicted'], fontsize=12)
plt.xlabel('No of Test Data', fontsize=12)
plt.ylabel('Closing Price', fontsize=12)
plt.grid()
# %% RIDGE REGRESSION
print("RIDGE REGRESSION")
run = 1
# import the packages

if run == 0:
    ridge = joblib.load('models/Ridge_Regression_model.pkl')
    y_pred_train_ridge = joblib.load('saved_pred/y_pred_train_ridge.pkl')
    y_pred_test_ridge = joblib.load('saved_pred/y_pred_test_ridge.pkl')
    print(ridge.score(X_train, y_train))
else:
    ridge = Ridge(alpha=0.1)
    # Fit để train model
    ridge.fit(X_train, y_train)
    # Lưu model
    joblib.dump(ridge, r'models/Ridge_Regression_model.pkl')
    # Kiểm tra điểm của tập train
    print(ridge.score(X_train, y_train))
    # Lấy ra kết quả tập train và tập test
    y_pred_train_ridge = ridge.predict(X_train)
    y_pred_test_ridge = ridge.predict(X_test)
    # Lưu dự đoán
    joblib.dump(y_pred_train_ridge, r'saved_pred/y_pred_train_ridge.pkl')
    joblib.dump(y_pred_test_ridge, r'saved_pred/y_pred_test_ridge.pkl')


# %% Kiểm tra số điểm trên tập train
# import the packages
# Tính toán MSE cho Ridge
MSE_r = mean_squared_error((y_train), (y_pred_train_ridge))
print("MSE :", 10**MSE_r)

# Tính toán RMSE cho Ridge
RMSE_r = np.sqrt(MSE_r)
print("RMSE :", 10 ** RMSE_r)


# import the packages
# Tính toán R2 Score
r2_r = r2_score(y_train, y_pred_train_ridge)
print("R2 :", r2_r)
# Tinh chỉnh lại điểm R2
# Ta thấy điểm R2 là 0.99 khá tốt. Có nghĩa là model của chúng ta dự đoán đúng hầu hết dữ liệu ở tập train

# Lưu số điểm lại để so sánh với các model khác
dict1 = {'Model': 'Ridge regression ',
         'MSE': round((10**MSE_r), 3),
         'RMSE': round((10**RMSE_r), 3),
         'R2_score': round((r2_r), 3),
         }
training_df = training_df.append(dict1, ignore_index=True)

# %% Kiểm tra trên tập test với Ridge Regression

MSE_r = mean_squared_error(y_test, y_pred_test_ridge)
print("MSE :", 10 ** MSE_r)

RMSE_r = np.sqrt(MSE_r)
print("RMSE :", 10 ** RMSE_r)


r2_r = r2_score((y_test), (y_pred_test_ridge))
print("R2 :", r2_r)
# Ở tập test ta thấy R2 score cũng khác cao và tương đương với tập train
# Từ đây có thể thấy được model này khá tốt

# Lưu dữ liệu điểm của tập test lại để so sánh với các model khác
dict2 = {'Model': 'Ridge regression ',
         'MSE': round((10**MSE_r), 3),
         'RMSE': round((10**RMSE_r), 3),
         'R2_score': round((r2_r), 3)
         }

test_df = test_df.append(dict2, ignore_index=True)


# %%Ridge Regression: Actual Vs. Predicted
# In ra đồ thị so sánh giữa giá thực tế và giá mà Ridge Regression dự đoán
# Giá đóng cửa
plt.figure(figsize=(10, 5))
plt.plot(10**(np.array(y_test['CLOSE'])))
plt.plot(10**(y_pred_test_ridge[:, 0]))
plt.suptitle('Actual Vs. Predicted Close Price: Ridge Regression', fontsize=16)
plt.legend(['Actualn', 'Predicted'], fontsize=12)
plt.xlabel('No of Test Data', fontsize=12)
plt.ylabel('Closing Price', fontsize=12)
plt.grid()


# %% Đồ thị mô tả sự chênh lệch giữa tập dự đoán và thực tế
plt.scatter(10**(y_pred_test_ridge), 10**((y_test)-(y_pred_test_ridge)))
# %%  ELASTIC NET REGRESSION
print("ELASTIC NET REGRESSION")

# import the packages
#a * L1 + b * L2
# alpha = a + b and l1_ratio = a / (a + b)
elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.1)
# FIT THE MODEL
elasticnet.fit(X_train, y_train)
# check the score
elasticnet.score(X_train, y_train)
# get the X_train and X-test value
y_pred_train_en = elasticnet.predict(X_train)
y_pred_test_en = elasticnet.predict(X_test)
# import the packages
# calculate MSE
MSE_e = mean_squared_error((y_train), (y_pred_train_en))
print("MSE :", 10 ** MSE_e)

# calculate RMSE
RMSE_e = np.sqrt(MSE_e)
print("RMSE :", 10**RMSE_e)

# import the packages
# calculate r2 and adjusted r2
r2_e = r2_score(y_train, y_pred_train_en)
print("R2 :", r2_e)

# storing the test set metrics value in a dataframe for later comparison
dict1 = {'Model': 'Elastic net regression ',
         'MSE': round((10**MSE_e), 3),
         'RMSE': round((10**RMSE_e), 3),
         'R2_score': round((r2_e), 3),
         }

training_df = training_df.append(dict1, ignore_index=True)
# import the packages
# calculate MSE
MSE_e = mean_squared_error(y_test, y_pred_test_en)
print("MSE :", 10 ** MSE_e)

# calculate RMSE
RMSE_e = np.sqrt(MSE_e)
print("RMSE :", 10**RMSE_e)

# import the packages
# calculate r2 and adjusted r2
r2_e = r2_score((y_test), (y_pred_test_en))
print("R2 :", r2_e)
# storing the test set metrics value in a dataframe for later comparison
dict2 = {'Model': 'Elastic net regression Test',
         'MSE': round((10**MSE_e), 3),
         'RMSE': round((10**RMSE_e), 3),
         'R2_score': round((r2_e), 3),
         }

test_df = test_df.append(dict2, ignore_index=True)
# Actual Price vs. Predicted Price for Elastic Net Plotting
plt.figure(figsize=(10, 5))
plt.plot(10**(np.array(y_test['CLOSE'])))
plt.plot(10**(y_pred_test_en[:, 0]))
plt.suptitle('Actual Vs. Predicted Close Price: Elastic Net', fontsize=16)
plt.legend(['Actual', 'Predicted'], fontsize=12)
plt.xlabel('No of Test Data', fontsize=12)
plt.ylabel('Closing Price', fontsize=12)
plt.grid()
# %% Heteroscadacity
plt.scatter((y_pred_test_en), (y_test)-(y_pred_test_en))
# %%RANDOM FOREST
print("RANDOM FOREST")
# import the packages
# Create an instance of the RandomForestRegressor
rf_model = RandomForestRegressor()

rf_model.fit(X_train, y_train)
# Making predictions on train and test data

y_pred_train_r = rf_model.predict(X_train)
y_pred_test_r = rf_model.predict(X_test)

# %% TÍnh toán trên tập train
print("Model Score:", rf_model.score(X_train, y_train))

MSE_rf = mean_squared_error(y_train, y_pred_train_r)
print("MSE :", 10**MSE_rf)

RMSE_rf = np.sqrt(MSE_rf)
print("RMSE :", 10**RMSE_rf)


r2_rf = r2_score(y_train, y_pred_train_r)
print("R2 :", r2_rf)

dict1 = {'Model': 'Random forest regression ',
         'MSE': round((10**MSE_rf), 3),
         'RMSE': round((10**RMSE_rf), 3),
         'R2_score': round((r2_rf), 3),
         }
training_df = training_df.append(dict1, ignore_index=True)

# %% Tính toán tập test

MSE_rf = mean_squared_error(y_test, y_pred_test_r)
print("MSE :", 10 ** MSE_rf)

RMSE_rf = np.sqrt(MSE_rf)
print("RMSE :", 10 ** RMSE_rf)

r2_rf = r2_score((y_test), (y_pred_test_r))
print("R2 :", r2_rf)


dict2 = {'Model': 'Random forest regression ',
         'MSE': round((10**MSE_rf), 3),
         'RMSE': round((10**RMSE_rf), 3),
         'R2_score': round((r2_rf), 3)
         }

# %% Đồ thị giá đóng cửa giữa dự đoán và thực tế
plt.figure(figsize=(10, 5))
plt.plot(10**(np.array(y_test['CLOSE'])))
plt.plot(10**(y_pred_test_r[:, 0]))
plt.suptitle('Actual Vs. Predicted Close Price: Random Forest', fontsize=16)
plt.legend(['Actual', 'Predicted'], fontsize=12)
plt.xlabel('No of Test Data', fontsize=12)
plt.ylabel('Closing Price', fontsize=12)
plt.grid()
# %%
test_df = test_df.append(dict2, ignore_index=True)


rf_model.feature_importances_
importances = rf_model.feature_importances_

importance_dict = {'Feature': list(X_train.columns),
                   'Feature Importance': importances}

importance_df = pd.DataFrame(importance_dict)
importance_df['Feature Importance'] = round(
    importance_df['Feature Importance'], 2)
importance_df.sort_values(by=['Feature Importance'], ascending=False)
# FIT THE MODEL
rf_model.fit(X_train, y_train)
features = X_train.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)
# Plot the figure
plt.figure(figsize=(5, 10))
plt.title('Feature Importance')
plt.barh(range(len(indices)),
         importances[indices], color='blue', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')

plt.show()
# %%Hyperparameter tuning
# Using GridSearchCV
# Gradient Boosting Regressor with GridSearchCV
# Importing Gradient Boosting Regressor
# Hyper-Parameter Tuning for Elastic Net
# Create an instance of the ElasticNet Regression
print("Hyperparameter tuning")
elastic_cv = ElasticNet()
parameters = {'alpha': [0.1, 1e-15, 1e-13, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5,
                        10, 20, 30, 40, 45, 50, 55, 60, 100], 'l1_ratio': [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2]}
# -l1_ratio:  Đây được gọi là tham số trộn ElasticNet. Phạm vi của nó là 0 <= l1_ratio <= 1.
# -alpha, hằng số nhân số hạng L1 / L2, là tham số điều chỉnh quyết định mức độ chúng ta muốn penalty model. Giá trị mặc định là 1,0.
elastic_model_cv = GridSearchCV(
    elastic_cv, parameters, scoring='neg_mean_squared_error', cv=5)
elastic_model_cv.fit(X_train, y_train)
# Grid search
et_grid = GridSearchCV(elastic_cv, parameters,
                       scoring='neg_mean_squared_error', cv=5)
et_grid.fit(X_train, y_train)
et_grid.best_estimator_
et_optimal_model = et_grid.best_estimator_
et_grid.best_params_
# Making predictions on train and test data

y_pred_train_g_g = et_optimal_model.predict(X_train)
y_pred_g_g = et_optimal_model.predict(X_test)
print("Model Score:", et_optimal_model.score(X_train, y_train))
MSE_gbh = mean_squared_error(y_train, y_pred_train_g_g)
print("MSE :", 10**MSE_gbh)

RMSE_gbh = np.sqrt(MSE_gbh)
print("RMSE :", 10 ** RMSE_gbh)

r2_gbh = r2_score(y_train, y_pred_train_g_g)
print("R2 :", r2_gbh)
# storing the test set metrics value in a dataframe for later comparison
dict1 = {'Model': 'ElasticNet gridsearchcv ',
         'MSE': round((10**MSE_gbh), 3),
         'RMSE': round((10**RMSE_gbh), 3),
         'R2_score': round((r2_gbh), 3),
         }
training_df = training_df.append(dict1, ignore_index=True)
MSE_gbh = mean_squared_error(y_test, y_pred_g_g)
print("MSE :", 10**MSE_gbh)

RMSE_gbh = np.sqrt(MSE_gbh)
print("RMSE :", 10**RMSE_gbh)

r2_gbh = r2_score((y_test), (y_pred_g_g))
print("R2 :", r2_gbh)
# storing the test set metrics value in a dataframe for later comparison
dict2 = {'Model': 'ElacticNet gridsearchcv ',
         'MSE': round((10**MSE_gbh), 3),
         'RMSE': round((10**RMSE_gbh), 3),
         'R2_score': round((r2_gbh), 3),
         }
test_df = test_df.append(dict2, ignore_index=True)
# Evaluation Matrics Comparison
# %% displaying the results of evaluation metric values for all models
result = pd.concat([training_df, test_df], keys=['Training set', 'Test set'])
result
# %%

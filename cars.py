import umap
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
#Zdroje: Seminar 1 (cvicenie1.py), Seminar 2 (seminar2.py), Seminar 3 (main.py), ChatGPT

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('zadanie2_dataset.csv')

df = df[df['Price'] >= 950]
df = df[df['Price'] <= 500000]
df = df[df['Prod. year'] >= 1950]
df = df[df['Engine volume'] >= 0]
df = df[df['Cylinders'] >= 0]
df = df[df['Airbags'] >= 0]

print("*"*20, "After removing outliers", "*"*20)
print("-"*10, "Min", "-"*10)
print(df.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(df.max(numeric_only=True))
levy_oszlop=df['Levy']
df['Levy'] = [obj if obj == "-" else int(obj) for obj in levy_oszlop]
df['Levy'] = np.where(df['Levy'] == "-", np.nan, df['Levy'])
df['Levy'] = df['Levy'].astype(float)
print("*"*20, "Missing values", "*"*20)
print(f"Lenght of dataset: {len(df)}")

print(df.isnull().sum())

print("*"*20, "Column types", "*"*20)
print(df.dtypes)

print("*"*20, "Before removing ID", "*"*20)
print(df.columns)

df = df.drop(columns=['ID'])

print("*"*20, "After removing ID", "*"*20)
print(df.columns)

print("*"*20, "Before drop duplicate", "*"*20)
print(f"Lenght of dataset: {len(df)}")

df = df.drop_duplicates()

print("*"*20, "After drop dupicate", "*"*20)
print(f"Lenght of dataset: {len(df)}")

print(f"Mileage data type: {df['Mileage'].dtype}")
df['Mileage'] = df['Mileage'].str.replace(' km', '').astype(int)
print(f"Mileage data type: {df['Mileage'].dtype}")
df = df[df['Manufacturer'] != 'სხვა']

df_with_topgenre = df[df['Fuel type'].notnull()]

emotions_df = df_with_topgenre['Fuel type']


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

manufacture = df['Prod. year']
category = df['Engine volume']
fuel_type = df['Cylinders']
price = df['Price']
scatter = ax.scatter(manufacture, category, fuel_type, c=price, cmap='viridis')

ax.set_xlabel('Prod. year')
ax.set_ylabel('Engine volume')
ax.set_zlabel('Cylinders')
ax.set_title('3D Scatter Plot - Price vs Prod. year, Engine volume, Cylinders')

colorbar = plt.colorbar(scatter)
colorbar.set_label('Price')
plt.show()

filtered_df = df[df['Price'] <= 20000000]

fig = px.scatter_3d(filtered_df, x='Prod. year', y='Engine volume', z='Cylinders', color='Price', color_continuous_scale='Viridis', title='3D Scatter Plot - Price vs Prod. year, Engine volume, Cylinders')

fig.update_layout(scene=dict(xaxis_title='Prod. year', yaxis_title='Engine volume', zaxis_title='Cylinders'))
fig.show()

dummies = pd.get_dummies(df['Manufacturer'])
dummies = dummies.astype(int)
df = pd.concat([df.drop('Manufacturer', axis=1), dummies], axis=1)

dummies = pd.get_dummies(df['Category'])
dummies = dummies.astype(int)
df = pd.concat([df.drop('Category', axis=1), dummies], axis=1)

dummies = pd.get_dummies(df['Leather interior'])
dummies = dummies.astype(int)
df = pd.concat([df.drop('Leather interior', axis=1), dummies], axis=1)

dummies = pd.get_dummies(df['Fuel type'])
dummies = dummies.astype(int)
df = pd.concat([df.drop('Fuel type', axis=1), dummies], axis=1)

dummies = pd.get_dummies(df['Gear box type'])
dummies = dummies.astype(int)
df = pd.concat([df.drop('Gear box type', axis=1), dummies], axis=1)
print(df)
df.drop(columns=['Model'], inplace=True)
df.drop(columns=['Levy'], inplace=True)
df.drop(columns=['Drive wheels'], inplace=True)
df.drop(columns=['Doors'], inplace=True)
df.drop(columns=['Color'], inplace=True)
df.drop(columns=['Turbo engine'], inplace=True)
df.drop(columns=['Left wheel'], inplace=True)

X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

decision_tree_model = DecisionTreeRegressor(max_depth=15, min_samples_split=4, min_samples_leaf=4, random_state=42)
decision_tree_model.fit(X_train, y_train)
y_pred = decision_tree_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Decision tree")
print(f"R2 Score: {r2}")

plt.figure(figsize=(15, 10))
plot_tree(decision_tree_model, feature_names=X.columns, filled=True)
plt.show()

y_train_pred_rf = decision_tree_model.predict(X_train)
y_test_pred_rf = decision_tree_model.predict(X_test)

mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)
mse_test_rf = mean_squared_error(y_test, y_test_pred_rf)

rmse_train_rf = sqrt(mse_train_rf)
rmse_test_rf = sqrt(mse_test_rf)

r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
print("*"*50)
print("Decision Tee Regression - Train")
print(f"Mean Squared Error: {mse_train_rf}")
print(f"Root Mean Squared Error: {rmse_train_rf}")
print(f"R2 Score: {r2_train_rf}")

print("Decision Tee Regression - Test")
print(f"Mean Squared Error: {mse_test_rf}")
print(f"Root Mean Squared Error: {rmse_test_rf}")
print(f"R2 Score: {r2_test_rf}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred_rf, y_train_pred_rf - y_train, c='blue', marker='o', label='Trénovacia množina')
plt.scatter(y_test_pred_rf, y_test_pred_rf - y_test, c='lightgreen', marker='s', label='Testovacia množina')
plt.xlabel('Predikované hodnoty')
plt.ylabel('Reziduály')
plt.legend(loc='upper left')
plt.title('Decision Tee Regression - Reziduálne hodnoty')

plt.subplot(1, 2, 2)
plt.scatter(y_train_pred_rf, y_train, c='blue', marker='o', label='Trénovacia množina')
plt.scatter(y_test_pred_rf, y_test, c='lightgreen', marker='s', label='Testovacia množina')
plt.xlabel('Predikované hodnoty')
plt.ylabel('Skutočné hodnoty')
plt.legend(loc='upper left')
plt.title('Decision Tee Regression - Skutočné vs. Predikované hodnoty')

plt.tight_layout()
plt.show()

random_forest_model = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=2, min_samples_leaf=2, random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Bagging random forest")
print(f"R2 Score: {r2}")

single_tree = random_forest_model.estimators_[0]

plt.figure(figsize=(15, 10))
plot_tree(single_tree, feature_names=X.columns, filled=True)
plt.show()

y_train_pred_rf = random_forest_model.predict(X_train)
y_test_pred_rf = random_forest_model.predict(X_test)

mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)
mse_test_rf = mean_squared_error(y_test, y_test_pred_rf)

rmse_train_rf = sqrt(mse_train_rf)
rmse_test_rf = sqrt(mse_test_rf)

r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
print("*"*50)
print("Random Forest Regression - Train")
print(f"Mean Squared Error: {mse_train_rf}")
print(f"Root Mean Squared Error: {rmse_train_rf}")
print(f"R2 Score: {r2_train_rf}")

print("Random Forest Regression - Test")
print(f"Mean Squared Error: {mse_test_rf}")
print(f"Root Mean Squared Error: {rmse_test_rf}")
print(f"R2 Score: {r2_test_rf}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train_pred_rf, y_train_pred_rf - y_train, c='blue', marker='o', label='Trénovacia množina')
plt.scatter(y_test_pred_rf, y_test_pred_rf - y_test, c='lightgreen', marker='s', label='Testovacia množina')
plt.xlabel('Predikované hodnoty')
plt.ylabel('Reziduály')
plt.legend(loc='upper left')
plt.title('Random Forest Regression - Reziduálne hodnoty')

plt.subplot(1, 2, 2)
plt.scatter(y_train_pred_rf, y_train, c='blue', marker='o', label='Trénovacia množina')
plt.scatter(y_test_pred_rf, y_test, c='lightgreen', marker='s', label='Testovacia množina')
plt.xlabel('Predikované hodnoty')
plt.ylabel('Skutočné hodnoty')
plt.legend(loc='upper left')
plt.title('Random Forest Regression - Skutočné vs. Predikované hodnoty')

plt.tight_layout()
plt.show()

feature_importance = random_forest_model.feature_importances_
feature_names = X.columns
top_n = 7
sorted_idx = np.argsort(feature_importance)[::-1][:top_n]
selected_features = [feature_names[i] for i in sorted_idx]
filtered_feature_importance = [feature_importance[i] for i in sorted_idx]
plt.figure(figsize=(12, 6))
plt.bar(selected_features, filtered_feature_importance,color='pink')
plt.xlabel('Vstupné parametre')
plt.ylabel('Dôležitosť')
plt.title('Dôležitosť vstupných parametrov pre Random Forest (Top 7)')
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.3)
plt.show()

svm_model = SVR(kernel='rbf', C=1000, epsilon=0.1,gamma=0.8)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("SVM")
print(f"R2 Score: {r2}")
y_train_pred_svm = svm_model.predict(X_train)
y_test_pred_svm = svm_model.predict(X_test)
r2_train_svm = r2_score(y_train, y_train_pred_svm)
r2_test_svm = r2_score(y_test, y_test_pred_svm)
print("*" * 50)
print("SVM Regression - Train")
print(f"R2 Score: {r2_train_svm}")

print("SVM Regression - Test")
print(f"R2 Score: {r2_test_svm}")

y_train_pred_rf = svm_model.predict(X_train)
y_test_pred_rf = svm_model.predict(X_test)

mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)
mse_test_rf = mean_squared_error(y_test, y_test_pred_rf)

rmse_train_rf = sqrt(mse_train_rf)
rmse_test_rf = sqrt(mse_test_rf)

r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
print("*"*50)
print("SVM Regression - Train")
print(f"Mean Squared Error: {mse_train_rf}")
print(f"Root Mean Squared Error: {rmse_train_rf}")
print(f"R2 Score: {r2_train_rf}")

print("SVM Regression - Test")
print(f"Mean Squared Error: {mse_test_rf}")
print(f"Root Mean Squared Error: {rmse_test_rf}")
print(f"R2 Score: {r2_test_rf}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train_pred_rf, y_train_pred_rf - y_train, c='blue', marker='o', label='Trénovacia množina')
plt.scatter(y_test_pred_rf, y_test_pred_rf - y_test, c='lightgreen', marker='s', label='Testovacia množina')
plt.xlabel('Predikované hodnoty')
plt.ylabel('Reziduály')
plt.legend(loc='upper left')
plt.title('SVM Regression - Reziduálne hodnoty')

plt.subplot(1, 2, 2)
plt.scatter(y_train_pred_rf, y_train, c='blue', marker='o', label='Trénovacia množina')
plt.scatter(y_test_pred_rf, y_test, c='lightgreen', marker='s', label='Testovacia množina')
plt.xlabel('Predikované hodnoty')
plt.ylabel('Skutočné hodnoty')
plt.legend(loc='upper left')
plt.title('SVM Regression - Skutočné vs. Predikované hodnoty')

plt.tight_layout()
plt.show()
print(df.columns)

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

umap_reducer = umap.UMAP(n_components=3)
X_umap = umap_reducer.fit_transform(X_pca)
reduced_df = pd.DataFrame(X_umap, columns=['Prod. year', 'Engine volume', 'Cylinders'])
reduced_df['Price'] = df['Price']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

manufacture = reduced_df['Prod. year']
category = reduced_df['Engine volume']
fuel_type = reduced_df['Cylinders']
price = df['Price']

scatter = ax.scatter(manufacture, category, fuel_type, c=price, cmap='viridis')

ax.set_xlabel('Prod. year')
ax.set_ylabel('Engine volume')
ax.set_zlabel('Cylinders')
ax.set_title('3D Scatter Plot - Price vs Prod. year, Engine volume, Cylinders')

colorbar = plt.colorbar(scatter)
colorbar.set_label('Price')
plt.show()

filtered_df = reduced_df[reduced_df['Price'] <= 20000000]

fig = px.scatter_3d(filtered_df, x='Prod. year', y='Engine volume', z='Cylinders', color='Price', color_continuous_scale='Viridis', title='3D Scatter Plot - Price vs Prod. year, Engine volume, Cylinders')

fig.update_layout(scene=dict(xaxis_title='Prod. year', yaxis_title='Engine volume', zaxis_title='Cylinders'))
fig.show()

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
correlation_values = correlation_matrix.abs().unstack()
highly_correlated = correlation_values[(correlation_values > 0.65) & (correlation_values < 1)] #?
print(highly_correlated)
selected_features = ['Price', 'Cylinders', 'Engine volume','Automatic', 'Tiptronic']
df1 = df[selected_features]
X1 = df1.drop(columns=['Price'])
y1 = df1['Price']
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.1, random_state=45)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

random_forest_model = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=2, min_samples_leaf=2, random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Bagging random forest")
print(f"R2 Score: {r2}")

single_tree = random_forest_model.estimators_[0]

plt.figure(figsize=(15, 10))
plot_tree(single_tree, feature_names=X.columns, filled=True)
plt.show()

y_train_pred_rf = random_forest_model.predict(X_train)
y_test_pred_rf = random_forest_model.predict(X_test)

mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)
mse_test_rf = mean_squared_error(y_test, y_test_pred_rf)

rmse_train_rf = sqrt(mse_train_rf)
rmse_test_rf = sqrt(mse_test_rf)

r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
print("*"*50)
print("Random Forest Regression - Train")
print(f"Mean Squared Error: {mse_train_rf}")
print(f"Root Mean Squared Error: {rmse_train_rf}")
print(f"R2 Score: {r2_train_rf}")

print("Random Forest Regression - Test")
print(f"Mean Squared Error: {mse_test_rf}")
print(f"Root Mean Squared Error: {rmse_test_rf}")
print(f"R2 Score: {r2_test_rf}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train_pred_rf, y_train_pred_rf - y_train, c='blue', marker='o', label='Trénovacia množina')
plt.scatter(y_test_pred_rf, y_test_pred_rf - y_test, c='lightgreen', marker='s', label='Testovacia množina')
plt.xlabel('Predikované hodnoty')
plt.ylabel('Reziduály')
plt.legend(loc='upper left')
plt.title('Random Forest Regression - Reziduálne hodnoty')

plt.subplot(1, 2, 2)
plt.scatter(y_train_pred_rf, y_train, c='blue', marker='o', label='Trénovacia množina')
plt.scatter(y_test_pred_rf, y_test, c='lightgreen', marker='s', label='Testovacia množina')
plt.xlabel('Predikované hodnoty')
plt.ylabel('Skutočné hodnoty')
plt.legend(loc='upper left')
plt.title('Random Forest Regression - Skutočné vs. Predikované hodnoty')

plt.tight_layout()
plt.show()

selected_features = ['Price', 'Prod. year', 'Engine volume','Mileage']
df2 = df[selected_features]
X2 = df2.drop(columns=['Price'])
y2 = df2['Price']
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.1, random_state=45)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

random_forest_model = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=2, min_samples_leaf=2, random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Bagging random forest")
print(f"R2 Score: {r2}")

single_tree = random_forest_model.estimators_[0]

plt.figure(figsize=(15, 10))
plot_tree(single_tree, feature_names=X.columns, filled=True)
plt.show()

y_train_pred_rf = random_forest_model.predict(X_train)
y_test_pred_rf = random_forest_model.predict(X_test)

mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)
mse_test_rf = mean_squared_error(y_test, y_test_pred_rf)

rmse_train_rf = sqrt(mse_train_rf)
rmse_test_rf = sqrt(mse_test_rf)

r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
print("*"*50)
print("Random Forest Regression - Train")
print(f"Mean Squared Error: {mse_train_rf}")
print(f"Root Mean Squared Error: {rmse_train_rf}")
print(f"R2 Score: {r2_train_rf}")

print("Random Forest Regression - Test")
print(f"Mean Squared Error: {mse_test_rf}")
print(f"Root Mean Squared Error: {rmse_test_rf}")
print(f"R2 Score: {r2_test_rf}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train_pred_rf, y_train_pred_rf - y_train, c='blue', marker='o', label='Trénovacia množina')
plt.scatter(y_test_pred_rf, y_test_pred_rf - y_test, c='lightgreen', marker='s', label='Testovacia množina')
plt.xlabel('Predikované hodnoty')
plt.ylabel('Reziduály')
plt.legend(loc='upper left')
plt.title('Random Forest Regression - Reziduálne hodnoty')

plt.subplot(1, 2, 2)
plt.scatter(y_train_pred_rf, y_train, c='blue', marker='o', label='Trénovacia množina')
plt.scatter(y_test_pred_rf, y_test, c='lightgreen', marker='s', label='Testovacia množina')
plt.xlabel('Predikované hodnoty')
plt.ylabel('Skutočné hodnoty')
plt.legend(loc='upper left')
plt.title('Random Forest Regression - Skutočné vs. Predikované hodnoty')

plt.tight_layout()
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X_scaled = StandardScaler().fit_transform(X)
pca.fit(X_scaled)
explained_variance = pca.explained_variance_
components = pca.components_
component_weights = np.abs(components)

selected_features = [X.columns[i] for i in range(len(X.columns)) if any(component_weights[:, i] > 0.1)]

print("Variancie: ")
print(explained_variance)
print("Vybrané prźnaky")
print(selected_features)

selected_features = ['Price','Prod. year', 'Engine volume', 'Cylinders', 'Airbags', 'BMW', 'HONDA', 'HYUNDAI', 'MERCEDES-BENZ', 'OPEL', 'SSANGYONG', 'TOYOTA', 'Coupe', 'Goods wagon', 'Hatchback', 'Jeep', 'Microbus', 'Minivan', 'Sedan', 'CNG', 'Diesel', 'Hybrid', 'Petrol', 'Automatic', 'Manual', 'Tiptronic', 'Variator']
df3 = df[selected_features]
X3 = df3.drop(columns=['Price'])
y3 = df3['Price']
X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.1, random_state=45)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

random_forest_model = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=2, min_samples_leaf=2, random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Bagging random forest")
print(f"R2 Score: {r2}")

single_tree = random_forest_model.estimators_[0]

plt.figure(figsize=(15, 10))
plot_tree(single_tree, feature_names=X.columns, filled=True)
plt.show()

y_train_pred_rf = random_forest_model.predict(X_train)
y_test_pred_rf = random_forest_model.predict(X_test)

mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)
mse_test_rf = mean_squared_error(y_test, y_test_pred_rf)

rmse_train_rf = sqrt(mse_train_rf)
rmse_test_rf = sqrt(mse_test_rf)

r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)
print("*"*50)
print("Random Forest Regression - Train")
print(f"Mean Squared Error: {mse_train_rf}")
print(f"Root Mean Squared Error: {rmse_train_rf}")
print(f"R2 Score: {r2_train_rf}")

print("Random Forest Regression - Test")
print(f"Mean Squared Error: {mse_test_rf}")
print(f"Root Mean Squared Error: {rmse_test_rf}")
print(f"R2 Score: {r2_test_rf}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train_pred_rf, y_train_pred_rf - y_train, c='blue', marker='o', label='Trénovacia množina')
plt.scatter(y_test_pred_rf, y_test_pred_rf - y_test, c='lightgreen', marker='s', label='Testovacia množina')
plt.xlabel('Predikované hodnoty')
plt.ylabel('Reziduály')
plt.legend(loc='upper left')
plt.title('Random Forest Regression - Reziduálne hodnoty')

plt.subplot(1, 2, 2)
plt.scatter(y_train_pred_rf, y_train, c='blue', marker='o', label='Trénovacia množina')
plt.scatter(y_test_pred_rf, y_test, c='lightgreen', marker='s', label='Testovacia množina')
plt.xlabel('Predikované hodnoty')
plt.ylabel('Skutočné hodnoty')
plt.legend(loc='upper left')
plt.title('Random Forest Regression - Skutočné vs. Predikované hodnoty')

plt.tight_layout()
plt.show()

df = pd.read_csv('zadanie2_dataset.csv')

turbo_counts = df[df['Turbo engine'] == True]['Prod. year'].value_counts().sort_index()
plt.figure(figsize=(8,4))
plt.bar(turbo_counts.index, turbo_counts.values, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of cars with turbo')
plt.title('Number of cars with turbo by years')
plt.xticks(rotation=45)
plt.show()

fuel_type_counts = df['Fuel type'].value_counts()
plt.figure(figsize=(8,4))
fuel_type_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Fuel Type')
plt.ylabel('Frequency')
plt.title('Fuel Type Frequency Distribution')
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.3)
plt.show()

manufacturer_counts = df['Manufacturer'].value_counts()
wordcloud_text = ' '.join([f'{manufacturer}: {count}' for manufacturer, count in manufacturer_counts.items()])
wordcloud = WordCloud(width=800, height=800, background_color='white').generate(wordcloud_text)
plt.figure(figsize=(8, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

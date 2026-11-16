import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Create a synthetic dataset (Area in sq. ft., BHK, Price in ₹ Lakhs)
data = {
    'area_sqft': [
        650,
        800,
        950,
        1100,
        1200,
        1350,
        1500,
        1650,
        1800,
        2000,
        2200,
        2500,
    ],
    'bhk': [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4],
    'price_lakhs': [42, 55, 64, 72, 78, 92, 105, 115, 128, 142, 160, 185],
}

df = pd.DataFrame(data)

# 2. Separate features (X) and target variable (y)
X = df[['area_sqft', 'bhk']]
y = df['price_lakhs']

# 3. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model on unseen test data
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Model Accuracy (R² Score): {r2:.2f}')
print(f'Average Error (MAE): ₹{mae:.2f} Lakhs\n')

# 6. Predict price for a new, unseen flat (e.g., 1400 sq. ft., 3 BHK)
new_flat = pd.DataFrame({'area_sqft': [1400], 'bhk': [3]})
predicted_price = model.predict(new_flat)
print(
    f'Predicted price for a 1400 sq. ft., 3 BHK flat: ₹{predicted_price[0]:.2f}'
    ' Lakhs'
)

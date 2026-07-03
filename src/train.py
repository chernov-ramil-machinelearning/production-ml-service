from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from EDA import df
import numpy as np
from sklearn.model_selection import train_test_split

X = df.drop(columns=['CustomerId', 'Exited'])
y = df['Exited']
cat_features = ['Geography', 'Gender']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, shuffle=True, test_size=0.2)
model = CatBoostClassifier(
    iterations=400,
    learning_rate=0.04,
    depth=5,
    cat_features=cat_features,
    auto_class_weights='Balanced',
    verbose=0
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))

model.save_model(r'E:\python_projects\sorted_roadmap\05_production_projects\production_ml_service\model\catboost_model.cbm', format='cbm')

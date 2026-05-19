import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ===============================
# VERİYİ YÜKLE
# ===============================
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()


# ===============================
# KATEGORİK VERİLERİ SAYIYA ÇEVİR
# ===============================
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Train ve test sütunlarını eşitle
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)



# ===============================
# MODELLER
# ===============================
models = {
    "Lojistik Regresyon": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1
    )
}


# ===============================
# EĞİT + TEST
# ===============================
results = []

for name, model in models.items():
    print(f"\n[MODEL] {name} eğitiliyor...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    results.append([name, acc, prec, rec, f1, auc])


# ===============================
# SONUÇ TABLOSU
# ===============================
df_results = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1", "AUC"
])

df_results.to_csv("model_results.csv", index=False)

print("\n=== TÜM SONUÇLAR ===")
print(df_results)
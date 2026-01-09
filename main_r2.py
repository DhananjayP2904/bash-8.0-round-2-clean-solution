import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


##BASIC CLEANING
TEXT_COLS = ["Headline", "Key Insights", "Reasoning"]

for col in TEXT_COLS:
    train[col] = train[col].fillna("")
    test[col] = test[col].fillna("")

LIST_COLS = ["Lead Types", "Power Mentions", "Agencies", "Tags"]

def count_items(x):
    if pd.isna(x) or str(x).strip() == "":
        return 0
    return len([i for i in str(x).split(";") if i.strip()])

for col in LIST_COLS:
    if col in train.columns:
        train[f"num_{col.lower().replace(' ', '_')}"] = train[col].apply(count_items)
        test[f"num_{col.lower().replace(' ', '_')}"] = test[col].apply(count_items)


##FULL TEXT

train["full_text"] = (
    train["Headline"] + " " +
    train["Key Insights"] + " " +
    train["Reasoning"]
)

test["full_text"] = (
    test["Headline"] + " " +
    test["Key Insights"] + " " +
    test["Reasoning"]
)


tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,
    max_df=0.95,
    stop_words="english",
    sublinear_tf=True
)

X_tfidf_train = tfidf.fit_transform(train["full_text"])
X_tfidf_test = tfidf.transform(test["full_text"])

svd = TruncatedSVD(
    n_components=150,
    n_iter=20,
    random_state=42
)

X_svd_train = svd.fit_transform(X_tfidf_train)
X_svd_test = svd.transform(X_tfidf_test)

sbert = SentenceTransformer("all-MiniLM-L6-v2")
emb_train = sbert.encode(
    train["full_text"].tolist(),
    batch_size=64,
    show_progress_bar=True
)

emb_test = sbert.encode(
    test["full_text"].tolist(),
    batch_size=64,
    show_progress_bar=True
)

from sklearn.decomposition import PCA

pca = PCA(n_components=50, random_state=42)
X_emb_train = pca.fit_transform(emb_train)
X_emb_test = pca.transform(emb_test)

count_features = [
    "num_lead_types",
    "num_power_mentions",
    "num_agencies",
    "num_tags"
]

X_count_train = train[count_features].values
X_count_test = test[count_features].values

X_train = np.hstack([
    X_svd_train,
    X_emb_train,
    X_count_train
])

X_test = np.hstack([
    X_svd_test,
    X_emb_test,
    X_count_test
])

X_train = np.hstack([
    X_svd_train,
    X_emb_train,
    X_count_train
])

X_test = np.hstack([
    X_svd_test,
    X_emb_test,
    X_count_test
])

y = train["Importance Score"].values

from lightgbm import LGBMRegressor
lgb = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

from xgboost import XGBRegressor
xgb = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))

for tr_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    lgb.fit(X_tr, y_tr)
    xgb.fit(X_tr, y_tr)

    oof_lgb[val_idx] = lgb.predict(X_val)
    oof_xgb[val_idx] = xgb.predict(X_val)

    pred_lgb += lgb.predict(X_test) / 5
    pred_xgb += xgb.predict(X_test) / 5

stack_train = np.column_stack([oof_lgb, oof_xgb])
stack_test = np.column_stack([pred_lgb, pred_xgb])

meta = Ridge(alpha=1.0)
meta.fit(stack_train, y)
final_pred = meta.predict(stack_test)

rmse = np.sqrt(mean_squared_error(y, meta.predict(stack_train)))
print("CV RMSE:", rmse)

final_pred = np.clip(final_pred, 0, 100)

submission = pd.DataFrame({
    "id": test["id"],
    "Importance Score": final_pred
})

submission.to_csv("submission.csv", index=False)

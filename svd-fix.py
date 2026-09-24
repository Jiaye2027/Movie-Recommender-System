import pandas as pd
import numpy as np
import random
from surprise import SVD, SVDpp, BaselineOnly, NMF, Reader, Dataset
from surprise.model_selection import train_test_split, GridSearchCV
from recommenders.evaluation.python_evaluation import (
    rmse, mae, rsquared,
    map_at_k, ndcg_at_k, precision_at_k, recall_at_k
)
from recommenders.models.surprise.surprise_utils import compute_ranking_predictions

# ============================================================
# 1. Load data
# ============================================================
ratings = pd.read_csv('/kaggle/input/the-movies-dataset/ratings.csv')
ratings = ratings.rename(columns={'userId': 'userID', 'movieId': 'itemID'})

n_users  = ratings['userID'].nunique()
n_items  = ratings['itemID'].nunique()
n_ratings = len(ratings)
sparsity = 1 - n_ratings / (n_users * n_items)

print(f"Ratings: {n_ratings:,} | Users: {n_users:,} | Items: {n_items:,}")
print(f"Sparsity: {sparsity:.6f}")

# ============================================================
# 2. Surprise dataset
# ============================================================
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userID', 'itemID', 'rating']], reader=reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# ============================================================
# 3. Grid search SVD
# ============================================================
print("\nGrid search SVD ...")
param_grid = {
    'n_factors': [50, 100, 150],
    'n_epochs':  [20, 30],
    'lr_all':    [0.005, 0.01],
    'reg_all':   [0.05, 0.1]
}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
gs.fit(data)

best_params = gs.best_params['rmse']
print(f"Best CV RMSE: {gs.best_score['rmse']:.4f}")
print(f"Best params:  {best_params}")

# ============================================================
# 4. Train several models for comparison
# ============================================================
models = {
    'SVD':          SVD(**best_params, random_state=42),
    'SVD++':        SVDpp(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.05, random_state=42),
    'BaselineOnly': BaselineOnly(bsl_options={'method': 'sgd',
                                              'learning_rate': 0.005,
                                              'reg': 0.02}),
    'NMF':          NMF(n_factors=50, n_epochs=30, random_state=42),
}

predictions = {}
for name, model in models.items():
    print(f"Training {name} ...")
    model.fit(trainset)
    predictions[name] = pd.DataFrame(model.test(testset))
    rmse_val = np.sqrt(np.mean((predictions[name]['r_ui'] - predictions[name]['est'])**2))
    print(f"  {name} test RMSE = {rmse_val:.4f}")

# ============================================================
# 5. Ranking evaluation with RELEVANCE THRESHOLD
# ============================================================
TOP_K = 10
SAMPLE_SIZE = 1000
RELEVANCE_THRESHOLD = 4.0        # rating >= 4 -> relevant

all_users = ratings['userID'].unique()
sampled_users = random.sample(list(all_users), SAMPLE_SIZE)

# Pre-build train sample once (used for remove_seen)
train_df = pd.DataFrame(trainset.all_ratings(),
                        columns=['uid_inner', 'iid_inner', 'rating'])
train_df['userID'] = train_df['uid_inner'].apply(lambda x: trainset.to_raw_uid(int(x)))
train_df['itemID'] = train_df['iid_inner'].apply(lambda x: trainset.to_raw_iid(int(x)))
train_sample = train_df[train_df['userID'].isin(sampled_users)]

def evaluate(model, pred_df, threshold=4.0, k=10):
    # only rating >= threshold counts as relevant ground truth
    rel = pred_df[(pred_df['uid'].isin(sampled_users)) &
                  (pred_df['r_ui'] >= threshold)]
    rating_true = rel[['uid', 'iid', 'r_ui']].rename(
        columns={'uid': 'userID', 'iid': 'itemID', 'r_ui': 'rating'})
    rating_pred = rel[['uid', 'iid', 'est']].rename(
        columns={'uid': 'userID', 'iid': 'itemID', 'est': 'prediction'})

    ranking_pred = compute_ranking_predictions(
        model, train_sample,
        usercol='userID', itemcol='itemID',
        remove_seen=True
    )

    return {
        'RMSE':         rmse(rating_true, rating_pred,
                             col_user='userID', col_item='itemID',
                             col_rating='rating', col_prediction='prediction'),
        'MAP@10':       map_at_k(rating_true, ranking_pred, k=k,
                                 col_user='userID', col_item='itemID',
                                 col_prediction='prediction'),
        'NDCG@10':      ndcg_at_k(rating_true, ranking_pred, k=k,
                                  col_user='userID', col_item='itemID',
                                  col_rating='rating',
                                  col_prediction='prediction'),
        'Precision@10': precision_at_k(rating_true, ranking_pred, k=k,
                                       col_user='userID', col_item='itemID',
                                       col_prediction='prediction'),
        'Recall@10':    recall_at_k(rating_true, ranking_pred, k=k,
                                    col_user='userID', col_item='itemID',
                                    col_prediction='prediction'),
    }

print("\n" + "="*72)
print(f"Final Evaluation  |  relevance threshold = {RELEVANCE_THRESHOLD}")
print("="*72)
print(f"{'Model':<14}{'RMSE':>10}{'MAP@10':>10}{'NDCG@10':>10}"
      f"{'Prec@10':>10}{'Rec@10':>10}")
for name, pred_df in predictions.items():
    m = evaluate(models[name], pred_df)
    print(f"{name:<14}{m['RMSE']:>10.4f}{m['MAP@10']:>10.4f}"
          f"{m['NDCG@10']:>10.4f}{m['Precision@10']:>10.4f}"
          f"{m['Recall@10']:>10.4f}")
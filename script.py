import os
import argparse
from datetime import timedelta
import polars as pl

from scipy.sparse import csr_matrix
import numpy as np
import implicit
import mlflow
from dotenv import load_dotenv

EVAL_DAYS_TRESHOLD = 14
DATA_DIR = 'data/'

load_dotenv()


def get_data():
    df_test_users = pl.read_parquet(f'{DATA_DIR}/test_users.pq')
    df_clickstream = pl.read_parquet(f'{DATA_DIR}/clickstream.pq')
    df_event = pl.read_parquet(f'{DATA_DIR}/events.pq')
    return df_test_users, df_clickstream, df_event


def split_train_test(df_clickstream: pl.DataFrame, df_event: pl.DataFrame, baseline: bool):
    treshhold = df_clickstream['event_date'].max() - timedelta(days=EVAL_DAYS_TRESHOLD)

    df_train = df_clickstream.filter(df_clickstream['event_date'] <= treshhold)
    df_eval = df_clickstream.filter(df_clickstream['event_date'] > treshhold)[['cookie', 'node', 'event']]

    df_eval = df_eval.join(df_train, on=['cookie', 'node'], how='anti')

    df_eval = df_eval.filter(
        pl.col('event').is_in(
            df_event.filter(pl.col('is_contact') == 1)['event'].unique()
        )
    )
    df_eval = df_eval.filter(
        pl.col('cookie').is_in(df_train['cookie'].unique())
    ).filter(
        pl.col('node').is_in(df_train['node'].unique())
    )

    df_eval = df_eval.unique(['cookie', 'node'])

    if not baseline:
        PARAM_CONTACT_WEIGHT = 6.0
        PARAM_VIEW_WEIGHT = 2.5

        df_train_with_event = df_train.join(df_event.select(['event', 'is_contact']), on='event', how='left')

        df_train = df_train_with_event.with_columns(
            pl.when(pl.col('is_contact') == 1)
            .then(pl.lit(PARAM_CONTACT_WEIGHT, dtype=pl.Float64))
            .otherwise(pl.lit(PARAM_VIEW_WEIGHT, dtype=pl.Float64))
            .alias('event_based_weight')
        )[['cookie', 'node', 'event_based_weight']]

    return df_train, df_eval


def get_als_pred(users, nodes, user_to_pred, factors: int, iterations: int, regularization: float, values: None | list, random_state: int):
    user_ids = users.unique().to_list()
    item_ids = nodes.unique().to_list()

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    index_to_item_id = {v: k for k, v in item_id_to_index.items()}

    rows = users.replace_strict(user_id_to_index).to_list()
    cols = nodes.replace_strict(item_id_to_index).to_list()

    if values is None:
        values = [1] * len(users)

    sparse_matrix = csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    model = implicit.als.AlternatingLeastSquares(
        iterations=iterations,
        factors=factors,
        regularization=regularization,
        use_gpu=False,
        random_state=random_state
    )
    model.fit(sparse_matrix)

    user4pred = np.array([user_id_to_index[i] for i in user_to_pred])

    recommendations, scores = model.recommend(
        user4pred,
        sparse_matrix[user4pred],
        N=40,
        filter_already_liked_items=True,
    )

    df_pred = pl.DataFrame(
        {
            'node': [
                [index_to_item_id[i] for i in i] for i in recommendations.tolist()
            ],
            'cookie': list(user_to_pred),
            'scores': scores.tolist()

        }
    )
    df_pred = df_pred.explode(['node', 'scores'])
    return df_pred


def train(df_train: pl.DataFrame, df_eval: pl.DataFrame, factors: int, iterations: int, regularization: float, baseline: bool, random_state: int):
    users = df_train["cookie"]
    nodes = df_train["node"]
    values = None
    if not baseline:
        values = df_train['event_based_weight'].to_list()
    eval_users = df_eval['cookie'].unique().to_list()
    df_pred = get_als_pred(users, nodes, eval_users, factors, iterations, regularization, values, random_state)
    return df_pred


def recall_at(df_true, df_pred, k=40):
    return df_true[['node', 'cookie']].join(
        df_pred.group_by('cookie').head(k).with_columns(value=1)[['node', 'cookie', 'value']],
        how='left',
        on=['cookie', 'node']
    ).select(
        [pl.col('value').fill_null(0), 'cookie']
    ).group_by(
        'cookie'
    ).agg(
        [
            pl.col('value').sum() / pl.col(
                'value'
            ).count()
        ]
    )['value'].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--run_name', required=True)
    
    parser.add_argument('--factors', type=int, default=60)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--regularization', type=float, default=0.01)
    parser.add_argument('--baseline', type=str, default=True)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    experiment_name = args.experiment_name
    factors = args.factors
    iterations = args.iterations
    regularization = args.regularization
    baseline = args.baseline
    random_state = args.random_state

    mlflow.set_tracking_uri(
        os.environ.get('MLFLOW_TRACKING_URI')
    )

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name, artifact_location='mlflow-artifacts:/')

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        df_test_users, df_clickstream, df_event = get_data()
        df_train, df_eval = split_train_test(df_clickstream, df_event, baseline)
        df_pred = train(df_train, df_eval, factors, iterations, regularization, baseline, random_state)
        mlflow.log_params({
        'model_type': 'als',
        'launch': 'airflow',
        'factors': factors,
        'iterations': iterations,
        'regularization': regularization,
        'random_state': random_state
        })

        metric = recall_at(df_eval, df_pred, k=40)

        mlflow.log_metric('Recall_40', metric)


if __name__ == '__main__':
    main()

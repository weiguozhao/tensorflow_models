# encoding=utf-8

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def onehot_encoder(labels, NUM_CLASSES):
    enc = LabelEncoder()
    labels = enc.fit_transform(labels)
    labels = labels.astype(np.int32)
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels], 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0)
    with tf.Session() as sess:
        return sess.run(onehot_labels)


def load_iris_dataset():
    header = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
    df_iris = pd.read_csv('../data/iris.csv', sep=',', names=header)
    df_iris = df_iris[(df_iris['label'] != 'setosa')]
    labels = onehot_encoder(df_iris['label'], 2)
    cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X_train, X_test, y_train, y_test = train_test_split(df_iris[cols].values, labels, test_size=0.2, random_state=0)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return X_train, y_train, X_test, y_test


def load_dataset():
    header = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    df_user = pd.read_csv('../data/MovieLens_100K_Dataset/u.user', sep='|', names=header)
    df_user['age'] = pd.cut(df_user['age'], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                            labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
                                    '90-100'])
    df_user = pd.get_dummies(df_user, columns=['gender', 'occupation', 'age'])
    df_user = df_user.drop(columns=['zip_code'])

    header = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
              'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    df_item = pd.read_csv('../data/MovieLens_100K_Dataset/u.item', sep='|', names=header, encoding="ISO-8859-1")
    df_item = df_item.drop(columns=['title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown'])

    user_features = df_user.columns.values.tolist()
    movie_features = df_item.columns.values.tolist()
    cols = user_features + movie_features
    cols.remove('user_id')
    cols.remove('item_id')
    print(cols)
    feature2field = {}
    index2cols = {}
    last_field = ''
    ind = 0
    field_index = 0
    for col in cols:
        index2cols[ind] = col
        infos = col.split('_')
        field = ''
        feature = ''
        if len(infos) == 2:
            field = infos[0]
            feature = infos[1]
        elif len(infos) == 1:
            field = infos[0]

        if last_field == field:
            feature2field[ind] = field_index
        else:
            if last_field != '':
                if field_index <= 3:
                    field_index += 1
                feature2field[ind] = field_index
                last_field = field
            else:
                feature2field[ind] = field_index
                last_field = field
        ind += 1

    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df_train = pd.read_csv('../data/MovieLens_100K_Dataset/ua.base', sep='\t', names=header)
    df_train['rating'] = df_train.rating.apply(lambda x: 1 if int(x) == 5 else 0)
    df_train = df_train.merge(df_user, on='user_id', how='left')
    df_train = df_train.merge(df_item, on='item_id', how='left')

    df_test = pd.read_csv('../data/MovieLens_100K_Dataset/ua.test', sep='\t', names=header)
    df_test['rating'] = df_test.rating.apply(lambda x: 1 if int(x) == 5 else 0)
    df_test = df_test.merge(df_user, on='user_id', how='left')
    df_test = df_test.merge(df_item, on='item_id', how='left')

    train_labels = onehot_encoder(df_train['rating'].astype(np.int32), 2)
    test_labels = onehot_encoder(df_test['rating'].astype(np.int32), 2)
    return df_train[cols].values, train_labels, df_test[cols].values, test_labels, feature2field


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, feature2field = load_dataset()
    print("x_train", x_train[0])
    print("y_train", y_train[0])
    print("x_test", x_test[0])
    print("y_test", y_test[0])
    """
    ['gender_F', 'gender_M', 'occupation_administrator', 'occupation_artist', 'occupation_doctor', 'occupation_educator', 'occupation_engineer', 'occupation_entertainment', 'occupation_executive', 'occupation_healthcare', 'occupation_homemaker', 'occupation_lawyer', 'occupation_librarian', 'occupation_marketing', 'occupation_none', 'occupation_other', 'occupation_programmer', 'occupation_retired', 'occupation_salesman', 'occupation_scientist', 'occupation_student', 'occupation_technician', 'occupation_writer', 'age_0-10', 'age_10-20', 'age_20-30', 'age_30-40', 'age_40-50', 'age_50-60', 'age_60-70', 'age_70-80', 'age_80-90', 'age_90-100', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    x_train [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1
     1 0 0 0 0 0 0 0 0 0 0 0 0 0]
    y_train [0. 1.]
    x_test [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 1 0 0 0 0 0 1 0 0 0 0]
    y_test [1. 0.]
    """

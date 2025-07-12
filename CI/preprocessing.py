from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import pandas as pd

def preprocess_data(train_df, test_df, target_column, save_path, file_path):
    # Deteksi fitur numerik dan kategorik
    numerik = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    kategorikal = train_df.select_dtypes(include=['object']).columns.tolist()

    if target_column in numerik:
        numerik.remove(target_column)
    if target_column in kategorikal:
        kategorikal.remove(target_column)

    # Fitur kategorik yang ingin di-onehot atau ordinal
    one_hot_fitur = ['DP', 'Varietas', 'Rayon', 'Bulan Tanam']
    ordinal_fitur = ['Tingkat Tanam']

    # Pipeline masing-masing tipe
    onehot_transform = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    ordinal_transform = Pipeline([
        ('encoder', OrdinalEncoder(categories=[['PC', 'R1', 'R2', 'R3']], handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    numeric_fitur = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Column transformer
    preprocessor = ColumnTransformer([
        ('num', numeric_fitur, numerik),
        ('onehot', onehot_transform, one_hot_fitur),
        ('ordinal', ordinal_transform, ordinal_fitur)
    ])

    # Pisahkan X dan y
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Fit dan transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Simpan pipeline dan header
    dump(preprocessor, save_path)
    feature_names = preprocessor.get_feature_names_out()
    pd.DataFrame(columns=feature_names).to_csv(file_path, index=False)

    return X_train_processed, X_test_processed, y_train, y_test

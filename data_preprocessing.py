import numpy as np
import matplotlib.pyplot as plt
from helpers import load_csv_data

data_path = "dataset"
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)

def remove_first_n_columns(x_train, x_test, first_n=9):
    original_features = x_train.shape[1]
    x_train = x_train[:, first_n:]
    x_test = x_test[:, first_n:]
    deleted = original_features - x_train.shape[1]
    print(f"remove_first_n_columns: Deleted {deleted} features")
    return x_train, x_test
#not necessary for the implementation of the model 
#see why we get ridd of these columns in particular in the report

def replace_special_values_with_nan(x_train, x_test):
    special_values = [7, 77, 777, 7777, 77777, 9, 99, 900, 999, 9999, 999999, 99900]
    x_train_new = x_train.copy()
    x_test_new = x_test.copy()
    for val in special_values:
        x_train_new[x_train_new == val] = np.nan
        x_test_new[x_test_new == val] = np.nan
    return x_train_new, x_test_new
#we saw the documentation of the dataset that these values were used to indicate missing data (person did not answer the question)


def remove_features_with_missing_values(x_train, x_test, threshold_missing_values=0.2):
    original_features = x_train.shape[1]
    n_rows = x_train.shape[0]
    missing_counts = np.isnan(x_train).sum(axis=0)
    missing_proportions = missing_counts / n_rows

    cols_to_keep = missing_proportions <= threshold_missing_values

    x_train_reduced = x_train[:, cols_to_keep]
    x_test_reduced = x_test[:, cols_to_keep]

    deleted = original_features - x_train_reduced.shape[1]
    print(f"remove_features_with_missing_values: Deleted {deleted} features")
    return x_train_reduced, x_test_reduced
#we remove the features with too many missing values (Nan values) because not interesting for the model
#ici on remove que les NaN? pas les valeurs manquantes en general?

def get_numerical_and_categorical_features(x_train, unique_threshold=20):
    numerical_indices = []
    categorical_indices = []

    for i in range(x_train.shape[1]):
        unique_vals = np.unique(x_train[~np.isnan(x_train[:, i]), i])
        if len(unique_vals) <= unique_threshold:
            categorical_indices.append(i)
        else:
            numerical_indices.append(i)
    return numerical_indices, categorical_indices
#we separate numerical and categorical features based on the number of unique values (20) because in data exploration
#we saw a graph feature count by number of unique feature values (see chat gpt)

def impute_numerical_with_median(x_train, x_test, numerical_indices):
    x_train_imputed = x_train.copy()
    x_test_imputed = x_test.copy()
    for idx in numerical_indices:
        train_col = x_train[:, idx]
        median_val = np.nanmedian(train_col)
        x_train_imputed[np.isnan(x_train[:, idx]), idx] = median_val
        x_test_imputed[np.isnan(x_test[:, idx]), idx] = median_val
    return x_train_imputed, x_test_imputed
#we replace the nan values in numerical features with the median of the column

def impute_categorical_with_mode(x_train, x_test, categorical_indices):
    x_train_imputed = x_train.copy()
    x_test_imputed = x_test.copy()
    for idx in categorical_indices:
        train_col = x_train[:, idx]
        vals, counts = np.unique(train_col[~np.isnan(train_col)], return_counts=True)
        mode_val = vals[np.argmax(counts)]
        x_train_imputed[np.isnan(x_train[:, idx]), idx] = mode_val
        x_test_imputed[np.isnan(x_test[:, idx]), idx] = mode_val
    return x_train_imputed, x_test_imputed
#we replace the nan values in categorical features with the mode of the column


def remove_outliers(x_train, y_train, numerical_indices, lower_quantile=0.01, upper_quantile=0.99):
    x_train = x_train.copy()
    outlier_mask = np.zeros(x_train.shape[0], dtype=bool)
    for idx in numerical_indices:
        col = x_train[:, idx]
        lower_q = np.quantile(col, lower_quantile)
        upper_q = np.quantile(col, upper_quantile)
        outlier_mask = outlier_mask | (col < lower_q) | (col > upper_q)
    num_rows_deleted = np.sum(outlier_mask)
    print(f"remove_outliers: Deleted {num_rows_deleted} rows")
    x_train_filtered = x_train[~outlier_mask]
    y_train_filtered = y_train[~outlier_mask]
    return x_train_filtered, y_train_filtered
#we remove the first and last percent iles of numerical features to get rid of outliers
    
def standardize(x_train, x_test):
    x_train_std = x_train.copy()
    x_test_std = x_test.copy()

    means = np.nanmean(x_train, axis=0)
    stds = np.nanstd(x_train, axis=0)
    # Avoid division by zero
    stds[stds == 0] = 1.0

    x_train_std = (x_train - means) / stds
    x_test_std = (x_test - means) / stds

    return x_train_std, x_test_std

def remove_low_variance_features(x_train, x_test, threshold_variance=1e-5):
    original_features = x_train.shape[1]
    variances = np.nanvar(x_train, axis=0)
    keep_indices = np.where(variances > threshold_variance)[0]
    x_train_filtered = x_train[:, keep_indices]
    x_test_filtered = x_test[:, keep_indices]
    deleted = original_features - x_train_filtered.shape[1]
    print(f"remove_low_variance_features: Deleted {deleted} features")
    return x_train_filtered, x_test_filtered
#we remove features with very low variance because they do not provide useful information for the model



def remove_highly_correlated_features(x_train, x_test, threshold_correlation=0.95):
    original_features = x_train.shape[1]
    corr_matrix = np.corrcoef(x_train, rowvar=False)
    num_features = corr_matrix.shape[0]
    to_remove = set()
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if np.abs(corr_matrix[i, j]) > threshold_correlation:
                to_remove.add(j)
    keep_indices = [idx for idx in range(num_features) if idx not in to_remove]
    x_train_filtered = x_train[:, keep_indices]
    x_test_filtered = x_test[:, keep_indices]
    deleted = original_features - x_train_filtered.shape[1]
    print(f"remove_highly_correlated_features: Deleted {deleted} features")
    return x_train_filtered, x_test_filtered
#we remove features that are highly correlated with each other to reduce redundancy in the data


def preprocess_data(x_train, x_test, y_train, threshold_missing_values=0.2, unique_threshold=20, lower_quantile=0.01, upper_quantile=0.99, threshold_variance=1e-5, threshold_correlation=0.95):

    x_train, x_test = remove_first_n_columns(x_train, x_test)
    x_train, x_test = replace_special_values_with_nan(x_train, x_test)
    x_train, x_test = remove_features_with_missing_values(x_train, x_test, threshold_missing_values)
    numerical_indices, categorical_indices = get_numerical_and_categorical_features(x_train, unique_threshold)
    x_train, x_test = impute_numerical_with_median(x_train, x_test, numerical_indices)
    x_train, x_test = impute_categorical_with_mode(x_train, x_test, categorical_indices)
    x_train, y_train = remove_outliers(x_train, y_train, numerical_indices, lower_quantile, upper_quantile)
    x_train, x_test = standardize(x_train, x_test)
    x_train, x_test = remove_low_variance_features(x_train, x_test, threshold_variance=1e-5)
    x_train, x_test = remove_highly_correlated_features(x_train, x_test, threshold_correlation)
    return x_train, x_test, y_train

preprocessed_x_train, preprocessed_x_test, preprocessed_y_train = preprocess_data(
    x_train, x_test, y_train,
    threshold_missing_values=0.2, unique_threshold=20,
    lower_quantile=0.01, upper_quantile=0.99,
    threshold_variance=1e-5, threshold_correlation=0.95
)

print("Any NaN in x_train?:", np.isnan(preprocessed_x_train).any())
print("Any NaN in x_test?:", np.isnan(preprocessed_x_test).any())

np.save("x_train_preprocessed.npy", preprocessed_x_train)
np.save("x_test_preprocessed.npy", preprocessed_x_test)
np.save("y_train_preprocessed.npy", preprocessed_y_train)
np.save("train_ids.npy", train_ids)
np.save("test_ids.npy", test_ids)

print(f"Preprocessed x_train shape: {preprocessed_x_train.shape}")
print(f"Preprocessed x_test shape: {preprocessed_x_test.shape}")
print(f"Preprocessed y_train shape: {preprocessed_y_train.shape}")

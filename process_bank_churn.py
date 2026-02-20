import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_data(df, target_col):
    """
    Split dataframe into train and validation parts.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[target_col]
    )
    return train_df, val_df


def create_inputs_targets(train_df, val_df, input_cols, target_col):
    """
    Create inputs and targets for train and validation sets.
    """
    data = {}

    data['train_inputs'] = train_df[input_cols].copy()
    data['train_targets'] = train_df[target_col].copy()

    data['val_inputs'] = val_df[input_cols].copy()
    data['val_targets'] = val_df[target_col].copy()

    return data


def scale_numeric_features(data, numeric_cols):
    """
    Scale numeric features using StandardScaler.
    """
    scaler = StandardScaler().fit(data['train_inputs'][numeric_cols])

    data['train_inputs'][numeric_cols] = scaler.transform(data['train_inputs'][numeric_cols])
    data['val_inputs'][numeric_cols] = scaler.transform(data['val_inputs'][numeric_cols])

    return scaler


def encode_categorical_features(data, categorical_cols):
    """
    One-hot encode categorical features.
    """
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore'
    ).fit(data['train_inputs'][categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    for split in ['train', 'val']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoded_cols,
            index=data[f'{split}_inputs'].index
        )

        data[f'{split}_inputs'] = pd.concat(
            [data[f'{split}_inputs'], encoded_df],
            axis=1
        )

        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)

    return encoder


def preprocess_data(raw_df, scaler_numeric=True):
    """
    Full preprocessing for Bank Churn dataset.
    """

    target_col = 'Exited'

    # 1️⃣ Drop Surname
    raw_df = raw_df.drop(columns=['Surname', 'CustomerId', 'id'])

    # 2️⃣ Split data
    train_df, val_df = split_data(raw_df, target_col)

    # 3️⃣ Define columns
    input_cols = list(raw_df.columns)
    input_cols.remove(target_col)

    data = create_inputs_targets(train_df, val_df, input_cols, target_col)

    # 4️⃣ Detect numeric and categorical
    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes(include='object').columns.tolist()

    # 5️⃣ Encode categorical
    encoder = encode_categorical_features(data, categorical_cols)

    # 6️⃣ Scale numeric (optional)
    scaler = None
    if scaler_numeric:
        scaler = scale_numeric_features(data, numeric_cols)

    # Updated input columns after encoding
    input_cols = data['train_inputs'].columns.tolist()

    return (
        data['train_inputs'],
        data['train_targets'],
        data['val_inputs'],
        data['val_targets'],
        input_cols,
        scaler,
        encoder
    )


def preprocess_new_data(new_df, input_cols, scaler, encoder, scaler_numeric=True):
    """
    Preprocess new (test) data using trained scaler and encoder.
    """

    new_df = new_df.drop(columns=['Surname', 'CustomerId', 'id'])

    numeric_cols = new_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = new_df.select_dtypes(include='object').columns.tolist()

    # Encode categorical
    encoded = encoder.transform(new_df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_cols,
        index=new_df.index
    )

    new_df = pd.concat([new_df, encoded_df], axis=1)
    new_df.drop(columns=categorical_cols, inplace=True)

    # Scale numeric
    if scaler_numeric and scaler is not None:
        new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])

    # Keep correct column order
    new_df = new_df[input_cols]

    return new_df

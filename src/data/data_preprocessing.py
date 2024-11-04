import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(data):
    """
    Preprocess the data for each commodity
    """
    processed_data = {}
    for commodity, df in data.items():
        # Save the date column
        date_column = df['Reported Date'].copy()
        
        # Identify numeric and categorical columns
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = df.select_dtypes(include=['object']).columns.drop('Reported Date')
        
        # Set threshold to distinguish between low and high cardinality
        high_cardinality_threshold = 100
        categorical_features_low_card = [col for col in categorical_features if df[col].nunique() < high_cardinality_threshold]
        categorical_features_high_card = [col for col in categorical_features if df[col].nunique() >= high_cardinality_threshold]
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer_low_card = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=True))
        ])

        categorical_transformer_high_card = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat_low_card', categorical_transformer_low_card, categorical_features_low_card),
                ('cat_high_card', categorical_transformer_high_card, categorical_features_high_card)
            ]
        )

        # Fit and transform the data
        processed_array = preprocessor.fit_transform(df)

        # Get feature names
        numeric_feature_names = numeric_features.tolist()
        categorical_feature_names_low_card = preprocessor.named_transformers_['cat_low_card'].named_steps['onehot'].get_feature_names_out(categorical_features_low_card).tolist()
        categorical_feature_names_high_card = categorical_features_high_card
        feature_names = numeric_feature_names + categorical_feature_names_low_card + categorical_feature_names_high_card

        # Convert to DataFrame
        processed_df = pd.DataFrame.sparse.from_spmatrix(processed_array, columns=feature_names, index=df.index)
        
        # Add back the date column
        processed_df['date'] = date_column
        
        processed_data[commodity] = processed_df

    return processed_data

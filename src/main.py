from data.data_collection import collect_all_data
from data.data_preprocessing import preprocess_data
from features.feature_engineering import engineer_features
from models.model_training import train_and_evaluate_models

def main():
    # Collect raw data
    raw_data = collect_all_data()
    
    results = {}
    
    for commodity, df in raw_data.items():
        print(f"Processing {commodity}...")
        
        # Print raw data info
        print(f"Raw data for {commodity}:")
        print(df.info())
        print("\n")

        # Preprocess data
        processed_data = preprocess_data({commodity: df})
        
        # Print processed data info
        print(f"Processed data for {commodity}:")
        print(processed_data[commodity].info())
        print("\n")

        # Engineer features
        if 'Reported Date' in processed_data[commodity].columns:
            featured_data = engineer_features(processed_data)
        else:
            print(f"Warning: 'Reported Date' column not found for {commodity}. Skipping feature engineering.")
            featured_data = processed_data
        
        # Train and evaluate models
        result = train_and_evaluate_models(featured_data)
        results[commodity] = result[commodity]
        
        # Print results
        print(f"Results for {commodity}:")
        print(f"RMSE: {results[commodity]['evaluation']['RMSE']:.2f}")
        print(f"R2 Score: {results[commodity]['evaluation']['R2']:.2f}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
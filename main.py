import os
import json
import pandas as pd
from preprocessing import load_data, preprocess_dataframe, encode_labels, extract_features, TEXT_COL
from model_chained import ChainedModel
from evaluation import evaluate_predictions
from sklearn.model_selection import train_test_split

# Filepaths
DATA_DIR = 'CA_Code/data'
CSV_FILES = ['AppGallery.csv', 'Purchasing.csv']

# Map the actual CSV columns to the required labels
# 'Type 2' -> intent, 'Type 3' -> tone, 'Type 4' -> resolution_type
LABEL_COLS = ['Type 2', 'Type 3', 'Type 4']

# Output files
PREDICTIONS_CSV = 'CA_Code/predictions_output.csv'
EVAL_JSON = 'CA_Code/evaluation_report.json'

def chained_accuracy(y_true_tuple, y_pred_tuple):
    """Compute accuracy where all three labels must be correct."""
    y_true = list(zip(*y_true_tuple))
    y_pred = list(zip(*y_pred_tuple))
    correct = sum([t == p for t, p in zip(y_true, y_pred)])
    return correct / len(y_true) if y_true else 0.0

def main():
    print("Step 1: Loading and concatenating data...")
    dfs = [load_data(os.path.join(DATA_DIR, f)) for f in CSV_FILES]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows.")

    print("Step 2: Preprocessing text...")
    df = preprocess_dataframe(df)

    print("Step 3: Encoding labels...")
    df, encoders = encode_labels(df, LABEL_COLS)
    y_intent = df['Type 2_enc'].values
    y_tone = df['Type 3_enc'].values
    y_resolution = df['Type 4_enc'].values

    print("Step 4: Extracting features from text column...")
    X, vectorizer = extract_features(df)

    print("Step 5: Splitting data into train and test sets...")
    X_train, X_test, y_intent_train, y_intent_test, y_tone_train, y_tone_test, y_resolution_train, y_resolution_test = train_test_split(
        X, y_intent, y_tone, y_resolution, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    print("Step 6: Training chained model...")
    chained_model = ChainedModel()
    chained_model.fit(X_train, y_intent_train, y_tone_train, y_resolution_train)
    print("Model training complete.")

    print("Step 7: Predicting on test set...")
    intent_pred, tone_pred, resolution_pred = chained_model.predict(X_test)
    print("Intent predictions (head):", intent_pred[:5])
    print("Tone predictions (head):", tone_pred[:5])
    print("Resolution predictions (head):", resolution_pred[:5])

    print("Step 8: Evaluating models...")
    metrics = {}
    metrics['intent'] = evaluate_predictions(y_intent_test, intent_pred, 'Intent')
    metrics['tone'] = evaluate_predictions(y_tone_test, tone_pred, 'Tone')
    metrics['resolution_type'] = evaluate_predictions(y_resolution_test, resolution_pred, 'Resolution Type')

    # Chained accuracy: all three correct
    print("Step 9: Calculating overall chained accuracy...")
    overall_acc = chained_accuracy(
        (y_intent_test, y_tone_test, y_resolution_test),
        (intent_pred, tone_pred, resolution_pred)
    )
    print(f"\nOverall chained accuracy (all three labels correct): {overall_acc:.4f}")
    metrics['overall_chained_accuracy'] = overall_acc

    # Step 10: Save predictions to CSV
    print("Step 10: Saving predictions to CSV...")
    # Decode labels for readability
    intent_decoder = encoders['Type 2'].inverse_transform
    tone_decoder = encoders['Type 3'].inverse_transform
    resolution_decoder = encoders['Type 4'].inverse_transform
    pred_df = pd.DataFrame({
        TEXT_COL: X_test.shape[0] * [None],  # placeholder
        'actual_intent': intent_decoder(y_intent_test),
        'predicted_intent': intent_decoder(intent_pred),
        'actual_tone': tone_decoder(y_tone_test),
        'predicted_tone': tone_decoder(tone_pred),
        'actual_resolution_type': resolution_decoder(y_resolution_test),
        'predicted_resolution_type': resolution_decoder(resolution_pred),
    })
    # Add the original message for each test sample
    # Find the indices of the test set in the original DataFrame
    _, test_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42
    )
    pred_df[TEXT_COL] = df.iloc[list(test_idx)][TEXT_COL].values
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Predictions saved to {PREDICTIONS_CSV}")

    # Step 11: Save evaluation report as JSON
    print("Step 11: Saving evaluation report as JSON...")
    with open(EVAL_JSON, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation report saved to {EVAL_JSON}")

if __name__ == "__main__":
    main() 
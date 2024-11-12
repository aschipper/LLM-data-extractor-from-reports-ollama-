from sklearn.metrics import f1_score
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

class Evaluator:
    def __init__(
        self,
        ground_truth_path: Path,
        predictions_path: Path,
        metric: str = 'macro',
    ):
        self.ground_truth_path = ground_truth_path
        self.predictions_path = predictions_path
        self.metric = metric

    def evaluate(self):
        # Load ground truth and predictions
        gt_df = pd.read_json(self.ground_truth_path, lines=True)
        pred_df = pd.read_json(self.predictions_path, lines=True)

        # Merge on 'uid'
        merged_df = pd.merge(gt_df, pred_df, on='uid', suffixes=('_gt', '_pred'))

        # Get label fields from ground truth (excluding 'uid')
        label_fields = [col for col in gt_df.columns if col != 'uid']

        total_score = 0.0
        for label_field in label_fields:
            y_true = merged_df[f"{label_field}_gt"]
            y_pred = merged_df[f"{label_field}_pred"]

            # Convert lists stored as strings back to lists if necessary
            y_true = y_true.apply(eval) if y_true.dtype == object else y_true
            y_pred = y_pred.apply(eval) if y_pred.dtype == object else y_pred

            # Handle different types (list, bool, etc.)
            if isinstance(y_true.iloc[0], list):
                # Multi-label
                y_true_flat = [item for sublist in y_true for item in sublist]
                y_pred_flat = [item for sublist in y_pred for item in sublist]
                labels = list(set(y_true_flat + y_pred_flat))
                mlb = MultiLabelBinarizer(classes=labels)
                y_true_bin = mlb.fit_transform(y_true)
                y_pred_bin = mlb.transform(y_pred)
                score = f1_score(y_true_bin, y_pred_bin, average=self.metric)
            else:
                # Single-label or binary classification
                score = f1_score(y_true, y_pred, average=self.metric)

            print(f"F1 Score for {label_field} ({self.metric}): {score}")
            total_score += score

        average_score = total_score / len(label_fields)
        print(f"Average F1 Score ({self.metric}): {average_score}")
        return average_score
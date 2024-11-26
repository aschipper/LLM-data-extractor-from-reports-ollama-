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
        gt_df = pd.read_json('/home/aschipper/projects/LLM_data_extractor_optuna/data/df_llm_GT.jsonl', lines=True)
        pred_df = pd.read_json(self.predictions_path)

        # Merge on 'uid'
        merged_df = pd.merge(gt_df, pred_df, on='uid', suffixes=('_gt', '_pred'))

        # Get label fields from ground truth (excluding 'uid')
        label_fields = [col for col in gt_df.columns if col != 'uid']

        total_score = 0.0

        def clean_sample(sample):
            if isinstance(sample, list):
                cleaned = []
                for item in sample:
                    if item is None or (isinstance(item, float) and np.isnan(item)):
                        cleaned.append('niet gerapporteerd')
                    elif isinstance(item, dict):
                        # Extract the 'description' if present
                        if 'description' in item:
                            cleaned.append(item['description'])
                        else:
                            cleaned.append('niet gerapporteerd')
                    else:
                        cleaned.append(str(item))
                return cleaned
            elif sample is None or (isinstance(sample, float) and np.isnan(sample)):
                return ['niet gerapporteerd']
            elif isinstance(sample, dict):
                if 'description' in sample:
                    return [sample['description']]
                else:
                    return ['niet gerapporteerd']
            else:
                return [str(sample)]

        for label_field in label_fields:
            y_true = merged_df[f"{label_field}_gt"]
            y_pred = merged_df[f"{label_field}_pred"]

            # Convert lists stored as strings back to lists if necessary
            y_true = y_true.apply(eval) if y_true.dtype == object else y_true
            y_pred = y_pred.astype(str).apply(eval) if y_pred.dtype == object else y_pred

            # Clean y_true and y_pred
            y_true_cleaned = y_true.apply(clean_sample).tolist()
            y_pred_cleaned = y_pred.apply(clean_sample).tolist()

            # Flatten to get the set of labels
            y_true_flat = [item for sublist in y_true_cleaned for item in sublist]
            y_pred_flat = [item for sublist in y_pred_cleaned for item in sublist]

            # Create the set of labels
            labels = list(set(y_true_flat + y_pred_flat))

            if isinstance(y_true_cleaned[0], list):
                # Multi-label classification
                mlb = MultiLabelBinarizer(classes=labels)
                y_true_bin = mlb.fit_transform(y_true_cleaned)
                y_pred_bin = mlb.transform(y_pred_cleaned)

                score = f1_score(y_true_bin, y_pred_bin, average=self.metric)
            else:
                # Single-label classification
                y_true_flat_single = [items[0] for items in y_true_cleaned]
                y_pred_flat_single = [items[0] for items in y_pred_cleaned]

                score = f1_score(y_true_flat_single, y_pred_flat_single, average=self.metric)

            print(f"F1 Score for {label_field} ({self.metric}): {score}")
            total_score += score

        average_score = total_score / len(label_fields)
        print(f"Average F1 Score ({self.metric}): {average_score}")
        return average_score

    # def evaluate(self):
    #     # Load ground truth and predictions
    #     gt_df = pd.read_json('/home/aschipper/projects/LLM_data_extractor_optuna/data/df_llm_GT.jsonl', lines=True)
    #     pred_df = pd.read_json(self.predictions_path)
    #
    #     # Merge on 'uid'
    #     merged_df = pd.merge(gt_df, pred_df, on='uid', suffixes=('_gt', '_pred'))
    #
    #     # Get label fields from ground truth (excluding 'uid')
    #     label_fields = [col for col in gt_df.columns if col != 'uid']
    #
    #     total_score = 0.0
    #
    #     for label_field in label_fields:
    #         y_true = merged_df[f"{label_field}_gt"]
    #         y_pred = merged_df[f"{label_field}_pred"]
    #
    #         # Convert lists stored as strings back to lists if necessary
    #         y_true = y_true.apply(eval) if y_true.dtype == object else y_true
    #         y_pred = y_pred.astype(str).apply(eval) if y_pred.dtype == object else y_pred
    #         # Replace None values with 'niet gerapporteerd'
    #         y_pred = y_pred.apply(lambda x: 'niet gerapporteerd' if x is None else x)
    #
    #         # If elements are supposed to be lists, ensure they are lists
    #         #y_pred = y_pred.apply(lambda x: x if isinstance(x, list) else [x])
    #
    #         # If sublists might contain None, replace them as well
    #         y_pred = y_pred.apply(lambda sublist: ['niet gerapporteerd' if item is None else item for item in sublist])
    #
    #         # Handle different types (list, bool, etc.)
    #         if isinstance(y_true.iloc[0], list):
    #             # Multi-label
    #             y_true_flat = [item for sublist in y_true for item in sublist]
    #             y_pred_flat = [item for sublist in y_pred for item in sublist]
    #             print(y_pred_flat)
    #             print(y_true_flat)
    #             labels = list(set(y_true_flat + y_pred_flat))
    #             mlb = MultiLabelBinarizer(classes=labels)
    #             y_true_bin = mlb.fit_transform(y_true)
    #             y_pred_bin = mlb.transform(y_pred)
    #             score = f1_score(y_true_bin, y_pred_bin, average=self.metric)
    #         else:
    #             # Single-label or binary classification
    #             score = f1_score(y_true, y_pred, average=self.metric)
    #
    #         print(f"F1 Score for {label_field} ({self.metric}): {score}")
    #         total_score += score
    #
    #     average_score = total_score / len(label_fields)
    #     print(f"Average F1 Score ({self.metric}): {average_score}")
    #     return average_score
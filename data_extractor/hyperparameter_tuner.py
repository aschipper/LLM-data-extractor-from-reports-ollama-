import optuna
from pathlib import Path
from data_extractor.prediction_task import PredictionTask
from data_extractor.utils import save_json
from data_extractor.evaluator import Evaluator

class HyperparameterTuner:
    def __init__(
            self,
            model_name: str,
            task_id: int,
            num_examples: int,
            n_trials: int,
            temperature_range: tuple,
            top_k_range: tuple,
            top_p_range: tuple,
            min_p_range: tuple,
            output_dir: Path,
            task_dir: Path,
            data_dir: Path,
            ground_truth_path: Path,
            metric: str = 'macro',  # 'micro' or 'macro'
    ):
        self.model_name = model_name
        self.task_id = task_id
        self.num_examples = num_examples
        self.n_trials = n_trials
        self.temperature_range = temperature_range
        self.top_k_range = top_k_range
        self.top_p_range = top_p_range
        self.min_p_range = min_p_range
        self.output_dir = output_dir
        self.task_dir = task_dir
        self.data_dir = data_dir
        self.ground_truth_path = ground_truth_path
        self.metric = metric

    def objective(self, trial):
        # Suggest hyperparameters
        temperature = trial.suggest_float("temperature", *self.temperature_range)
        top_k = trial.suggest_int("top_k", *self.top_k_range)
        top_p = trial.suggest_float("top_p", *self.top_p_range)
        min_p = trial.suggest_float("min_p", *self.min_p_range)

        # Define the output path for this trial
        trial_output_dir = self.output_dir / f"trial_{trial.number}"
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        # Run the prediction task with the suggested hyperparameters
        prediction_task = PredictionTask(
            task_id=self.task_id,
            model_name=self.model_name,
            output_path_base=trial_output_dir,
            num_examples=self.num_examples,
            n_runs=1,  # Run once per trial
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            task_dir=self.task_dir,
            data_dir=self.data_dir,
        )
        prediction_task.run()

        # Evaluate the predictions
        predictions_path = trial_output_dir / "fold0" / "nlp-predictions-dataset.json"
        evaluator = Evaluator(
            ground_truth_path=self.ground_truth_path,
            predictions_path=predictions_path,
            metric=self.metric,
        )
        f1_score = evaluator.evaluate()

        # Report the F1 score to Optuna
        return f1_score

    def tune(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        # Save the best hyperparameters
        best_params = study.best_params
        save_json(best_params, self.output_dir, "best_hyperparameters.json")
        print(f"Best hyperparameters: {best_params}")
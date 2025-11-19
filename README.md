# 💻 CLI Usage: `LLM_data_extractor_optuna_repo`
` LLM_data_extractor_optuna_repo ` is a command-line tool for running and evaluating task-based generation experiments using customizable models, input data, and output settings.

# Installation
## **Install Ollama** on Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

###  **Change to your project (repo) directory, e.g. **
```bash
cd /path/to/your/LLM_data_extractor_optuna_repo
```

###  **Pull a model and test**
``` bash 
ollama pull qwen2.5:14b-instruct
```

## 🧰 Basic CLI Usage
Basic command structure:

```bash
python -m LLM_data_extractor_optuna_repo.main --task_id 028 ---output_dir ./output --data_dir /path/to/your/LLM_data_extractor_optuna_repo/data/data_example_ed_report.jsonl
```
## 🚩 Required Flags
| Flag         | Type  | Description                         |
|--------------|-------|-------------------------------------|
| `--task_id`  | `int` | ID of the task to run (3 digit number, required) |
---
## ⚙️ General Options, and Directories

| Flag          | Type   | Default   | Description                                 |
|---------------|--------|-----------|---------------------------------------------|
| `--run_name`  | `str`  | `"run"`   | Unique name for this run (used in logging) |
| `--n_runs`    | `int`  | `1`       | Number of repetitions for the task run |
| `-- data_dir`   | `str` | `None `   | Directory to input data file |
| `-- output_dir` | `str` | `None`   | Output directory |

---

## 🧠 Model Configuration

| Flag               | Type    | Default   | Description                                                    |
|--------------------|---------|-----------|----------------------------------------------------------------|
| `--model_name`      | `str`   | `mistral-nemo`  | Name of the model to use                                       |
| `--temperature`     | `float` | `0.1`     | Sampling temperature (0 = deterministic output)                |
| `--top_k`           | `int`   | `5`    | Top-K sampling: only consider the top K predictions            |
| `--top_p`           | `float` | `0.9`    | Top-P sampling (nucleus): consider top tokens summing to p     |
| `--min_p`     | `float`   | `0.05`     | Minimum probability threshold for token selection (filters out low-probability tokens) |

## 🧠 OPTIONAL: hyperparameter tuning (OPTUNA framework)
### Please consult the model's website or repo to find appropriate ranges.

| Flag               | Type    | Default  | Description                                                    |
|--------------------|---------|----------|----------------------------------------------------------------|
| `--model_name`      | `str`   | `mistral-nemo` | Name of the model to use                                       |
| `--temperature`     | `list`  | `[0.0, 0.7]`    | Sampling temperature (0 = deterministic output)                |
| `--top_k`           | `list`   | `[1, 50]`    | Top-K sampling: only consider the top K predictions            |
| `--top_p`           | `list` | `[0.8, 1.0]`   | Top-P sampling (nucleus): consider top tokens summing to p     |
| `--min_p`     | `list` | `[0.0, 0.2]`    | Minimum probability threshold for token selection (filters out low-probability tokens) |
---
# 💡 Examples
Run a task with default settings:
```bash
python -m LLM_data_extractor_optuna_repo.main --task_id 028 --output_dir ./output --data_dir /path/to/your/LLM_data_extractor_optuna_repo/data/data_example_ed_report.jsonl
```
Run with a different temperature and model
```bash
python -m LLM_data_extractor_optuna_repo.main --task_id 028 --model_name qwen2.5:14b-instruct --temperature 0.5 --output_dir ./output --data_dir /path/to/your/LLM_data_extractor_optuna_repo/data/data_example_ed_report.jsonl
```
Run with a hyperparameter tuning
```bash
python -m LLM_data_extractor_optuna_repo.main --task_id 028 --model_name qwen2.5:14b-instruct --hyperparameter_tuner --output_dir ./output --data_dir /path/to/your/LLM_data_extractor_optuna_repo/data/data_example_ed_report.jsonl
```

Run multiple data files and multiple tasks using a shell script (see/adjust run_all.sh). 
```bash
bash run_all.sh
```
---

## 📂 Preparing Your Data

To run extraction tasks smoothly, your data must follow a simple structured format.

### ✅ Supported Format

- **JSONL** (JSON Lines)

### 🧾 Dataset Requirements

Each record must include:
- A **`uid`** (unique identifier)
- A **`text`** field containing the text to process

**Example:**
**`data/data_example_ed_report.jsonl`**
---
# 🛠️ Task Configuration
Create a JSON task file in the `tasks/` folder using the following naming format:
```bash
TaskXXX_taskname.json
```
---
## 📌 Required Fields
Each task file must include the following keys (see example `Task028_Palpation_Tenderness_example.json`):
- **`Label_Field`** → Column name containing the output data  
- **`Parser_Format`** → Dictionary specifying:  
  - `column` → the column containing the output text data  
  - `type` → one of `list`, `str`, `int`, or `float`  
  - `description` → prompt text used for extraction  
> ⚠️ Only modify `Label_Field` and `Parser_Format` when defining new tasks
---
## 🧾 Example File Structure
```
LLM_data_extractor_optuna_repo/
│
├── data/
│ └── data_example_ed_report.jsonl
│
├── tasks/
│ └── Task028_Palpation_Tenderness_example.json
│
├── output/
│ └── nlp_predictions_example_task028.json
│
├── run_all.sh
└── README.md

```




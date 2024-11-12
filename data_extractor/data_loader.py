import pandas as pd
from pathlib import Path
import os
import json
import re

class DataLoader:
    def __init__(self, train_path=None, test_path=None) -> None:
        self.train_path = Path(train_path) if train_path else None
        self.test_path = Path(test_path) if test_path else None
    
    def load_data(self):
        train = None
        test = None
        if self.train_path and self.train_path.exists():
            train = pd.read_json(self.train_path, lines=True)
        if self.test_path and self.test_path.exists():
            test = pd.read_json(self.test_path, lines=True)
        return train, test

class TaskLoader:
    def __init__(self, folder_path, task_id):
        self.folder_path = folder_path
        self.task_id = task_id

    def find_and_load_task(self):
        # Regex pattern to match files like TaskXXX_name.json
        pattern = re.compile(rf'.*{self.task_id}.*\.json')
        
        # List all files in the folder that match the pattern
        matching_files = [f for f in os.listdir(self.folder_path) if pattern.match(f)]

        # Check for exactly one match
        if len(matching_files) == 0:
            raise FileNotFoundError(f"No file found matching Task{self.task_id:03}*.json")
        elif len(matching_files) > 1:
            raise RuntimeError(f"Multiple files found matching Task{self.task_id:03}*.json")

        # Load the JSON file
        self.file_path = os.path.join(self.folder_path, matching_files[0])
        with open(self.file_path, 'r') as file:
            data = json.load(file)

        return data
    
    def get_task_name(self):
        return os.path.splitext(os.path.basename(self.file_path))[0]
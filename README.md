# AI Global Use – Data Workflow Guide

This document summarises how to manage the dataset with DVC and how to run the
helper utilities inside `data_cleaning.py`. Share it with teammates so we all
follow the same workflow.

## 1. DVC Usage

### 1.1 Prerequisites
- Install the project dependencies (Python ≥3.10 recommended). Activate whatever
  virtual environment you use (conda, venv, poetry, etc.), then run:
  ```
  pip install -r requirements.txt
  pip install dvc[gdrive]
  ```
- Make sure you have access to the shared Google Drive folder that stores the
  DVC cache. The project uses Google Drive folder ID: `1MWFvctU_Htge8k3MXDz1Y02qzK3eb1qA`

### 1.2 Configure the Remote (one-time per clone)
If the remote is not already configured, set it up inside the repo root:
```
dvc remote add -d gdrive gdrive://1MWFvctU_Htge8k3MXDz1Y02qzK3eb1qA
```

Your `.dvc/config` should contain:
```ini
remote = gdrive
['remote "gdrive"']
  url = gdrive://1MWFvctU_Htge8k3MXDz1Y02qzK3eb1qA
```

- On first push/pull DVC will open a browser window for OAuth. Sign in using the
  account that has access to the folder.
- If your project uses a shared Google Drive folder, you must have access to that folder.

Optional service-account setup:
```
dvc remote modify gdrive gdrive_use_service_account true
dvc remote modify gdrive gdrive_service_account_json_path path\to\key.json
```

### 1.3 Typical Commands
- Fetch data objects referenced in `.dvc` files:
  ```
  dvc pull
  ```
  The **first time**, DVC will:
  - open a browser window
  - ask you to authenticate with Google
  - ask permission to access files in that Drive folder
  
  After running `dvc pull`, DVC will download the actual file (e.g. `gpo-ai-data.csv`)
  described inside the `.dvc` file.

- Track new/updated data:
  ```
  dvc add gpo-ai-data.csv
  git add gpo-ai-data.csv.dvc .gitignore
  ```
- Push tracked data to the remote cache:
  ```
  dvc push
  ```
  If you see `File not found` errors, confirm the folder ID or that your account
  has access. For authentication issues, delete
  `%LOCALAPPDATA%\pydrive2fs\*` and try again.
- To list configured remotes:
  ```
  dvc remote list
  ```

### 1.4 Work With the Data Normally
Now that the file exists locally, you can open it like any other file:

```py
import pandas as pd

df = pd.read_csv("gpo-ai-data.csv")
```

Proceed with your normal data science workflow.

## 2. Column Harmonisation Helper (`data_cleaning.py`)

The script normalises free-text survey answers so that each entry matches one of
the canonical labels defined in `info/<column>.txt`. It translates inputs to
English, uses sentence embeddings to score label similarity, and records a full
audit report.

### 2.1 Before Running
1. Ensure the dictionary file exists (e.g. `info/respondent_industry.txt`).
2. Review the configuration block at the bottom of `data_cleaning.py`:
   ```python
   COLUMN_NAME = "respondent_industry"
   CSV_PATH = "gpo-ai-data.csv"
   INFO_DIRECTORY = "info"
   OUTPUT_PATH = None  # optional custom path
   SIMILARITY_THRESHOLD = 0.10  # tweak to control strictness
   REPORT_PATH = None  # defaults to output/<column>_similarity_report.csv
   ```
   - Set `OUTPUT_PATH` if you want a specific destination; otherwise the script
     writes `<original>_<column>_harmonised.csv`.
   - Adjust `SIMILARITY_THRESHOLD` to tune when entries fall back to `*Other`.
   - Set `REPORT_PATH` to `None` to use the default path in the `output/` directory.

### 2.2 Running the Helper
Execute from the repo root (after activating your preferred environment):
```
pip install -r requirements.txt  # skip if already done
python data_cleaning.py
```
You will see a progress bar while the column is processed.

Outputs:
- Harmonised CSV beside the original (or at `OUTPUT_PATH`).
- A similarity report saved under `output/<column>_similarity_report.csv` (or
  the custom path you set) with columns:
  - `original_value`: raw survey entry (empty string for blank rows).
  - `translated_value`: English translation used for scoring.
  - `detected_language`: language code detected by `langdetect`.
  - `harmonised_value`: canonical label chosen from the dictionary.
  - `cosine_similarity`: match strength; higher is better.
  - `used_fallback`: `True` when the entry fell back to the `*Other` label.

### 2.3 Tips
- Large dictionaries or noisy data may require lowering the threshold (e.g.
  `0.1`) to avoid excessive fallback assignments.
- The translation and embedding models cache results; reruns are faster when the
  column contains repeated values.
- If you need to harmonise a different column, create the corresponding
  dictionary file (one label per line, prefix fallback with `*`), update
  `COLUMN_NAME`, and rerun the script.
- Once you are satisfied with the harmonised output, rename the file back to
  `gpo-ai-data.csv` so downstream steps keep working. DVC will track this new
  version—remember to rerun `dvc add gpo-ai-data.csv` and `dvc push`.

## 3. Typical Workflow Summary

**Downloading data**
```bash
git clone git@github.com:YC-Qiu/ai-global-use.git
cd ai-global-use
pip install -r requirements.txt
pip install dvc[gdrive]
dvc pull
```

**Making changes**
```bash
python data_cleaning.py
# or your own scripts
```

**Saving updated dataset**
```bash
dvc add gpo-ai-data.csv
git add gpo-ai-data.csv.dvc .gitignore
git commit -m "Update dataset"
dvc push
```

Feel free to expand this README with additional project-specific notes as we add
more processing steps.

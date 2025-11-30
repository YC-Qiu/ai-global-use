# How to Work with this Project (DVC + Google Drive)

## 1. Install DVC (with Google Drive support)

**Mac / Linux**

```bash
pip install "dvc[gdrive]"
```

**Windows**

```bash
pip install "dvc[gdrive]"
```

Verify installation:

```bash
dvc --version
```

## 2. Clone the GitHub repository

If you have not already:

```bash
git clone git@github.com:YC-Qiu/ai-global-use.git
cd ai-global-use
```

## 3. Authenticate DVC with Google Drive

Your `.dvc/config` says:

```ini
remote = gdrive
['remote "gdrive"']
  url = gdrive://1MWFvctU_Htge8k3MXDz1Y02qzK3eb1qA
```

This means the data is stored in a **Google Drive folder** with that ID.

Run:

```bash
dvc pull
```

The **first time**, DVC will:

* open a browser window

* ask you to authenticate with Google

* ask permission to access files in that Drive folder

If your project uses a shared Google Drive folder, you must have access to that folder.

## 4. After Pulling the Data, Your Missing CSV Appears

You currently have:

```kotlin
cleaned-gpo-ai-data-nov-27.csv.dvc
```

After running:

```bash
dvc pull
```

DVC will download the actual file described inside (`cleaned-gpo-ai-data-nov-27.csv`).

It uses the hash and path inside the `.dvc` file:

```yaml
outs:
- md5: e873702c0b8b33f793c02e1108ab29a4
  size: 80359002
  path: cleaned-gpo-ai-data-nov-27.csv
```

This tells to put the recovered data file at:

```txt
cleaned-gpo-ai-data-nov-27.csv
```

## 5. Work With the Data Normally

Now that the file exists locally, you can open it like any other file:

```py
import pandas as pd

df = pd.read_csv("cleaned-gpo-ai-data-nov-27.csv")
```

Proceed with your normal data science workflow.

## 6. If You Modify the Data

You must tell DVC the file changed:

```bash
dvc add cleaned-gpo-ai-data-nov-27.csv
```

This updates:

* the `.dvc` metafile

* the DVC cache locally

Then push changes:

```bash
git add cleaned-gpo-ai-data-nov-27.csv.dvc
git commit -m "Update dataset"
dvc push
```

## 7. Typical Workflow Summary

**Downloading data**

```bash
git clone <repo>
cd <repo>
pip install "dvc[gdrive]"
dvc pull
```

**Making changes**

```bash
python script.py
```

**Saving updated dataset**

```bash
dvc add data.csv
git add data.csv.dvc
git commit -m "Update dataset"
dvc push
```

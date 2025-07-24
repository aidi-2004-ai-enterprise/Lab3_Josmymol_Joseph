# Lab 3: Penguins Classification with XGBoost and FastAPI

## 📌 Overview

This project implements a machine learning pipeline using the Seaborn Penguins dataset. An XGBoost model is trained to classify penguin species based on physical measurements. The model is deployed via a FastAPI application with robust input validation and logging.

---

## 🧠 Learning Objectives

- Load and preprocess a dataset using one-hot and label encoding.
- Train and evaluate an XGBoost classifier.
- Deploy a prediction API using FastAPI with Pydantic for input validation.
- Handle invalid inputs gracefully with proper HTTP responses.
- Manage dependencies using `uv` and organize the project structure professionally.

---

## 🗂️ Project Structure
```bash
Lab3_Josmymol_Joseph/
├── train.py
├── app/
│ ├── main.py
│ ├── data/
│ │ └── model.json
├── pyproject.toml
├── README.md
└── demo.mp4 ← (screen recording of test cases)
```


---

## ✅ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/aidi-2004-ai-enterprise/lab3_Josmymol_Joseph.git
cd lab3_Josmymol_Joseph
```
2. Create and Activate Virtual Environment
```bash
uv venv
.venv/Scripts/Activate.ps1        # PowerShell
# or .venv/Scripts/activate.bat   # Command Prompt
```
3. Install Dependencies
```bash
uv install
```
🚀 Running the App
1. Train the Model
bash
Copy
Edit
python train.py
2. Launch FastAPI Server
bash
Copy
Edit
uvicorn app.main:app --reload
3. Open in Browser
Visit:

arduino
Copy
Edit
http://127.0.0.1:8000/docs
Use the Swagger UI to test the /predict endpoint.

📦 Sample Request Payload
```bash
{
  "bill_length_mm": 45.2,
  "bill_depth_mm": 14.5,
  "flipper_length_mm": 210,
  "body_mass_g": 4200,
  "year": 2009,
  "sex": "male",
  "island": "Biscoe"
}
```
{
  "detail": [
    {
      "loc": ["body", "sex"],
      "msg": "value is not a valid enumeration member; permitted: 'male', 'female'",
      "type": "type_error.enum"
    }
  ]
}

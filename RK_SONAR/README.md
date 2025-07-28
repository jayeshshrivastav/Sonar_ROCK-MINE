# 🪨 Sonar Rock vs Mine Classifier

This project uses a Logistic Regression model to classify sonar signals as either **Rock** or **Mine** based on 60 numerical features from sonar data.

---

## 📁 Project Structure

sonar_classifier/
├── Dataset/
│ └── sonar_data.csv # Original dataset
├── model/
│ ├── sonar_model.pkl # Trained Logistic Regression model
│ └── scaler.pkl # StandardScaler object for input normalization
├── train.py # Script to train and save the model
├── predict.py # Script to make predictions using the trained model
├──requirements.txt # PKG Requirements File 
└── README.md # This file


---

## 📊 Dataset

- The dataset contains 208 rows and 61 columns.
- Each row consists of 60 sonar readings (`float`) followed by a label (`R` for Rock or `M` for Mine).
- [Source](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))

---

## ⚙️ Requirements

Install dependencies (only standard libraries used):

```bash
pip install -requirements.txt
```

## Usage 
Clone Repo:

```bash
git clone "repo url"
```

Install Requirements:


Run Train.py 
```bash 
python train.py 
```

Run Predict.py 
```bash 
python pridict.py <Provide  60 samples split by , >
```

Example:
```bash
python predict.py 0.0262,0.0582,0.1099,0.1083,0.0974,0.2280,0.2431,0.3771,0.5598,0.6194,0.6333,0.7060,0.5544,0.5320,0.6479,0.6931,0.6759,0.7551,0.8929,0.8619,0.7974,0.6737,0.4293,0.3648,0.5331,0.2413,0.5070,0.8533,0.6036,0.8514,0.8512,0.5045,0.1862,0.2709,0.4232,0.3043,0.6116,0.6756,0.5375,0.4719,0.4647,0.2587,0.2129,0.2222,0.2111,0.0176,0.1348,0.0744,0.0130,0.0106,0.0033,0.0232,0.0166,0.0095,0.0180,0.0244,0.0316,0.0164,0.0095,0.0078

```
# 📧 Phishing Email Detection Using Machine Learning  

## 📌 Project Overview  
This project aims to detect **phishing emails** using machine learning models. By analyzing email text data, we train classifiers to distinguish between **safe emails** and **phishing emails** effectively. The project includes **preprocessing**, **feature engineering**, **model training**, and **evaluation** steps to ensure high accuracy.  

## 📂 Dataset Overview  
- **Total Emails:** 18,650  
- **Safe Emails:** 61%  
- **Phishing Emails:** 39%  
- **Null/Empty Values:** 3% (Handled during preprocessing)  

## ⚙️ Preprocessing Steps  
- **Text Cleaning:** Tokenization, stopword removal, lemmatization.  
- **Vectorization:** Converting text into numerical format using **TF-IDF**.  
- **Feature Optimization:** Finding the best feature size for accuracy.  

## 🧠 Machine Learning Models Used  
We experimented with multiple classification models:  
- **Support Vector Classifier (SVC - SVM)**  
- **Random Forest Classifier**  
- **Logistic Regression**  
- **Multinomial Naive Bayes**  

## 📊 Model Performance Comparison  
| Model | Precision | Recall | F1-Score | Accuracy |  
|------------|------------|------------|------------|------------|  
| **SVC (SVM)** | 0.99 | 0.97 | 0.98 | **98%** |  
| **Random Forest** | 0.99 | 0.96 | 0.97 | 97% |  
| **Logistic Regression** | 0.99 | 0.96 | 0.97 | 97% |  
| **Naive Bayes** | 0.90 | 0.98 | 0.94 | 92% |  

## 📊 Feature Optimization  
- **Exact feature size after vectorization:** **7543**  
- **Optimized feature size for best accuracy:** **6200**  
- **Best Model:** **SVC with 98% accuracy after optimization**  

## 📌 Project Structure  
📁 phishing-email-detection
│── 📂 data # Dataset and preprocessing scripts
│── 📂 models # Trained models and evaluation metrics
│── 📂 notebooks # Jupyter notebooks for EDA & model training
│── 📜 requirements.txt # Dependencies for the project
│── 📜 README.md # Project documentation
│── 📜 main.py # Main script for training and evaluation

## 📌 Future Improvements  
- Implementing **real-time phishing detection** in emails.  
- Experimenting with **deep learning models (LSTMs, Transformers)**.  
- Enhancing **hyperparameter tuning** for better efficiency.  

## 📜 License  
This project is open-source and available under the **MIT License**.  

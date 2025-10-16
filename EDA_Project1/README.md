# ⚾ MLB Swing Probability Prediction

This project develops a **machine learning model** to predict the probability that a batter will swing at a pitch.  
Using MLB pitch-tracking data from pitcher **Zac Gallen**, the model estimates swing probabilities based on **pitch characteristics** and **in-game context**, with the goal of supporting **data-driven pitching strategies**.

---

## 📌 Project Overview

- **Objective:** Build a calibrated probability model to predict whether a batter will swing at a given pitch.  
- **Data:** Simulated dataset based on real MLB pitch distributions (Zac Gallen).  
- **Features:** Pitch type, release speed, spin rate, release position, movement, and strike zone location.  
- **Output:** Predicted swing probability (0–1) for each pitch.  

---

## 🧭 Project Structure

mlb-swing-probability/
│
├─ data/ # Raw and processed datasets
├─ notebooks/ # Jupyter notebooks for EDA and modeling
├─ src/ # Python scripts (preprocessing, training, evaluation)
├─ models/ # Trained model files (.joblib)
├─ figures/ # Saved figures and plots
├─ reports/ # Project summary and slides
├─ README.md # Project documentation
└─ requirements.txt # Dependencies



---

## 📊 Exploratory Data Analysis (EDA)

Key patterns identified:

- **Pitch Type:** Changeups had the highest swing rates, while fastballs showed lower rates.  
- **Release Speed:** Bimodal distribution (off-speed vs. fastballs), reflecting realistic pitching strategy.  
- **Release Position:** Concentrated release zone with slight horizontal and vertical variance — critical for deception.

📈 **Example Visualizations**:
- `proportion_of_swings_by_pitch_type.png`  
- `distribution_of_release_speed.png`  
- `heatmap_of_release_position.png`

---

## 🤖 Modeling Approach

- **Model:** Random Forest Classifier  
- **Preprocessing:**  
  - Numerical features scaled with `StandardScaler`  
  - Categorical variables encoded with `OneHotEncoder`  
- **Tuning:** `GridSearchCV` with 5-fold cross-validation  
- **Evaluation Metrics:**  
  - Expected Calibration Error (ECE)  
  - Maximum Calibration Error (MCE)  
  - Brier Score  
  - Log Loss  
  - ROC AUC

---

## 📈 Results

| Metric                           | Value     |
|-----------------------------------|-----------|
| Expected Calibration Error (ECE) | 0.134 |
| Maximum Calibration Error (MCE)  | 0.210 |
| Brier Score                      | 0.111 |
| Log Loss                         | 0.377 |
| ROC AUC                           | 0.954 |

- ROC curve indicates **strong discriminative ability**.  
- ECE/MCE are close to acceptable thresholds but indicate room for calibration improvement.

📉 **Saved Plots**:
- `roc_curve.png`  
- `model_evaluation_metrics.png`  
- `top_10_feature_importances.png`

---

## 🧠 Feature Importance Insights

Top predictive features:

| Feature            | Importance |
|--------------------|------------|
| `plate_x`          | 0.208 |
| `plate_z`          | 0.149 |
| `sz_top` / `sz_bot`| ≈ 0.10 |
| `pfx_z`, `release_speed`, `pfx_x` | Moderate impact |

👉 Pitch **location** is the primary driver of swing decisions, aligning with real baseball intuition.

---

## 🧪 Discussion & Conclusion

If this model were to be applied in a real MLB coaching context, the focus should be on variables that pitchers can **directly control** or **significantly influence**.  
Key features such as **release extension**, **release speed**, and **pitch type** are pitcher-driven and can be strategically adjusted to increase swing probability.  
Release position also plays a critical role in how a pitch is perceived.

The current model provides **strong predictive power** but falls slightly short of the **calibration standards** required for deployment.  
Further improvements — such as isotonic regression, Platt scaling, or enhanced feature engineering — could turn this into a reliable decision-support tool for pitching strategies.

---

## ⚙️ How to Run

1. **Clone the repo**
   git clone https://github.com/yourusername/mlb-swing-probability.git
   cd mlb-swing-probability

2. **Install dependencies**


pip install -r requirements.txt

3. **Run the notebook**

jupyter notebook notebooks/EDA_and_Modeling.ipynb

4.**View saved figures**

figures/

## 🏅 Project Highlights
End-to-end ML workflow: EDA → preprocessing → model training → evaluation → interpretation.

Integration of probability calibration for more reliable predictions.

Clear and modular project structure for reproducibility.

Strong ROC AUC and interpretable feature importance results.

## 📚 Tech Stack
Python 3

scikit-learn

pandas / numpy

matplotlib / seaborn

joblib

## 👤 Author
Dongjing Wen
🎓 MSBA Student @ UIUC
📧 your.email@example.com
🔗 LinkedIn
📂 Portfolio

## 📄 License
This project is released under the MIT License.
Feel free to use, adapt, and share with attribution.
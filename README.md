# 📩 SMS Spam Classifier (FastAPI Demo)

This is a simple **SMS Spam Classifier** built with **scikit-learn** and served using **FastAPI**.  
It uses a trained pipeline (`TfidfVectorizer + LogisticRegression`) to classify text messages as **SPAM** or **HAM (not spam)**.

🚀 **Live Demo**: [https://inference-service.onrender.com/](https://inference-service.onrender.com/)  
Paste a message and click **Classify** to see predictions.

---

## 🔹 Example Test

Try this sample SMS in the demo page:

WINNER! You have won a $1000 Walmart gift card. Click http://bit.ly/fake


**Expected Result:**
```json
{
  "prediction": "SPAM",
  "prob": 0.85
}
```
(Your probability may differ slightly, but it should classify as SPAM.)
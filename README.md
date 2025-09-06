# SMS Spam Classifier (FastAPI Demo)

A simple machine learning project demonstrating **end-to-end model deployment**:

- Trained an SMS spam detection model using **scikit-learn** (TF-IDF + Logistic Regression).
- Packaged model with **joblib** and exposed via a **FastAPI** inference API.
- Deployed live on **Render free tier**, with token-authenticated REST endpoint and demo UI.


**Live Demo**: [https://inference-service.onrender.com/](https://inference-service.onrender.com/)  
Paste a message and click **Classify** to see predictions.

---

## Example Test

Try this sample SMS in the demo page:

```WINNER! You have won a $1000 Walmart gift card. Click http://bit.ly/fake```


**Expected Result:**
```json
{
  "prediction": "SPAM",
  "prob": 0.85
}
```
(Your probability may differ slightly, but it should classify as SPAM.)
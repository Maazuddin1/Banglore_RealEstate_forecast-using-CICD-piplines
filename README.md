---
layout: default
title: Real Estate Price Prediction API
description: A machine learning powered real estate price prediction tool with web interface
---

# 🏠 **Real Estate Price Prediction API**

## 🌟 **Project Motto**
This project aims to provide an accurate and interactive **Real Estate Price Prediction tool**. Users can input details such as property location, square footage, number of bedrooms, and bathrooms to get an **instant price prediction** based on a trained **machine learning model**.  

This API bridges the gap between **data science** and **user-friendly deployment**, allowing seamless integration of advanced predictions into real-world applications.  

---

## 💡 **How It Works**

1. **Data Processing & Model Training**  
   - A dataset of real estate transactions was cleaned and processed.  
   - Key features such as `location`, `total_sqft`, `bath`, and `bhk` were selected.  
   - A **Linear Regression model** was trained and stored as a `.pkl` file for deployment.  

2. **Prediction Mechanism**  
   - The trained model is loaded and predicts property prices based on user inputs.  
   - Location data is one-hot encoded to handle categorical features.  

3. **Interactive Frontend**  
   - A Flask-powered web app provides an intuitive interface for predictions.  
   - Users input details via forms, and results are displayed instantly.  

4. **API Integration**  
   - A `/predict` endpoint allows developers to integrate the model with other applications.
---

## 🎥 Watch the Demo(click image below👇)

[![Watch on YouTube](https://img.youtube.com/vi/NcmXkE907io/0.jpg)](https://www.youtube.com/watch?v=NcmXkE907io)


---

## 📷 **Screenshots**
### Home Page
![Home Page](images/homepage.png)

### Prediction Results
![Prediction Result](images/predicted_results.jpg)

---

## 📂 **Project Structure**

```
├── .github/
│   └── workflows/
│       └── python-app.yml          # CI/CD workflow configuration
├── data/                           # Dataset directory
│   └── bengaluru_house_prices.csv  # Dataset file for the project
├── models/                         # Saved models and feature names
│   ├── feature_names.pkl           # Pickled feature names
│   └── lr_regg.pkl                 # Trained regression model
├── src/                            # Source code for the project
│   ├── EDA.py                      # Exploratory Data Analysis script
│   ├── model.py                    # Model training and evaluation script
│   └── preprocessing.py            # Data preprocessing logic
├── templates/                      # HTML templates for the Flask web app
│   ├── index.html                  # User input form for predictions
│   └── results.html                # Displays prediction results
├── tests/                          # Unit testing for the project
│   ├── __init__.py                 # Marks the directory as a package
│   ├── test_model.py               # Tests for the model
│   └── test2direct.py              # Additional test script
├── .gitignore                      # Specifies ignored files for Git
├── app.py                          # Flask application entry point
├── main.py                         # Main execution script
├── requirements.txt                # List of dependencies for the project
├── setup.py                        # Setup script for packaging the project
├── README.md                       # Project overview and documentation

```

---

## 🚀 **Features**
- **Accurate Price Predictions** using a trained regression model.  
- **Interactive Web Interface** for user-friendly predictions.  
- **API Integration** for developers to use the model programmatically.  
- **Scalable and Extendable** to new locations or additional features.  

---

## 🛠️ **Installation and Setup**

### Prerequisites  
- Python 3.8+  
- Flask  
- Pickle  

### Installation Steps  
1. Clone the repository:  
   ```bash
 git clone https://github.com/Maazuddin1/Banglore_RealEstate_forecast-using-CICD-piplines.git  
 cd Banglore_RealEstate_forecast-using-CICD-piplines
 
   ```

2. Create a virtual environment:  
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate     # Windows
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask application:  
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000/`.  

---

## 🌐 **API Usage**

### Endpoint: `/predict`  
**Method**: `POST`  
**Input** (JSON):  
```json
{
  "location": "Whitefield",
  "sqft": 1200,
  "bath": 2,
  "bhk": 3
}
```

**Output**:  
```json
{
  "predicted_price": 94.23 Lakhs
}
```

---

## 🔍 **Model Details**
The trained model uses **Linear Regression** with key features like:
- **total_sqft**: Total square footage of the property.  
- **bath**: Number of bathrooms.  
- **bhk**: Number of bedrooms.  
- **Location**: One-hot encoded for categorical support.  

---

## 📈 **Future Enhancements**
- Add support for more advanced machine learning models like Random Forest or XGBoost.  
- Improve UI design with frameworks like Bootstrap.  
- Expand location datasets for better predictions.  
- Add real-time price scraping for dynamic updates.  

---

## 🖼️ **Visual Workflow**
```mermaid
graph TD
A[User Input] --> B[Flask App]
B --> C[Process Input Features]
C --> D[Trained ML Model]
D --> E[Predict Price]
E --> F[Display Results]
```

---

## 🌟 **Contributions**  
Contributions are welcome! Feel free to fork this repository, open issues, or submit pull requests.

---

## 📄 **License**
-
---

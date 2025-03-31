import numpy as np
import streamlit as st
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# 🔹 Streamlit UI
st.title("🏦 FinanceIQ: Banking Fraud & Churn Detection")

# 🔹 Sidebar Navigation
options = [
    "Credit Scoring", "Fraud Detection", "Customer Churn Prediction",
    "Loan Default Prediction", "Customer Segmentation", "Investment Optimization",
    "Sentiment Analysis", "Dynamic Pricing"
]
selected_option = st.sidebar.selectbox("📌 Select a feature", options)

# 🔹 Load trained models
model_paths = {
    "Credit Scoring": r"C:\Users\Mohit\OneDrive\Desktop\Bank fraud detection\models\credit_scoring_dataset.pkl",
    "Fraud Detection": r"C:\Users\Mohit\OneDrive\Desktop\Bank fraud detection\models\fraud_detection_model.pkl",
    "Churn Prediction": r"C:\Users\Mohit\OneDrive\Desktop\Bank fraud detection\models\churn_model.pkl",
    "Loan Default Prediction": r"C:\Users\Mohit\OneDrive\Desktop\Bank fraud detection\models\loan_default.pkl",
    "Customer Segmentation": r"C:\Users\Mohit\OneDrive\Desktop\Bank fraud detection\models\kmeans_model.pkl",
    "Investment Optimization": r"C:\Users\Mohit\OneDrive\Desktop\Bank fraud detection\models\investment_model.pkl",
    "Dynamic Pricing": r"C:\Users\Mohit\OneDrive\Desktop\Bank fraud detection\models\bank_dynamic_pricing.pkl",
    "Sentiment Analysis": r"C:\Users\Mohit\OneDrive\Desktop\Bank fraud detection\models\sentiment_model.pkl",
}

SCALER_PATH = r"C:\Users\Mohit\OneDrive\Desktop\Bank fraud detection\models\scaler.pkl"

models = {}
for model_name, path in model_paths.items():
    try:
        if model_name == "Customer Segmentation":
            models[model_name] = joblib.load(path)
        else:
            with open(path, 'rb') as f:
                models[model_name] = pickle.load(f)
        st.sidebar.success(f"✅ {model_name} model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading {model_name} model: {e}")


# =====================================================
# ✅ 1. Credit Scoring (Loan Approval Prediction)
# =====================================================
if selected_option == "Credit Scoring":
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    st.subheader("🏦 Credit Scoring: Loan Approval Prediction")

    # Load dataset
    df = pd.read_pickle(r"C:\Users\Mohit\OneDrive\Desktop\Bank fraud detection\models\credit_scoring_dataset.pkl")

    # Encode categorical variables
    df["Employment_Status"] = df["Employment_Status"].map({"Employed": 0, "Self-Employed": 1, "Unemployed": 2})
    df["Loan_Approval"] = df["Loan_Approval"].map({"Approved": 1, "Rejected": 0})

    # Split dataset into features and target
    X = df.drop("Loan_Approval", axis=1)
    y = df["Loan_Approval"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # User input form in Streamlit
    age = st.number_input("Age", min_value=21, max_value=65, value=30)
    income = st.number_input("Annual Income ($)", min_value=20000, max_value=150000, value=50000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    loan_amount = st.number_input("Loan Amount ($)", min_value=5000, max_value=50000, value=20000)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
    existing_loans = st.number_input("Existing Loans", min_value=0, max_value=5, value=1)
    dti_ratio = st.slider("Debt-to-Income Ratio", 0.1, 0.6, 0.3)

    # Encode user input
    employment_status_map = {"Employed": 0, "Self-Employed": 1, "Unemployed": 2}
    input_data = np.array([[age, income, credit_score, loan_amount, loan_term, employment_status_map[employment_status], existing_loans, dti_ratio]])

    # Predict approval
    if st.button("Predict Loan Approval"):
        prediction = model.predict(input_data)
        result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
        st.subheader(f"Loan Status: {result}")
        st.write(f"Model Accuracy: {accuracy:.2%}")


# =====================================================
# ✅ 2. Fraud Detection (IsolationForest)
# =====================================================
elif selected_option == "Fraud Detection": 
    st.subheader("🔍 Fraud Detection") 

    # User inputs
    transaction_amount = st.number_input("💰 Transaction Amount ($)", min_value=0)
    
    # Dropdown for selecting transaction location
    transaction_location = st.selectbox("📍 Transaction Location", 
                                        ["Illinois", "Texas", "California", "Ohio", 
                                         "Florida", "Nevada", "New York"])
    
    # Dropdown for selecting merchant details
    merchant_details = st.selectbox("🏪 Merchant Details", 
                                    ["Luxury Store", "Target", "Electronics Store", "McDonald's", 
                                     "Gas Station", "Amazon", "Apple Store", "Hotel Booking", 
                                     "Car Dealer", "Walmart", "Cinema", "Best Buy", "Local Market", 
                                     "Crypto Exchange", "Jewelers", "Starbucks", "Online Store", 
                                     "Car Rental"])

    user_behavior = st.selectbox("🛒 User Behavior", ["Normal", "Suspicious"])

    # Encoding categorical data
    location_encoded = hash(transaction_location) % 1000
    merchant_encoded = hash(merchant_details) % 1000
    user_behavior_encoded = 1 if user_behavior == "Suspicious" else 0

    # Prepare input data
    input_data = pd.DataFrame({
        'transaction_amount': [transaction_amount],
        'location': [location_encoded],
        'merchant_details': [merchant_encoded],
        'user_behavior': [user_behavior_encoded]
    })

    # Detect Fraud
    if st.button("⚠️ Detect Fraud"):
        try:
            model = models.get("Fraud Detection")
            if model:
                anomaly_score = model.decision_function(input_data)  
                fraud_prediction = model.predict(input_data)  

                # 🔹 Adjusted fraud detection threshold
                THRESHOLD = -0.1  

                # Fraud detection logic
                if fraud_prediction[0] == -1 or anomaly_score[0] < THRESHOLD:
                    st.error("🚨 **Fraudulent transaction detected!** ⚠️")
                else:
                    st.success("✅ **Transaction is legitimate.**")
            else:
                st.error("⚠️ Model not loaded properly.")
        except Exception as e:
            st.error(f"⚠️ Fraud prediction failed: {e}")



# =====================================================
# ✅ 3. Customer Churn Prediction
# =====================================================
elif selected_option == "Customer Churn Prediction":
    st.subheader("📉 Customer Churn Prediction")

    # User inputs
    account_activity = st.number_input("📊 Account Activity (1-100)", min_value=1, max_value=100, step=1)
    service_usage = st.number_input("📡 Service Usage (1-100)", min_value=1, max_value=100, step=1)
    customer_feedback = st.number_input("💬 Customer Feedback Score (1-5)", min_value=1, max_value=5, step=1)
    demographics = st.selectbox("👤 Demographics", ["Young", "Senior"])

    # Encode demographics correctly
    demographic_features = {
        "demographics_young": 1 if demographics == "Young" else 0,
        "demographics_senior": 1 if demographics == "Senior" else 0
    }

    # Create input DataFrame
    input_data = pd.DataFrame({
        'account_activity': [account_activity],
        'service_usage': [service_usage],
        'customer_feedback': [customer_feedback],
        **demographic_features
    })

    # Ensure correct feature order
    if st.button("🔍 Predict Churn"):
        try:
            model = models.get("Churn Prediction")
            if model:
                expected_features = list(model.feature_names_in_)
                input_data = input_data.reindex(columns=expected_features, fill_value=0)
                prediction = model.predict(input_data)

                result = "⚠️ Customer is likely to churn!" if prediction[0] == 1 else "✅ Customer is retained."
                if prediction[0] == 1:
                    st.warning(result)
                else:
                    st.success(result)

            else:
                st.error("⚠️ Model not loaded properly.")
        except Exception as e:
            st.error(f"⚠️ Churn prediction failed: {e}")

# =====================================================
# ✅ 4. Loan Defualt Prediction
# =====================================================
elif selected_option == "Loan Default Prediction":
    st.subheader("📉 Loan Default Prediction")

    # User Inputs
    credit_score = st.number_input("📊 Credit Score (300-850)", min_value=300, max_value=850, step=10)
    debt_ratio = st.number_input("💰 Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, step=0.1)
    income = st.number_input("🏦 Annual Income ($)", min_value=0, step=1000)
    loan_amount = st.number_input("💵 Loan Amount ($)", min_value=0, step=1000)

    # Load Model
    model = models.get("Loan Default Prediction")

    # Prediction Button
    if st.button("🔍 Predict Default Risk"):
        try:
            if model:
                input_data = pd.DataFrame({
                    'borrower_demographics': [0],  # Placeholder
                    'loan_amount': [loan_amount],
                    'income': [income],
                    'credit_score': [credit_score],
                    'repayment_history': [0],  # Placeholder
                    'economic_indicators': [0]  # Placeholder
                })

                prediction = model.predict(input_data)
                result = "⚠️ High Default Risk" if prediction[0] == 1 else "✅ Low Default Risk"
                
                # Display Result
                if prediction[0] == 1:
                    st.warning(result)
                else:
                    st.success(result)
            else:
                st.error("⚠️ Loan Default Prediction model not loaded properly.")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")



# =====================================================
# ✅ 5. Customer Segmentation (K-Means Clustering)
# =====================================================
elif selected_option == "Customer Segmentation":
    st.subheader("📊 Customer Segmentation")

    uploaded_file = st.file_uploader("📂 Upload customer dataset (CSV format)", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")
        st.write("📊 Available columns:", data.columns.tolist())

        required_columns = ['age', 'income', 'spending_score']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            st.error(f"❌ Missing columns: {missing_columns}")
            st.stop()

        X = data[required_columns]

        # Load scaler or create a new one
        try:
            scaler = joblib.load(SCALER_PATH)
        except:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            joblib.dump(scaler, SCALER_PATH)

        X_scaled = scaler.transform(X)

        # Find optimal clusters
        def elbow_method():
            inertia = []
            k_values = range(1, 11)
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(k_values, inertia, marker='o', linestyle='--', color='b')
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow Method for Optimal K")
            return fig

        if st.button("🔍 Find Optimal Clusters (Elbow Method)"):
            fig = elbow_method()
            st.pyplot(fig)

        optimal_k = st.slider("Select the number of clusters (K)", min_value=2, max_value=10, value=3)

        if st.button("🚀 Perform Customer Segmentation"):
            try:
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                data['Cluster'] = kmeans.fit_predict(X_scaled)

                # Save model
                joblib.dump(kmeans, model_paths["Customer Segmentation"])
                joblib.dump(scaler, SCALER_PATH)
                st.success(f"✅ Segmentation complete! Model saved.")

                st.write("📌 Sample segmented data:")
                st.dataframe(data.head())

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=data['age'], y=data['spending_score'], hue=data['Cluster'], palette='viridis', s=100, ax=ax)
                ax.set_xlabel("Age")
                ax.set_ylabel("Spending Score")
                ax.set_title("Customer Segmentation Clusters")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error: {e}")

# =====================================================
# ✅ 6. Investment Optimization Model
# =====================================================
elif selected_option == "Investment Optimization":
    st.subheader("Investment Optimization")

    import os
    import pickle
    import numpy as np

    # Define the model path
    model_file = r"C:\Users\Mohit\OneDrive\Desktop\Bank fraud detection\models\investment_model.pkl"

    # Function to load the model
    def load_model():
        if not os.path.exists(model_file):
            return None, "⚠️ Model file not found. Please train and save the model first."
        try:
            with open(model_file, 'rb') as file:
                model = pickle.load(file)
            return model, None
        except (EOFError, pickle.UnpicklingError):
            return None, "⚠️ Model file is corrupted. Retrain and save again."

    # Load the model
    model, error = load_model()

    # User Inputs
    risk_tolerance = st.slider("Risk Tolerance (0 to 1)", 0.0, 1.0, 0.5)
    economic_indicators = st.slider("Economic Indicators (0 to 1)", 0.0, 1.0, 0.5)

    if model is None:
        st.error(error)
    else:
        # Predict investment return
        input_features = np.array([[risk_tolerance, economic_indicators]])
        predicted_return = model.predict(input_features)

        # Display Result
        st.subheader("Predicted Investment Return")
        st.success(f"💰 Estimated Return: **{predicted_return[0]:.2f}**")

# =====================================================
# ✅ 7. Sentiment Analysis
# =====================================================
elif selected_option == "Sentiment Analysis":
    import os
    import pickle
    import streamlit as st
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    st.subheader("📊 Sentiment Analysis")

    # Define model paths
    model_path = "C:\\Users\\Mohit\\OneDrive\\Desktop\\Bank fraud detection\\models\\sentiment_model.pkl"
    vectorizer_path = "C:\\Users\\Mohit\\OneDrive\\Desktop\\Bank fraud detection\\models\\vectorizer.pkl"

    # Load the trained model & vectorizer
    model_loaded = False
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        model_loaded = True
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please train and save 'sentiment_model.pkl' and 'vectorizer.pkl'.")

    # User Inputs for Sentiment Analysis
    user_feedback = st.text_input("💬 Enter customer feedback:")

    # Additional Fields
    customer_type = st.selectbox("👤 Customer Type", ["Regular", "Premium", "VIP"])
    issue_type = st.selectbox("⚠️ Type of Issue", ["Transaction Error", "Account Lock", "Loan Query", "Card Declined", "Other"])
    resolution_time = st.slider("⏳ Resolution Time (in hours)", 0, 72, 5)
    satisfaction_rating = st.slider("⭐ Customer Satisfaction Rating (1-5)", 1, 5, 3)

    # Predict Sentiment
    if st.button("Analyze Sentiment") and model_loaded:
        if user_feedback.strip():
            input_vector = vectorizer.transform([user_feedback])
            prediction = model.predict(input_vector)[0]

            sentiment_label = "😊 Positive Sentiment" if prediction == "positive" else "😠 Negative Sentiment" if prediction == "negative" else "😐 Neutral Sentiment"

            # Display Analysis Results
            st.markdown(f"**💬 Customer Feedback Sentiment:** {sentiment_label}")
            st.markdown(f"**👤 Customer Type:** {customer_type}")
            st.markdown(f"**⚠️ Issue Type:** {issue_type}")
            st.markdown(f"**⏳ Resolution Time:** {resolution_time} hours")
            st.markdown(f"**⭐ Satisfaction Rating:** {satisfaction_rating} / 5")

        else:
            st.warning("⚠️ Please enter some feedback to analyze.")


# =====================================================
# ✅ 8. Dynamic Pricing Model
# =====================================================

elif selected_option == "Dynamic Pricing":
    import streamlit as st
    import pandas as pd
    import pickle
    import os

    st.subheader("💰 Dynamic Pricing Model")

    # Define the model path
    model_path = "C:/Users/Mohit/OneDrive/Desktop/Bank fraud detection/models/bank_dynamic_pricing.pkl"

    # Load the trained model
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        model_loaded = True
    else:
        model_loaded = False
        st.error("⚠️ Model file not found. Please train and save the model as 'bank_dynamic_pricing.pkl'.")

    # User Inputs for Pricing Factors
    customer_type = st.selectbox("👤 Customer Type", ["New", "Regular", "Premium"])
    account_balance = st.number_input("💰 Account Balance ($)", min_value=0.0, value=1000.0)
    transaction_amount = st.number_input("💳 Transaction Amount ($)", min_value=0.0, value=500.0)
    competitor_pricing = st.number_input("📊 Competitor's Pricing ($)", min_value=0.0, value=20.0)
    risk_score = st.slider("⚠️ Risk Score (0 to 100)", 0, 100, 50)

    # Convert categorical customer type to numerical format
    customer_type_map = {"New": 0, "Regular": 1, "Premium": 2}
    customer_type_encoded = customer_type_map[customer_type]

    # Predict button
    if st.button("🔍 Predict Optimal Pricing") and model_loaded:
        # Prepare input data
        input_data = pd.DataFrame([[customer_type_encoded, account_balance, transaction_amount, competitor_pricing, risk_score]], 
                                  columns=['Customer_Type', 'Account_Balance', 'Transaction_Amount', 'Competitor_Pricing', 'Risk_Score'])
        
        # Predict optimal pricing
        predicted_price = model.predict(input_data)[0]
        
        # Display result
        st.success(f"💰 Recommended Optimal Price: ${predicted_price:.2f}")


else:
    st.subheader(selected_option)
    st.info("🚧 This feature is under development. Stay tuned!")

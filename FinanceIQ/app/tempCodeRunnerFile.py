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
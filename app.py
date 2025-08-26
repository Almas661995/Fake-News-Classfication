import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # Replaced CountVectorizer with TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Custom CSS for the app
st.markdown("""
    <style>
        .header {
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            border-radius: 10px;
        }
        .footer {
            background: linear-gradient(to right, #4b6cb7, #182848);
            color: white;
            text-align: center;
            padding: 10px;
            position: fixed;
            width: 100%;
            bottom: 0;
            left: 0;
            font-size: 14px;
            border-radius: 10px;
        }
        .section-header {
            font-size: 24px;
            font-weight: bold;
            color: #ff5722;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="header">Fake News Detection System</div>', unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader("Upload Your Dataset (CSV)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", data.head())
    
    if 'tweet' in data.columns and 'label' in data.columns:
        st.success("Dataset is valid! Ready for preprocessing.")
        
        if st.button("Preprocess Data"):
            X = data['tweet']
            y = data['label']
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')  # TF-IDF applied here
            X_tfidf = vectorizer.fit_transform(X)
            st.session_state['X_tfidf'] = X_tfidf
            st.session_state['y'] = y
            st.session_state['vectorizer'] = vectorizer  # Store vectorizer for later use

        if 'X_tfidf' in st.session_state and 'y' in st.session_state:
            X_tfidf = st.session_state['X_tfidf']
            y = st.session_state['y']

            model_choice = st.selectbox("Select a Classification Algorithm", ["Naive Bayes", "Logistic Regression", "Random Forest", "SVM"])
            train_size = st.number_input("Enter Train Data Percentage", min_value=1, max_value=99, value=80)
            train_ratio = train_size / 100

            if st.button("Train and Evaluate Model"):
                X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=1-train_ratio, random_state=42)
                
                if model_choice == "Naive Bayes":
                    model = MultinomialNB()
                elif model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=200)
                elif model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_choice == "SVM":
                    model = SVC(probability=True, random_state=42)
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.session_state['model'] = model
                
                st.write("### Classification Report")
                report = classification_report(y_test, preds, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
                accuracy = accuracy_score(y_test, preds)
                st.write(f"### Accuracy: {accuracy * 100:.2f}%")
                
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)
                
                # ROC Curve
                st.write("### ROC Curve")
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    plt.figure()
                    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("Receiver Operating Characteristic (ROC) Curve")
                    plt.legend()
                    st.pyplot(plt)
                else:
                    st.warning("Selected model does not support probability estimates for ROC curve.")

        # **Real-Time Tweet Classification**
        st.write("### Real-Time Tweet Classification")
        user_input = st.text_area("Enter a Tweet:")
        
        if st.button("Classify Tweet"):
            if 'model' in st.session_state and 'vectorizer' in st.session_state:
                vectorizer = st.session_state['vectorizer']
                input_tfidf = vectorizer.transform([user_input])  # Transform using stored TF-IDF vectorizer
                model = st.session_state['model']
                prediction = model.predict(input_tfidf)[0]
                
                if prediction == 0:
                    st.success("Prediction: Real News ✅")
                else:
                    st.error("Prediction: Fake News ❌")
            else:
                st.warning("Please preprocess and train the model first.")

# **Footer**
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""
    <div class="footer">
        <p>Session: 2020-24 | Class: BSCS-VIII | Developed by: Usama Mughal & Usman Umer</p>
        <p>Last Updated: {current_time}</p>
    </div>
    """, unsafe_allow_html=True)

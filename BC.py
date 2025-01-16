import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pickle

# Function to preprocess data
def preprocess_data(df, target_col):
    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col:  # Skip the target column for now
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Ensure the target column is properly encoded if it's categorical
    if df[target_col].dtype == 'object':
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

    return df

# Function to train model
def train_model(X, y, model_type, n_estimators):
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    else:
        raise ValueError("Invalid model type")

    model.fit(X, y)
    return model

# Function to validate dataset
def validate_columns(train_columns, pred_columns):
    return set(train_columns) == set(pred_columns)

# Streamlit app
def main():
    st.title("Breast Cancer Prediction Platform")

    tab1, tab2 = st.tabs(["Train Model", "Predict"])

    # Tab 1: Train Model
    with tab1:
        st.header("Train a Classification Model")
        
        # File uploader
        data_file = st.file_uploader("Upload Gene Expression Dataset (CSV)", type=["csv"], key="train_file")

        if data_file is not None:
            df = pd.read_csv(data_file)
            st.write("Uploaded Dataset:", df.head())

            target_col = st.selectbox("Select Target Column", options=df.columns, key="target_column")

            if target_col:
                # Preprocess the dataset
                df = preprocess_data(df, target_col)
                
                X = df.drop(columns=[target_col])
                y = df[target_col]

                model_options = ["Random Forest", "Gradient Boosting"]
                model_type = st.selectbox("Choose Model Type", model_options, key="model_type")
                n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, step=10, value=100, key="n_estimators")

                if st.button("Train Model", key="train_button"):
                    model = train_model(X, y, model_type, n_estimators)
                    st.success(f"{model_type} model trained successfully!")

                    # Display model performance
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.subheader("Model Performance")
                    st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                    st.text(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))

                    # Save the model
                    model_filename = f"{model_type.lower().replace(' ', '_')}_model.pkl"
                    with open(model_filename, "wb") as f:
                        pickle.dump(model, f)
                    st.download_button(
                        label="Download Model",
                        data=pickle.dumps(model),
                        file_name=model_filename,
                        mime="application/octet-stream"
                    )

    # Tab 2: Predict
    with tab2:
        st.header("Predict Using a Trained Model")

        model_file = st.file_uploader("Upload Trained Model (PKL)", type=["pkl"], key="predict_model")
        
        if model_file is not None:
            model = pickle.load(model_file)
            st.success("Model loaded successfully.")

            prediction_file = st.file_uploader("Upload Dataset for Prediction (CSV)", type=["csv"], key="predict_data")
            
            if prediction_file is not None:
                pred_df = pd.read_csv(prediction_file)
                st.write("Dataset for Prediction:", pred_df.head())

                # Validate columns
                if not validate_columns(model.feature_names_in_, pred_df.columns):
                    st.error("Column mismatch! Ensure the prediction dataset has the same features as the training dataset.")
                else:
                    if st.button("Predict", key="predict_button"):
                        predictions = model.predict(pred_df)
                        pred_df["Predictions"] = predictions

                        st.write("Predictions:", pred_df)

                        # Allow download of predictions
                        csv = pred_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

if __name__ == "__main__":
    main()

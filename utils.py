import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_categorical(value):
    """Clean categorical values."""
    if pd.isna(value):
        return 'unknown'
    return str(value).lower().replace('-', '_').replace(' ', '_')

def preprocess_transaction(df, model, model_name):
    """
    Preprocess transaction data to match model's expected features exactly.
    """
    try:
        df = df.copy()

        # Convert date and create time-based features
        if 'trans_date_trans_time' in df.columns:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
            df['hour'] = df['trans_date_trans_time'].dt.hour
            df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
            df['is_night'] = df['hour'].isin([21,22,23,0,1,2,3,4,5]).astype(int)
            df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
            df = df.drop('trans_date_trans_time', axis=1)

        # Calculate distance
        if all(col in df.columns for col in ['lat', 'long', 'merch_lat', 'merch_long']):
            df['distance_km'] = np.sqrt(
                (df['lat'] - df['merch_lat'])**2 + 
                (df['long'] - df['merch_long'])**2
            ) * 111

        # Calculate amount per capita
        if 'city_pop' in df.columns and 'amt' in df.columns:
            df['city_pop'] = df['city_pop'].replace(0, df['city_pop'].mean())
            df['amount_per_capita'] = df['amt'] / df['city_pop']

        # Encode categorical variables
        categorical_cols = {
            'category': 'category_encoded',
            'merchant': 'merchant_encoded',
            'gender': 'gender_encoded',
            'city': 'city_encoded',
            'state': 'state_encoded',
            'job': 'job_encoded'
        }

        for col, encoded_col in categorical_cols.items():
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
                if col == 'gender':
                    df[encoded_col] = df[col].map({'m': 1, 'f': 0, 'unknown': -1})
                else:
                    le = LabelEncoder()
                    df[encoded_col] = le.fit_transform(df[col].astype(str))
            else:
                df[encoded_col] = 0

        # These are the exact features the model was trained on
        required_features = [
            'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
            'hour', 'is_night', 'day_of_week', 'is_weekend', 'distance_km',
            'amount_per_capita', 'category_encoded', 'merchant_encoded',
            'gender_encoded', 'city_encoded', 'state_encoded', 'job_encoded'
        ]

        # Ensure all required features are present
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0

        # Return only the required features in the correct order
        return df[required_features].astype(float)

    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        print("Available columns:", df.columns.tolist())
        print("Required features:", required_features)
        raise
def get_groq_explanation(probability, transaction_data, model_name, client):
    """Generate explanation using Groq AI."""
    prompt = f"""
    As an expert fraud detection analyst at a major financial institution, analyze this transaction and provide a comprehensive risk assessment.

    Transaction Details:
    {transaction_data.to_dict(orient='records')[0]}

    Model Analysis:
    - {model_name} model predicts a {probability:.1%} probability of fraud

    Please provide a detailed analysis following this structure:

    1. Risk Level Assessment:
    - Classify as Low/Medium/High risk
    - Provide a clear justification for the risk level

    2. Key Risk Factors:
    - Highlight the most significant indicators
    - Quantify risks where possible
    - Compare against typical patterns

    3. Suspicious Patterns Analysis:
    - Transaction amount patterns
    - Geographical considerations
    - Timing and frequency analysis
    - Merchant relationship evaluation
    - Customer profile assessment

    4. Recommended Actions:
    - Immediate steps for fraud team
    - Verification procedures
    - Monitoring suggestions
    - Customer contact recommendations

    Please focus on actionable insights and specific details that would help the fraud team make an informed decision.
    Format the response in clear sections with bullet points for better readability.
    """

    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000  # Increased for more detailed response
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating explanation: {str(e)}"
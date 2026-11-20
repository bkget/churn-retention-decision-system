"""Shared fixtures: tiny hand-built frames in both raw source formats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_settings


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture
def kaggle_raw() -> pd.DataFrame:
    """Five customers in the original 21-column format (TotalCharges as text)."""
    return pd.DataFrame(
        {
            "customerID": ["A1", "A2", "A3", "A4", "A5"],
            "gender": ["Female", "Male", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1],
            "Partner": ["Yes", "No", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No", "No"],
            "tenure": [0, 5, 40, 72, 3],
            "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes"],
            "MultipleLines": ["No", "Yes", "No phone service", "Yes", "No"],
            "InternetService": ["Fiber optic", "Fiber optic", "DSL", "No", "Fiber optic"],
            "OnlineSecurity": ["No", "No", "Yes", "No internet service", "No"],
            "OnlineBackup": ["No", "Yes", "Yes", "No internet service", "No"],
            "DeviceProtection": ["No", "No", "Yes", "No internet service", "No"],
            "TechSupport": ["No", "No", "Yes", "No internet service", "No"],
            "StreamingTV": ["Yes", "Yes", "No", "No internet service", "No"],
            "StreamingMovies": ["No", "Yes", "No", "No internet service", "No"],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
            ],
            "PaperlessBilling": ["Yes", "Yes", "No", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [80.0, 95.5, 60.0, 25.0, 70.0],
            "TotalCharges": [" ", "470.25", "2400.0", "1800", "210.0"],
            "Churn": ["No", "Yes", "No", "No", "Yes"],
        }
    )


@pytest.fixture
def ibm_raw() -> pd.DataFrame:
    """Three customers in the IBM v11.1.3+ merged format (subset of columns)."""
    return pd.DataFrame(
        {
            "Customer ID": ["B1", "B2", "B3"],
            "Gender": ["Female", "Male", "Male"],
            "Age": [30, 70, 45],
            "Senior Citizen": [0, 1, 0],
            "Under 30": [0, 0, 0],
            "Partner": [1, 0, 1],
            "Married": [1, 0, 1],
            "Dependents": [0, 0, 1],
            "Number of Dependents": [0, 0, 2],
            "Tenure in Months": [2, 30, 60],
            "Phone Service": [1, 1, 0],
            "Multiple Lines": [0, 1, 0],
            "Internet Service": [1, 1, 0],
            "Internet Type": ["Fiber Optic", "DSL", np.nan],
            "Online Security": [0, 1, 0],
            "Online Backup": [0, 1, 0],
            "Device Protection Plan": [0, 1, 0],
            "Premium Tech Support": [0, 1, 0],
            "Streaming TV": [1, 0, 0],
            "Streaming Movies": [1, 0, 0],
            "Streaming Music": [1, 0, 0],
            "Unlimited Data": [1, 1, 0],
            "Contract": ["Month-to-Month", "One Year", "Two Year"],
            "Paperless Billing": [1, 0, 0],
            "Payment Method": ["Bank Withdrawal", "Credit Card", "Mailed Check"],
            "Offer": ["Offer E", np.nan, "Offer A"],
            "Monthly Charge": [90.0, 55.0, 20.0],
            "Total Charges": [180.0, 1650.0, 1200.0],
            "Number of Referrals": [0, 2, 5],
            "Referred a Friend": [0, 1, 1],
            "Satisfaction Score": [1, 4, 5],
            "Churn Score": [90, 30, 10],
            "CLTV": [3000, 4500, 6000],
            "Churn Reason": ["Competitor made better offer", np.nan, np.nan],
            "Churn Category": ["Competitor", np.nan, np.nan],
            "Customer Status": ["Churned", "Stayed", "Stayed"],
            "City": ["X", "Y", "Z"],
            "Churn": [1, 0, 0],
        }
    )

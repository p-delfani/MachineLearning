"""
Explainable AI (XAI) Module for Intrusion Detection Systems

Provides feature importance explanations using SHAP.
"""

import shap
import numpy as np

def explain_model(model, X_sample):
    """
    Generate SHAP explanations for a trained model.
    """
    explainer = shap.Explainer(model.predict, X_sample)
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values, X_sample)

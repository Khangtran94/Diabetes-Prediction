import json
import os
import tempfile
import logging
import boto3

from catboost import CatBoostClassifier

# Setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Path within the Lambda deployment package
MODEL_FILE = "catboost_model.pkl"

# Load model once at cold start
_model = None

def load_model():
    """
    Load and cache the CatBoost model.
    """
    global _model
    if _model is None:
        logger.info("Loading CatBoost model from %s", MODEL_FILE)
        try:
            model = CatBoostClassifier()
            model.load_model(MODEL_FILE)
            _model = model
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise
    return _model

def format_input(event_body):
    """
    Extract feature values from the event body.
    Expects JSON with a key "features": list of numeric values.
    """
    if "features" not in event_body:
        raise ValueError("No 'features' key provided in request body")

    features = event_body["features"]
    if not isinstance(features, list):
        raise ValueError("'features' must be a list of numeric values")
    return features

def predict(features):
    """
    Predict using CatBoost model.
    Returns both class and probability.
    """
    model = load_model()
    # Convert to list-of-lists for single sample
    data = [features]

    preds = model.predict(data)
    probs = model.predict_proba(data)

    # For a binary classification, CatBoost returns prob of class 1
    return {
        "predicted_class": int(preds[0]),
        "probability_of_positive": float(probs[0][1])
    }

def build_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(body)
    }

def handler(event, context):
    """
    Main entrypoint for AWS Lambda.
    """
    logger.info("Received event: %s", event)

    try:
        # Parse JSON body
        if "body" in event:
            event_body = json.loads(event["body"])
        else:
            event_body = event

        features = format_input(event_body)

        result = predict(features)
        return build_response(200, {"status": "success", "prediction": result})

    except ValueError as ve:
        logger.error("Input error: %s", ve)
        return build_response(400, {"status": "error", "message": str(ve)})

    except Exception as e:
        logger.error("Prediction error: %s", e)
        return build_response(500, {"status": "error", "message": "Internal server error"})

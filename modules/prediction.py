
import pandas as pd


def predict_sla_breach(model, input_data):
    """
    Predict possible SLA breach using the saved model pipeline.

    input_data must contain the same feature columns used
    when the model was trained.
    """
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif not isinstance(input_data, pd.DataFrame):
        raise TypeError(
            "input_data must be a dictionary or pandas DataFrame."
        )

    if len(input_data) != 1:
        raise ValueError(
            "Provide exactly one service request for prediction."
        )

    prediction = model.predict(input_data)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        if 1 in classes:
            probability = float(probabilities[classes.index(1)])
        elif True in classes:
            probability = float(probabilities[classes.index(True)])

    return {
        "prediction": prediction,
        "breach_probability": probability,
    }

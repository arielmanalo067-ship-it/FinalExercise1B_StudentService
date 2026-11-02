
import pandas as pd

FEATURE_COLUMNS = [
    "Service_Type",
    "Requester_Type",
    "Requested_Channel",
    "Priority",
    "Assigned_Office",
    "Queue_Length",
    "Required_Documents_Complete",
    "SLA_Days",
    "Submission_Weekday",
    "Queue_Category",
]

REQUIRED_COLUMNS = FEATURE_COLUMNS + ["SLA_Breached"]


def validate_dataset(df):
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        return False, missing

    if df.empty:
        return False, ["Dataset is empty"]

    return True, []


def load_dataset(file):
    df = pd.read_csv(file)
    valid, issues = validate_dataset(df)

    if not valid:
        raise ValueError(
            "Dataset validation failed: " + ", ".join(issues)
        )

    return df

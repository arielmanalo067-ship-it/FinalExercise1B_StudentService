
import pandas as pd


def get_dashboard_metrics(df):
    """Calculate summary metrics for the service dashboard."""
    total_requests = len(df)

    if "SLA_Breached" in df.columns:
        breached = df["SLA_Breached"].astype(str).str.lower().isin(
            ["1", "true", "yes", "breached"]
        ).sum()
    else:
        breached = 0

    if "Assigned_Office" in df.columns:
        offices = df["Assigned_Office"].nunique()
    else:
        offices = 0

    return {
        "total_requests": int(total_requests),
        "sla_breaches": int(breached),
        "offices": int(offices),
    }


def get_office_workload(df):
    """Count requests assigned to each office."""
    if "Assigned_Office" not in df.columns:
        return pd.DataFrame(
            columns=["Assigned_Office", "Request_Count"]
        )

    workload = (
        df.groupby("Assigned_Office")
        .size()
        .reset_index(name="Request_Count")
        .sort_values("Request_Count", ascending=False)
    )

    return workload

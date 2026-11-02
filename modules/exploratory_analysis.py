
import pandas as pd
import matplotlib.pyplot as plt


def get_summary(df):
    """Return basic descriptive statistics for the dataset."""
    return df.describe(include="all").fillna("N/A")


def requests_by_office(df):
    """Count service requests handled by each office."""
    if "Assigned_Office" not in df.columns:
        return pd.Series(dtype="int64")

    return df["Assigned_Office"].value_counts()


def sla_breach_counts(df):
    """Count records by SLA breach status."""
    if "SLA_Breached" not in df.columns:
        return pd.Series(dtype="int64")

    return df["SLA_Breached"].value_counts(dropna=False)


def plot_requests_by_office(df):
    """Create a chart of requests by office."""
    counts = requests_by_office(df)

    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)

    ax.set_title("Student Service Requests by Office")
    ax.set_xlabel("Assigned Office")
    ax.set_ylabel("Number of Requests")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig

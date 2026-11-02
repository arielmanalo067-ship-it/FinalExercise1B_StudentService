
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Student Service Monitoring",
    page_icon="📊",
    layout="wide"
)

st.title("Student Service Request Monitoring")
st.caption("SLA breach prediction and decision-support dashboard")

MODEL_PATH = os.path.join("models", "trained_model.joblib")
DATA_PATH = "student_service_requests_cleaned.csv"

FEATURES = [
    "Service_Type",
    "Requester_Type",
    "Requested_Channel",
    "Priority",
    "Assigned_Office",
    "Queue_Length",
    "Required_Documents_Complete",
    "SLA_Days",
    "Submission_Weekday",
    "Queue_Category"
]

REQUIRED_COLUMNS = [
    "Request_ID",
    "Service_Type",
    "Requester_Type",
    "Submission_Date",
    "Requested_Channel",
    "Priority",
    "Assigned_Office",
    "Queue_Length",
    "Required_Documents_Complete",
    "Processing_Days",
    "SLA_Days",
    "Status",
    "Reopened",
    "Satisfaction_Score",
    "Submission_Weekday",
    "Queue_Category",
    "SLA_Breached"
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


def prepare_data(data):
    data = data.copy()

    if "Submission_Date" in data.columns:
        data["Submission_Date"] = pd.to_datetime(
            data["Submission_Date"], errors="coerce"
        )

        if "Submission_Weekday" not in data.columns:
            data["Submission_Weekday"] = (
                data["Submission_Date"].dt.day_name()
            )

    if "Queue_Length" in data.columns:
        data["Queue_Length"] = pd.to_numeric(
            data["Queue_Length"], errors="coerce"
        )

        if "Queue_Category" not in data.columns:
            data["Queue_Category"] = pd.cut(
                data["Queue_Length"],
                bins=[-float("inf"), 5, 15, float("inf")],
                labels=["Low", "Medium", "High"]
            ).astype("object")

    return data


page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "Data Analysis",
        "Prediction",
        "Dashboard",
        "System Design"
    ]
)

st.sidebar.info(
    "Predictions support staff review. "
    "They do not automatically approve or reject requests."
)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found: models/trained_model.joblib")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

uploaded_file = st.sidebar.file_uploader(
    "Upload cleaned CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
    except Exception as error:
        st.error(f"Unable to read CSV: {error}")
        st.stop()
elif os.path.exists(DATA_PATH):
    try:
        df = load_csv(DATA_PATH)
    except Exception as error:
        st.error(f"Unable to read the default CSV: {error}")
        st.stop()
else:
    df = None

if page == "Home":
    st.subheader("System Overview")
    st.write(
        "This application reviews student service request data, "
        "explores workload patterns, and predicts possible SLA breaches."
    )

    st.markdown("""
    **Main functions**
    - Dataset upload and validation
    - Interactive exploratory data analysis
    - Prediction using the saved Decision Tree pipeline
    - Workload and SLA monitoring
    - System requirements and decision workflow
    """)

    st.warning(
        "Use authorized or synthetic records. "
        "A prediction is not a final administrative decision."
    )

elif page == "Dataset":
    st.subheader("Dataset Upload and Validation")

    if df is None:
        st.info("Upload the cleaned CSV using the sidebar.")
    else:
        missing = [
            column for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            st.error("Required columns are missing:")
            st.write(missing)
            st.write("Columns found in the uploaded file:")
            st.write(df.columns.tolist())
        elif df.empty:
            st.warning("The uploaded dataset has no records.")
        else:
            df = prepare_data(df)
            st.success("Required columns are present.")
            st.metric("Total records", len(df))
            st.metric("Total columns", len(df.columns))
            st.write("Preview")
            st.dataframe(df.head(20), use_container_width=True)

            st.subheader("Missing Values")
            missing_values = (
                df.isna().sum()
                .rename("Missing Count")
                .reset_index()
                .rename(columns={"index": "Column"})
            )
            st.dataframe(missing_values, use_container_width=True)

elif page == "Data Analysis":
    st.subheader("Exploratory Data Analysis")

    if df is None:
        st.info("Upload the cleaned CSV first.")
    else:
        df = prepare_data(df)
        missing = [
            column for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            st.error(f"Dataset schema mismatch. Missing: {missing}")
        elif df.empty:
            st.warning("No data available for analysis.")
        else:
            service_counts = (
                df["Service_Type"].fillna("Unknown")
                .value_counts()
                .rename_axis("Service Type")
                .reset_index(name="Requests")
            )
            fig1 = px.bar(
                service_counts,
                x="Service Type",
                y="Requests",
                title="Requests by Service Type"
            )
            st.plotly_chart(fig1, use_container_width=True)

            queue_counts = (
                df["Queue_Category"].fillna("Unknown")
                .value_counts()
                .rename_axis("Queue Category")
                .reset_index(name="Requests")
            )
            fig2 = px.bar(
                queue_counts,
                x="Queue Category",
                y="Requests",
                title="Requests by Queue Category"
            )
            st.plotly_chart(fig2, use_container_width=True)

            breach_counts = (
                df["SLA_Breached"].value_counts(dropna=False)
                .rename_axis("SLA Breached")
                .reset_index(name="Records")
            )
            fig3 = px.pie(
                breach_counts,
                names="SLA Breached",
                values="Records",
                title="Observed SLA Breach Labels"
            )
            st.plotly_chart(fig3, use_container_width=True)

elif page == "Prediction":
    st.subheader("SLA Breach Risk Prediction")
    st.write(
        "Enter information available when the request is submitted. "
        "Do not enter the final processing outcome."
    )

    if df is None:
        st.info("Upload the cleaned CSV first to populate form choices.")
    else:
        df = prepare_data(df)
        missing_features = [
            column for column in FEATURES
            if column not in df.columns
        ]

        if missing_features:
            st.error(
                "Dataset does not match the model's expected features: "
                f"{missing_features}"
            )
        else:
            with st.form("prediction_form"):
                service = st.selectbox(
                    "Service Type",
                    sorted(df["Service_Type"].dropna().astype(str).unique())
                )
                requester = st.selectbox(
                    "Requester Type",
                    sorted(df["Requester_Type"].dropna().astype(str).unique())
                )
                channel = st.selectbox(
                    "Requested Channel",
                    sorted(df["Requested_Channel"].dropna().astype(str).unique())
                )
                priority = st.selectbox(
                    "Priority",
                    sorted(df["Priority"].dropna().astype(str).unique())
                )
                office = st.selectbox(
                    "Assigned Office",
                    sorted(df["Assigned_Office"].dropna().astype(str).unique())
                )
                queue_length = st.number_input(
                    "Queue Length", min_value=0, value=5
                )
                documents = st.selectbox(
                    "Required Documents Complete",
                    sorted(
                        df["Required_Documents_Complete"]
                        .dropna().astype(str).unique()
                    )
                )
                sla_days = st.number_input(
                    "SLA Days", min_value=1, value=5
                )
                weekday = st.selectbox(
                    "Submission Weekday",
                    [
                        "Monday", "Tuesday", "Wednesday",
                        "Thursday", "Friday", "Saturday", "Sunday"
                    ]
                )

                if queue_length <= 5:
                    queue_category = "Low"
                elif queue_length <= 15:
                    queue_category = "Medium"
                else:
                    queue_category = "High"

                submitted = st.form_submit_button("Predict Risk")

            if submitted:
                row = pd.DataFrame([{
                    "Service_Type": service,
                    "Requester_Type": requester,
                    "Requested_Channel": channel,
                    "Priority": priority,
                    "Assigned_Office": office,
                    "Queue_Length": queue_length,
                    "Required_Documents_Complete": documents,
                    "SLA_Days": sla_days,
                    "Submission_Weekday": weekday,
                    "Queue_Category": queue_category
                }], columns=FEATURES)

                try:
                    prediction = model.predict(row)[0]
                    st.subheader("Prediction Result")

                    if int(prediction) == 1:
                        st.error("Potential SLA breach — staff review recommended.")
                    else:
                        st.success("No SLA breach predicted by the model.")

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(row)[0]
                        classes = list(model.classes_)
                        if 1 in classes:
                            breach_probability = probabilities[classes.index(1)]
                            st.metric(
                                "Estimated breach probability",
                                f"{breach_probability:.1%}"
                            )

                    st.caption(
                        "This result is an estimate from the trained model, "
                        "not a guarantee or a final staff decision."
                    )
                except Exception as error:
                    st.error(f"Prediction failed: {error}")

elif page == "Dashboard":
    st.subheader("Decision-Support Dashboard")

    if df is None:
        st.info("Upload the cleaned CSV first.")
    else:
        df = prepare_data(df)
        missing = [
            column for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            st.error(f"Dataset schema mismatch. Missing: {missing}")
        elif df.empty:
            st.warning("No records available.")
        else:
            total = len(df)
            completed = (
                df["Status"].astype(str).str.strip().str.lower()
                == "completed"
            ).sum()
            observed = pd.to_numeric(
                df["SLA_Breached"], errors="coerce"
            ).dropna()

            breach_rate = (
                observed.mean() * 100 if len(observed) else 0
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Requests", total)
            col2.metric("Completed Requests", int(completed))
            col3.metric("Observed SLA Breach Rate", f"{breach_rate:.1f}%")

            st.subheader("Workload by Office")
            office_counts = (
                df.groupby("Assigned_Office", dropna=False)
                .size()
                .reset_index(name="Requests")
            )
            fig = px.bar(
                office_counts,
                x="Assigned_Office",
                y="Requests",
                title="Request Workload by Assigned Office"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Observed Breach Labels")
            st.dataframe(
                df["SLA_Breached"].value_counts(dropna=False)
                .rename_axis("SLA_Breached")
                .reset_index(name="Records"),
                use_container_width=True
            )

elif page == "System Design":
    st.subheader("System Design")

    st.markdown("""
    ### Functional Requirements
    1. The system accepts a cleaned CSV dataset.
    2. The system validates required columns and reports errors.
    3. The system displays at least three interactive charts.
    4. The system predicts possible SLA breaches using the saved pipeline.
    5. The system displays workload information to support staff review.

    ### Non-Functional Requirements
    1. The application should be usable and easy to navigate.
    2. The application should validate inputs and handle errors clearly.
    3. Student records should be protected and accessible only to authorized staff.

    ### Decision Workflow
    Request submitted → Data validated → Risk estimated → Staff reviews context
    → Authorized staff records the final decision.

    ### Use Case Actors
    - Student Services Staff: upload data, view charts, review predictions.
    - Administrator: manage deployment and model files.
    - Student: submits a request through the institution's normal service process.

    ### Important Limitation
    Predictions depend on the quality and representativeness of the training data.
    The model must be evaluated before real-world use. A prediction must not
    automatically approve, reject, or deprioritize a student's request.
    """)


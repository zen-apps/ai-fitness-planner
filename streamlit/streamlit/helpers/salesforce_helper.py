import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from langchain_salesforce import SalesforceTool
import os
import streamlit as st

# Load environment variables
load_dotenv("./config/dev.env")


# Tool initialization functions
@st.cache_resource
def get_salesforce_tool():
    """Initialize Salesforce tool with caching."""
    return SalesforceTool(
        username=os.getenv("SALESFORCE_USERNAME", "your-username"),
        password=os.getenv("SALESFORCE_PASSWORD", "your-password"),
        security_token=os.getenv("SALESFORCE_SECURITY_TOKEN", "your-token"),
        domain=os.getenv("SALESFORCE_DOMAIN", "login"),
    )


# Helper functions
def salesforce_to_dataframe(query_result):
    """Convert Salesforce query result to pandas DataFrame."""
    if not query_result.get("records"):
        return pd.DataFrame()

    records = []
    for record in query_result["records"]:
        flat_record = {}
        for key, value in record.items():
            if key == "attributes":
                continue
            elif isinstance(value, dict) and "attributes" in value:
                nested_name = key
                for nested_key, nested_value in value.items():
                    if nested_key != "attributes":
                        flat_record[f"{nested_name}_{nested_key}"] = nested_value
            else:
                flat_record[key] = value
        records.append(flat_record)

    df = pd.DataFrame(records)

    # Convert date columns
    date_columns = [col for col in df.columns if "Date" in col or "DateTime" in col]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass

    return df


def create_pipeline_analysis_chart(opportunities_df):
    """Create pipeline analysis visualization."""
    if opportunities_df.empty:
        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Pipeline by Stage",
            "Opportunities by Amount",
            "Win Rate by Stage",
            "Pipeline Trend",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
    )

    # Pipeline by stage
    stage_counts = opportunities_df["StageName"].value_counts()
    fig.add_trace(
        go.Bar(x=stage_counts.index, y=stage_counts.values, name="Count by Stage"),
        row=1,
        col=1,
    )

    # Opportunities by amount (scatter)
    fig.add_trace(
        go.Scatter(
            x=opportunities_df["CloseDate"],
            y=opportunities_df["Amount"],
            mode="markers",
            marker=dict(
                size=opportunities_df["Probability"],
                color=opportunities_df["Amount"],
                colorscale="Viridis",
                showscale=True,
            ),
            text=opportunities_df["Name"],
            name="Amount vs Close Date",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=600, showlegend=False)
    return fig


def check_api_connections():
    """Check the status of API connections."""
    openai_key = os.getenv("OPENAI_API_KEY")
    sf_username = os.getenv("SALESFORCE_USERNAME")

    return {
        "openai_connected": bool(openai_key),
        "salesforce_connected": bool(sf_username),
    }


def get_sample_queries():
    """Return dictionary of sample SOQL queries."""
    return {
        "Recent Contacts": "SELECT Id, Name, Email, Account.Name FROM Contact WHERE CreatedDate >= LAST_N_DAYS:30",
        "High Value Opportunities": "SELECT Id, Name, Amount, StageName, Account.Name FROM Opportunity WHERE Amount > 50000 AND IsClosed = false",
        "Active Tasks": "SELECT Id, Subject, ActivityDate, Who.Name, What.Name FROM Task WHERE IsClosed = false",
        "Accounts by Industry": "SELECT Id, Name, Industry, AnnualRevenue FROM Account WHERE Industry != null ORDER BY AnnualRevenue DESC",
    }


def get_soql_reference():
    """Return SOQL quick reference documentation."""
    return """
    **Basic Structure:**
    ```sql
    SELECT fields 
    FROM object 
    WHERE conditions 
    ORDER BY field 
    LIMIT number
    ```
    
    **Common Objects:**
    - `Account` - Companies
    - `Contact` - People
    - `Opportunity` - Sales opportunities
    - `Task` - Activities
    - `Lead` - Potential customers
    
    **Date Functions:**
    - `TODAY`
    - `YESTERDAY`
    - `LAST_N_DAYS:7`
    - `THIS_MONTH`
    - `LAST_QUARTER`
    
    **Relationships:**
    - `Account.Name`
    - `Owner.Email`
    - `Who.Name`
    """

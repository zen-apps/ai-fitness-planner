import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
import base64
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv("./config/dev.env")

# Configure page
st.set_page_config(
    page_title="PDF Data Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Pydantic models for structured output
class ExtractedTable(BaseModel):
    """Represents a table extracted from PDF."""

    table_name: str = Field(description="Descriptive name for the table")
    headers: List[str] = Field(description="Column headers")
    rows: List[List[str]] = Field(description="Table rows as list of lists")
    confidence_score: int = Field(
        description="Confidence in extraction quality (1-10)", ge=1, le=10
    )
    notes: Optional[str] = Field(description="Any notes about the extraction")


class ExtractedKeyValue(BaseModel):
    """Represents key-value pairs extracted from PDF."""

    key: str = Field(description="The key or field name")
    value: str = Field(description="The extracted value")
    confidence: int = Field(description="Confidence in extraction (1-10)", ge=1, le=10)
    page_number: Optional[int] = Field(description="Page number where found")


class ExtractionResult(BaseModel):
    """Complete extraction result from PDF."""

    extraction_type: str = Field(description="Type of extraction performed")
    tables: List[ExtractedTable] = Field(description="Extracted tables")
    key_values: List[ExtractedKeyValue] = Field(description="Extracted key-value pairs")
    raw_text: str = Field(description="Raw extracted text")
    summary: str = Field(description="Summary of what was extracted")
    recommendations: List[str] = Field(description="Recommendations for data cleanup")


# Cached LLM initialization
@st.cache_resource
def get_extraction_llm():
    """Initialize OpenAI LLM for data extraction."""
    return ChatOpenAI(
        model="gpt-4o-mini", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY")
    )


def file_to_base64(file):
    """Convert uploaded file to base64 string."""
    encoded_string = base64.b64encode(file.getvalue()).decode("utf-8")
    return encoded_string


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyPDF2."""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        # Extract text content using PyPDF2
        pdf_reader = PdfReader(tmp_file_path)
        text_content = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"

        # Clean up temp file
        os.unlink(tmp_file_path)

        return text_content, len(pdf_reader.pages)

    except Exception as e:
        st.error(f"PDF text extraction failed: {e}")
        return "", 0


def extract_text_from_demo_pdf(file_path):
    """Extract text from demo PDF file using PyPDF2."""
    try:
        # Extract text content using PyPDF2
        pdf_reader = PdfReader(file_path)
        text_content = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"

        return text_content, len(pdf_reader.pages)

    except Exception as e:
        st.error(f"Demo PDF text extraction failed: {e}")
        return "", 0


class DemoFile:
    """Mock file object for demo PDF processing."""

    def __init__(self, file_path):
        self.name = os.path.basename(file_path)
        self.file_path = file_path

    def getvalue(self):
        with open(self.file_path, "rb") as f:
            return f.read()


def extract_tables_from_text(text: str, llm) -> ExtractionResult:
    """Extract tables from text using AI."""

    prompt = f"""
    You are an expert data extraction specialist. Analyze the following text and extract any tabular data.
    
    Instructions:
    1. Identify all tables or structured data that could be converted to CSV format
    2. For each table, determine appropriate column headers
    3. Extract all rows of data
    4. Rate your confidence in the extraction quality (1-10)
    5. Provide recommendations for data cleanup
    
    Text to analyze:
    {text[:8000]}  # Limit text length for API
    
    Focus on finding:
    - Financial data (revenues, expenses, etc.)
    - Inventory lists
    - Contact information
    - Invoices or billing data
    - Any structured tabular information
    
    Return your analysis in the specified JSON format.
    """

    try:
        message = HumanMessage(content=prompt)
        response = llm.with_structured_output(ExtractionResult).invoke([message])
        return response
    except Exception as e:
        st.error(f"AI extraction failed: {e}")
        return ExtractionResult(
            extraction_type="error",
            tables=[],
            key_values=[],
            raw_text=text[:1000],
            summary=f"Extraction failed: {e}",
            recommendations=["Try a different extraction method"],
        )


def extract_key_values_from_text(text: str, llm) -> ExtractionResult:
    """Extract key-value pairs from text using AI."""

    prompt = f"""
    You are an expert data extraction specialist. Analyze the following text and extract key-value pairs.
    
    Instructions:
    1. Find all important key-value relationships (e.g., "Invoice Number: 12345")
    2. Look for form fields, labels, and their corresponding values
    3. Extract contact information, dates, amounts, IDs, etc.
    4. Rate your confidence in each extraction (1-10)
    5. Note the page number if identifiable
    
    Text to analyze:
    {text[:8000]}  # Limit text length for API
    
    Focus on finding:
    - Names, addresses, phone numbers
    - Invoice/order numbers
    - Dates and amounts
    - Product codes or SKUs
    - Any labeled information
    
    Return your analysis in the specified JSON format.
    """

    try:
        message = HumanMessage(content=prompt)
        response = llm.with_structured_output(ExtractionResult).invoke([message])
        return response
    except Exception as e:
        st.error(f"AI extraction failed: {e}")
        return ExtractionResult(
            extraction_type="error",
            tables=[],
            key_values=[],
            raw_text=text[:1000],
            summary=f"Extraction failed: {e}",
            recommendations=["Try a different extraction method"],
        )


def table_to_dataframe(table: ExtractedTable) -> pd.DataFrame:
    """Convert extracted table to pandas DataFrame."""
    try:
        if not table.rows:
            return pd.DataFrame()

        # Ensure all rows have the same number of columns as headers
        max_cols = len(table.headers)
        cleaned_rows = []

        for row in table.rows:
            # Pad or truncate row to match header length
            if len(row) < max_cols:
                row.extend([""] * (max_cols - len(row)))
            elif len(row) > max_cols:
                row = row[:max_cols]
            cleaned_rows.append(row)

        df = pd.DataFrame(cleaned_rows, columns=table.headers)
        return df
    except Exception as e:
        st.error(f"Error converting table to DataFrame: {e}")
        return pd.DataFrame()


def key_values_to_dataframe(key_values: List[ExtractedKeyValue]) -> pd.DataFrame:
    """Convert key-value pairs to pandas DataFrame."""
    try:
        data = []
        for kv in key_values:
            data.append(
                {
                    "Key": kv.key,
                    "Value": kv.value,
                    "Confidence": f"{kv.confidence}/10",
                    "Page": kv.page_number or "Unknown",
                }
            )
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error converting key-values to DataFrame: {e}")
        return pd.DataFrame()


def main():
    """Main application function."""

    st.title("📄 PDF Data Extractor")
    st.markdown(
        "*Extract tabular data and key-value pairs from PDFs with AI-powered precision*"
    )

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Extraction Settings")

        extraction_mode = st.selectbox(
            "Extraction Mode:",
            [
                "🗂️ Table Extraction",
                "🔑 Key-Value Extraction",
                "🔄 Both (Comprehensive)",
            ],
        )

        st.markdown("---")
        st.header("📋 Instructions")
        st.markdown(
            """
        **Table Extraction** - Best for:
        • Financial statements
        • Invoices with line items
        • Reports with data tables
        • Inventory lists
        
        **Key-Value Extraction** - Best for:
        • Forms and contracts
        • Contact information
        • Invoice headers
        • Document metadata
        
        **Both** - Comprehensive extraction for complex documents
        """
        )

    # Main content area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📤 Upload PDF Files")

        # Option to use demo file or upload own
        file_source = st.radio(
            "Choose file source:",
            ["📄 Use Demo Invoice PDF", "📁 Upload Your Own Files"],
            help="Try the demo invoice or upload your own PDF files",
        )

        uploaded_files = []
        demo_file_path = None

        if file_source == "📄 Use Demo Invoice PDF":
            demo_file_path = "./streamlit/data/pdf/sample_invoice.pdf"
            if os.path.exists(demo_file_path):
                st.success("✅ Using demo invoice PDF")
                st.info(
                    "📋 This sample invoice contains tables and key-value pairs perfect for testing both extraction modes."
                )

                # Show demo file details
                file_size = os.path.getsize(demo_file_path) / 1024  # KB
                st.write(f"**📄 File:** sample_invoice.pdf ({file_size:.1f} KB)")

                with st.expander("ℹ️ About the Demo Invoice"):
                    st.markdown(
                        """
                    This sample invoice includes:
                    - **Invoice header** with company details, invoice number, and dates
                    - **Bill-to information** with customer details  
                    - **Line items table** with products, quantities, and pricing
                    - **Payment terms** and totals
                    
                    Perfect for testing:
                    - **Table Extraction**: Extract the line items table
                    - **Key-Value Extraction**: Extract invoice number, dates, totals, etc.
                    - **Both modes**: Get comprehensive data extraction
                    """
                    )
            else:
                st.error("❌ Demo file not found. Please upload your own files.")
                demo_file_path = None
        else:
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Upload one or more PDF files to extract data from",
            )

            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

            # File details
            for i, file in enumerate(uploaded_files, 1):
                file_size = len(file.getvalue()) / 1024  # KB
                st.write(f"**{i}.** {file.name} ({file_size:.1f} KB)")

    with col2:
        # Determine which files to process
        files_to_process = []

        if demo_file_path and os.path.exists(demo_file_path):
            # Use demo file
            demo_file = DemoFile(demo_file_path)
            files_to_process = [demo_file]
        elif uploaded_files:
            # Use uploaded files
            files_to_process = uploaded_files

        if files_to_process:
            st.subheader("🔍 Extraction Results")

            # Process files
            for file_idx, file_to_process in enumerate(files_to_process):
                st.markdown(f"### 📄 Processing: {file_to_process.name}")

                # Extract text from PDF
                with st.spinner("Extracting text from PDF..."):
                    if hasattr(file_to_process, "file_path"):
                        # Demo file
                        extracted_text, num_pages = extract_text_from_demo_pdf(
                            file_to_process.file_path
                        )
                    else:
                        # Uploaded file
                        extracted_text, num_pages = extract_text_from_pdf(
                            file_to_process
                        )

                if not extracted_text.strip():
                    st.warning("⚠️ No text found in PDF. The file might be image-based.")
                    continue

                st.info(f"📊 Extracted text from {num_pages} pages")

                # Show raw text preview
                with st.expander("📝 Raw Text Preview"):
                    st.text_area(
                        "Extracted Text:",
                        value=(
                            extracted_text[:2000] + "..."
                            if len(extracted_text) > 2000
                            else extracted_text
                        ),
                        height=200,
                        key=f"raw_text_{file_idx}",
                    )

                # Perform extraction based on mode
                llm = get_extraction_llm()

                if extraction_mode in ["🗂️ Table Extraction", "🔄 Both (Comprehensive)"]:
                    with st.spinner("🤖 Extracting tables with AI..."):
                        table_results = extract_tables_from_text(extracted_text, llm)

                    if table_results.tables:
                        st.subheader("📊 Extracted Tables")

                        for i, table in enumerate(table_results.tables):
                            st.markdown(f"**Table {i+1}: {table.table_name}**")
                            st.markdown(f"*Confidence: {table.confidence_score}/10*")

                            # Convert to DataFrame and display
                            df = table_to_dataframe(table)

                            if not df.empty:
                                # Allow column renaming
                                st.markdown("**Column Management:**")
                                col_rename_cols = st.columns(min(len(df.columns), 4))

                                new_columns = {}
                                for idx, col in enumerate(df.columns):
                                    with col_rename_cols[idx % 4]:
                                        new_name = st.text_input(
                                            f"Rename '{col}':",
                                            value=col,
                                            key=f"rename_{file_idx}_{i}_{idx}",
                                        )
                                        if new_name != col:
                                            new_columns[col] = new_name

                                # Apply column renames
                                if new_columns:
                                    df = df.rename(columns=new_columns)

                                # Display table
                                st.dataframe(df, use_container_width=True)

                                # Data cleanup options
                                with st.expander("🧹 Data Cleanup Options"):
                                    col_clean1, col_clean2 = st.columns(2)

                                    with col_clean1:
                                        remove_empty = st.checkbox(
                                            "Remove empty rows",
                                            key=f"remove_empty_{file_idx}_{i}",
                                        )
                                        strip_whitespace = st.checkbox(
                                            "Strip whitespace",
                                            value=True,
                                            key=f"strip_{file_idx}_{i}",
                                        )

                                    with col_clean2:
                                        remove_duplicates = st.checkbox(
                                            "Remove duplicates",
                                            key=f"remove_dupes_{file_idx}_{i}",
                                        )
                                        convert_numeric = st.checkbox(
                                            "Auto-convert numbers",
                                            value=True,
                                            key=f"convert_num_{file_idx}_{i}",
                                        )

                                    # Apply cleanup
                                    if remove_empty:
                                        df = df.dropna(how="all")
                                    if strip_whitespace:
                                        df = df.applymap(
                                            lambda x: (
                                                x.strip() if isinstance(x, str) else x
                                            )
                                        )
                                    if remove_duplicates:
                                        df = df.drop_duplicates()
                                    if convert_numeric:
                                        for col in df.columns:
                                            df[col] = pd.to_numeric(
                                                df[col], errors="ignore"
                                            )

                                # Download options
                                csv_data = df.to_csv(index=False)
                                st.download_button(
                                    label=f"📥 Download Table {i+1} as CSV",
                                    data=csv_data,
                                    file_name=f"{file_to_process.name.replace('.pdf', '')}_{table.table_name.replace(' ', '_')}.csv",
                                    mime="text/csv",
                                    key=f"download_table_{file_idx}_{i}",
                                )
                            else:
                                st.warning("Unable to convert table to DataFrame")
                    else:
                        st.info("No tables found in this PDF")

                if extraction_mode in [
                    "🔑 Key-Value Extraction",
                    "🔄 Both (Comprehensive)",
                ]:
                    with st.spinner("🤖 Extracting key-value pairs with AI..."):
                        kv_results = extract_key_values_from_text(extracted_text, llm)

                    if kv_results.key_values:
                        st.subheader("🔑 Extracted Key-Value Pairs")

                        # Convert to DataFrame
                        kv_df = key_values_to_dataframe(kv_results.key_values)

                        # Display with filtering options
                        col_filter1, col_filter2 = st.columns(2)

                        with col_filter1:
                            min_confidence = st.slider(
                                "Minimum Confidence:",
                                1,
                                10,
                                5,
                                key=f"confidence_{file_idx}",
                            )

                        with col_filter2:
                            search_key = st.text_input(
                                "Search keys:",
                                placeholder="Enter keyword to filter...",
                                key=f"search_{file_idx}",
                            )

                        # Apply filters
                        filtered_df = kv_df.copy()
                        if search_key:
                            filtered_df = filtered_df[
                                filtered_df["Key"].str.contains(
                                    search_key, case=False, na=False
                                )
                                | filtered_df["Value"].str.contains(
                                    search_key, case=False, na=False
                                )
                            ]

                        # Filter by confidence (convert "x/10" format to int)
                        filtered_df["ConfidenceNum"] = (
                            filtered_df["Confidence"].str.split("/").str[0].astype(int)
                        )
                        filtered_df = filtered_df[
                            filtered_df["ConfidenceNum"] >= min_confidence
                        ]
                        filtered_df = filtered_df.drop("ConfidenceNum", axis=1)

                        st.dataframe(filtered_df, use_container_width=True)

                        # Download option
                        kv_csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Key-Value Pairs as CSV",
                            data=kv_csv,
                            file_name=f"{file_to_process.name.replace('.pdf', '')}_key_values.csv",
                            mime="text/csv",
                            key=f"download_kv_{file_idx}",
                        )
                    else:
                        st.info("No key-value pairs found in this PDF")

                # Show AI recommendations
                if (
                    extraction_mode == "🗂️ Table Extraction"
                    and "table_results" in locals()
                ):
                    if table_results.recommendations:
                        with st.expander("💡 AI Recommendations"):
                            for rec in table_results.recommendations:
                                st.write(f"• {rec}")

                elif (
                    extraction_mode == "🔑 Key-Value Extraction"
                    and "kv_results" in locals()
                ):
                    if kv_results.recommendations:
                        with st.expander("💡 AI Recommendations"):
                            for rec in kv_results.recommendations:
                                st.write(f"• {rec}")

                st.markdown("---")

        else:
            st.info("👆 Upload PDF files to begin extraction")

            # Show examples
            st.subheader("📖 Example Use Cases")

            examples = {
                "💰 Financial Reports": [
                    "Extract income statements",
                    "Parse balance sheet data",
                    "Analyze expense reports",
                ],
                "📋 Invoices & Bills": [
                    "Extract line items",
                    "Parse vendor information",
                    "Get payment terms",
                ],
                "📊 Business Documents": [
                    "Parse contract terms",
                    "Extract contact lists",
                    "Analyze survey data",
                ],
                "📈 Data Reports": [
                    "Extract performance metrics",
                    "Parse research data",
                    "Analyze survey results",
                ],
            }

            for category, items in examples.items():
                with st.expander(category):
                    for item in items:
                        st.write(f"• {item}")


if __name__ == "__main__":
    main()

# core.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder
)

from scipy.stats import chi2_contingency, pointbiserialr

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from google.colab import files


class PlottingMethods:
    """
    Utility plotting class for reusable Plotly charts.
    """

    @staticmethod
    def bar_chart(df, column):
        if df is None or column not in df.columns:
            return None

        counts = df[column].value_counts()
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            labels={"x": column, "y": "Count"},
            title=f"Bar Chart - {column}"
        )
        return fig.to_html()

    @staticmethod
    def pie_chart(df, column):
        if df is None or column not in df.columns:
            return None

        counts = df[column].value_counts()
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            title=f"Pie Chart - {column}"
        )
        return fig.to_html()

    @staticmethod
    def histogram(df, column):
        if df is None or column not in df.columns:
            return None

        fig = px.histogram(df, x=column, title=f"Histogram - {column}")
        return fig.to_html()


class DataInspector:
    """
    A reusable class for data ingestion, cleaning,
    preprocessing, normalization, and visualization.
    """

    def __init__(self):
        self.df = None

    def upload_data(self):
        """
        Upload CSV file in Google Colab and sanitize data.
        """
        uploaded = files.upload()
        filename = list(uploaded.keys())[0]
        self.df = pd.read_csv(filename)

        garbage_values = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ']
        self.df.replace(garbage_values, np.nan, inplace=True)

        self.auto_type_correction()

        print("Dataset Loaded Successfully")
        return self.df

    def auto_type_correction(self):
        """
        Convert columns to numeric if possible.
        """
        if self.df is None:
            return

        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors='coerce')
            if converted.notna().sum() > 0:
                self.df[col] = converted

    def data_summary(self):
        """
        Display dataset summary.
        """
        if self.df is None:
            print("No data loaded.")
            return

        print("=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)

        print("Rows:", self.df.shape[0])
        print("Columns:", self.df.shape[1])

        print("\nFIRST 20 ROWS")
        print(self.df.head(20))

        numeric_cols = self.df.select_dtypes(include=np.number).columns
        categorical_cols = self.df.select_dtypes(exclude=np.number).columns

        print("\nNUMERIC COLUMNS")
        print(list(numeric_cols))

        print("\nCATEGORICAL COLUMNS")
        print(list(categorical_cols))

    def handle_missing_values(self, strategy="mean", constant_value=None):
        """
        Impute missing values.
        Strategies: mean, median, mode, constant
        """
        if self.df is None:
            return

        for col in self.df.columns:
            if self.df[col].isna().sum() == 0:
                continue

            if strategy == "mean":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

            elif strategy == "median":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

            elif strategy == "mode":
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)

            elif strategy == "constant":
                self.df[col].fillna(constant_value, inplace=True)

        print("Missing values handled.")

    def remove_duplicates(self):
        """
        Remove duplicate rows.
        """
        if self.df is None:
            return

        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after = len(self.df)
        print(f"Removed {before - after} duplicate rows.")

    def handle_outliers(self, column, remove=False):
        """
        Detect or remove outliers using IQR.
        """
        if self.df is None:
            return None

        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (self.df[column] < lower) | (self.df[column] > upper)

        if remove:
            self.df = self.df[~mask]
            print("Outliers removed.")
        else:
            return self.df[mask]

    def delete_rows(self):
        """
        Delete rows via comma-separated input.
        """
        if self.df is None:
            return

        rows = input("Enter row indices separated by commas: ")
        rows = [int(i.strip()) for i in rows.split(",")]
        self.df.drop(rows, inplace=True, errors="ignore")
        print("Rows deleted.")

    def delete_columns(self):
        """
        Delete columns via comma-separated input.
        """
        if self.df is None:
            return

        cols = input("Enter columns separated by commas: ")
        cols = [c.strip() for c in cols.split(",")]
        self.df.drop(columns=cols, inplace=True, errors="ignore")
        print("Columns deleted.")

    def extract_normalized_numeric_data(self, method="minmax"):
        """
        Normalize numeric data.
        Methods: minmax, standard, robust
        """
        numeric = self.df.select_dtypes(include=np.number)

        if numeric.empty:
            return pd.DataFrame()

        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid method.")

        scaled = scaler.fit_transform(numeric)
        return pd.DataFrame(scaled, columns=numeric.columns)

    def extract_normalized_categorical_data(self, method="onehot"):
        """
        Encode categorical variables.
        Methods: onehot, ordinal, uniform
        """
        categorical = self.df.select_dtypes(exclude=np.number)

        if categorical.empty:
            return pd.DataFrame()

        if method == "onehot":
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = encoder.fit_transform(categorical)
            columns = encoder.get_feature_names_out(categorical.columns)
            return pd.DataFrame(encoded, columns=columns)

        elif method == "ordinal":
            encoder = OrdinalEncoder()
            encoded = encoder.fit_transform(categorical)
            return pd.DataFrame(encoded, columns=categorical.columns)

        elif method == "uniform":
            encoder = OrdinalEncoder()
            encoded = encoder.fit_transform(categorical)
            scaler = MinMaxScaler()
            encoded = scaler.fit_transform(encoded)
            return pd.DataFrame(encoded, columns=categorical.columns)

    def create_merged_dataset(self, cat_method="onehot"):
        """
        Merge numeric and encoded categorical data.
        """
        numeric = self.df.select_dtypes(include=np.number)
        categorical = self.extract_normalized_categorical_data(cat_method)
        return pd.concat([numeric, categorical], axis=1)

    def plot_numeric_distribution(self, column):
        """
        Create:
        1. Violin + Box Plot
        2. Scatter Plot
        3. Histogram
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Violin/Box", "Scatter", "Histogram"]
        )

        fig.add_trace(go.Violin(x=self.df[column], box_visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df[column], mode='markers'), row=1, col=2)
        fig.add_trace(go.Histogram(x=self.df[column]), row=1, col=3)

        fig.update_layout(title=f"Distribution Analysis: {column}")
        fig.show()

    def plot_relationship(self, col1, col2):
        """
        Automatically choose visualization based on data types.
        """
        num1 = pd.api.types.is_numeric_dtype(self.df[col1])
        num2 = pd.api.types.is_numeric_dtype(self.df[col2])

        if num1 and num2:
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols")
        elif (not num1) and num2:
            fig = px.box(self.df, x=col1, y=col2, points="all")
        elif num1 and (not num2):
            fig = px.box(self.df, x=col2, y=col1, points="all")
        else:
            temp = pd.crosstab(self.df[col1], self.df[col2])
            fig = px.bar(temp, barmode="group")

        fig.show()

    def categorical_frequency(self, column):
        """
        Plot counts with percentages.
        """
        counts = self.df[column].value_counts()
        percentages = (counts / counts.sum()) * 100

        fig = px.bar(
            x=counts.index,
            y=counts.values,
            text=[f"{p:.1f}%" for p in percentages]
        )
        fig.update_layout(title=f"Frequency Distribution: {column}")
        fig.show()

    def plot_all_associations_heatmap(self):
        """
        Correlation heatmap for numeric variables.
        """
        numeric = self.df.select_dtypes(include=np.number)

        if numeric.empty:
            print("No numeric columns found.")
            return

        corr = numeric.corr()
        fig = px.imshow(corr, text_auto=True, title="Association Heatmap")
        fig.show()
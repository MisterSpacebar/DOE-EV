import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set page title
st.set_page_config(page_title="EV Data Visualization", page_icon="truck", layout="wide")


# Function to read CSV files from a folder
def read_csv_files(folder_path):
    csv_files = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            csv_files[filename] = pd.read_csv(file_path)
    return csv_files


# Set the folder path relative to the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(current_dir, "EV_Data", "Reference", "vehicle")

# Read CSV files
try:
    csv_files = read_csv_files(folder_path)
except FileNotFoundError:
    st.error(f"Error: The folder '{folder_path}' was not found. Please check the path and try again.")
    st.stop()

# Sidebar for file selection
st.sidebar.title("File Selection")
st.sidebar.write("Choose a CSV file from the dropdown menu below to visualize its data.")
selected_file = st.sidebar.selectbox("Choose a CSV file", list(csv_files.keys()) if csv_files else [])

# Main content
st.title("EV Data Visualization Dashboard")
st.write("""
This dashboard allows you to explore and visualize data from electric vehicle (EV) CSV files. 
Select a file from the sidebar to begin your analysis.
""")

if selected_file:
    df = csv_files[selected_file]
    st.write(f"Displaying data for: **{selected_file}**")

    # Display basic information about the dataset
    st.header("Dataset Information")
    st.write("""
    This section provides an overview of the selected dataset, including its size and structure.
    """)
    st.write(f"**Number of rows:** {df.shape[0]}")
    st.write(f"**Number of columns:** {df.shape[1]}")

    # Display the first few rows of the dataset
    st.header("Data Preview")
    st.write("""
    Below is a preview of the first few rows of the dataset. This gives you a quick look at the 
    structure and content of the data.
    """)
    st.write(df.head())

    # Display summary statistics
    st.header("Summary Statistics")
    st.write("""
    This table shows summary statistics for the numerical columns in the dataset. It includes 
    measures like mean, standard deviation, minimum, and maximum values.
    """)
    st.write(df.describe())

    # Create a line plot of Total Distance over time
    st.header("Total Distance Over Time")
    st.write("""
    This line plot shows how the total distance traveled by the EV changes over time. It can help 
    identify trends or patterns in vehicle usage.
    """)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pd.to_datetime(df['Date']), df['Total Distance'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Distance')
    ax.set_title('Total Distance Over Time')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Create a scatter plot of Total Energy Consumption vs Total Distance
    st.header("Total Energy Consumption vs Total Distance")
    st.write("""
    This scatter plot compares the total energy consumption to the total distance traveled. It can 
    help visualize the relationship between distance and energy use, potentially revealing insights 
    about the vehicle's efficiency.
    """)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df['Total Distance'], df['Total Energy Consumption'])
    ax.set_xlabel('Total Distance')
    ax.set_ylabel('Total Energy Consumption')
    ax.set_title('Total Energy Consumption vs Total Distance')
    st.pyplot(fig)

    # Display correlation matrix
    st.header("Correlation Matrix")
    st.write("""
    The correlation matrix shows the strength and direction of relationships between different 
    numerical variables in the dataset. Values closer to 1 or -1 indicate stronger correlations.
    """)
    correlation_matrix = df.select_dtypes(include=[float, int]).corr()
    st.write(correlation_matrix)

    # Allow users to select columns for custom scatter plot
    st.header("Custom Scatter Plot")
    st.write("""
    Create your own scatter plot by selecting variables for the X and Y axes. This allows you to 
    explore relationships between any two variables in the dataset.
    """)
    col1, col2 = st.columns(2)
    with col1:
        x_column = st.selectbox("Select X-axis", df.columns)
    with col2:
        y_column = st.selectbox("Select Y-axis", df.columns)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df[x_column], df[y_column])
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f'{y_column} vs {x_column}')
    st.pyplot(fig)

else:
    st.write("""
    No CSV files found in the specified folder. Please check the folder path and ensure CSV files are present.

    Expected folder path: `{folder_path}`

    If you're sure the files exist in this location, there might be a permission issue or a problem with file naming.
    """)

# Add a footer with additional information
st.markdown("""
---
### About This Dashboard

This dashboard is designed to help analyze and visualize electric vehicle (EV) data. It provides various 
visualizations and statistics to gain insights into EV performance, usage patterns, and energy consumption.

For questions or support, please contact the development team.
""")
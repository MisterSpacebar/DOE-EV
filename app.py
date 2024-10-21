import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page title
st.set_page_config(page_title="EV Data Visualization", layout="wide")


# Function to read CSV files from a folder
def read_csv_files(folder_path):
    csv_files = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            csv_files[filename] = read_csv_with_encoding(file_path)
    return csv_files


# Function to read CSV with different encodings
def read_csv_with_encoding(file_path):
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    st.error(f"Unable to read the file {file_path} with any of the attempted encodings.")
    return None


# Function to convert time columns to hours
def convert_time_to_hours(df):
    time_columns = [col for col in df.columns if 'time' in col.lower() and 'date' not in col.lower()]
    for col in time_columns:
        if df[col].dtype == 'float64':
            df[col] = df[col] * 24  # Assuming the original values are in days
        elif df[col].dtype == 'object':
            # Try to convert string time to hours
            try:
                df[col] = pd.to_timedelta(df[col]).dt.total_seconds() / 3600
            except:
                st.warning(f"Could not convert column '{col}' to hours. It will remain unchanged.")
    return df


# Set the folder paths relative to the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
vehicle_data_folder = os.path.join(current_dir, "EV_Data", "Reference", "vehicle")
reference_data_file = os.path.join(current_dir, "EV_Data", "Reference", "data_reference", "vehicle_reference.csv")

# Read CSV files
try:
    csv_files = read_csv_files(vehicle_data_folder)
    reference_data = read_csv_with_encoding(reference_data_file)
    if reference_data is None:
        st.error("Failed to read the reference data file. Please check the file encoding.")
        st.stop()
except FileNotFoundError as e:
    st.error(f"Error: {e}. Please check the file paths and try again.")
    st.stop()

# Sidebar for file selection
st.sidebar.title("File Selection")
st.sidebar.write("Choose a CSV file from the dropdown menu below to visualize its data.")
selected_file = st.sidebar.selectbox("Choose a CSV file", list(csv_files.keys()) if csv_files else [])

# Main content
st.title("EV Data Visualization Dashboard")
st.write("""
This dashboard allows you to explore and visualize data from electric vehicle (EV) CSV files. 
Select a file from the sidebar to begin your analysis. Distance is measured in miles and time-based values are in hours.
""")

if selected_file:
    df = csv_files[selected_file]
    df = convert_time_to_hours(df)  # Convert time columns to hours

    # Extract vehicle ID from filename
    vehicle_id = selected_file.split('.')[
        1]  # Assuming the vehicle ID is always the second element when splitting by '.'

    # Look up additional information in the reference data
    vehicle_info = reference_data[reference_data['Vehicle ID'] == vehicle_id].iloc[0] if not reference_data[
        reference_data['Vehicle ID'] == vehicle_id].empty else None

    st.header("Vehicle Information")
    if vehicle_info is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Vehicle ID:** {vehicle_info['Vehicle ID']}")
            st.write(f"**Model Year:** {vehicle_info['Model Year']}")
            st.write(f"**Manufacturer:** {vehicle_info['Manufacturer']}")
        with col2:
            st.write(f"**Model Name:** {vehicle_info['Model Name']}")
            st.write(f"**Weight Class:** {vehicle_info['Weight Class']}")
            st.write(f"**Battery Chemistry:** {vehicle_info['Battery Chemistry']}")
        with col3:
            st.write(f"**Rated Energy:** {vehicle_info['Rated Energy']} kWh")
            st.write(f"**Max Charge Rate:** {vehicle_info['Max Charge Rate']} kW")
            st.write(f"**State:** {vehicle_info['State']}")
    else:
        st.write("No additional information available for this vehicle.")

    st.header("Data Preview")
    st.write("Below is a preview of the first few rows of the dataset.")
    st.write(df.head())

    st.header("Summary Statistics")
    st.write("This table shows summary statistics for the numerical columns in the dataset.")
    st.write(df.describe())

    # Create a line plot of Total Distance over time
    st.header("Total Distance Over Time")
    st.write(
        "This line chart shows how the total distance traveled by the EV changes over time. Distance is measured in miles.")

    date_column = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
    distance_column = next((col for col in df.columns if 'distance' in col.lower()), None)

    if date_column and distance_column:
        fig = px.line(df, x=date_column, y=distance_column, title='Total Distance Over Time')
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title='Total Distance (miles)')
        st.plotly_chart(fig)
    else:
        st.warning(
            "Couldn't find appropriate columns for Date and Total Distance. Please check the CSV file structure.")

    # Create a scatter plot of Total Energy Consumption vs Total Distance
    st.header("Total Energy Consumption vs Total Distance")
    st.write(
        "This scatter plot compares the total energy consumption to the total distance traveled. Distance is measured in miles.")

    energy_column = next((col for col in df.columns if 'energy' in col.lower() or 'consumption' in col.lower()), None)

    if distance_column and energy_column:
        fig = px.scatter(df, x=distance_column, y=energy_column,
                         title='Total Energy Consumption vs Total Distance')
        fig.update_xaxes(title='Total Distance (miles)')
        fig.update_yaxes(title='Total Energy Consumption')
        st.plotly_chart(fig)
    else:
        st.warning(
            "Couldn't find appropriate columns for Total Distance and Total Energy Consumption. Please check the CSV file structure.")

    # Create a bar chart of Driving Time vs Idling Time
    st.header("Driving Time vs Idling Time")
    st.write("This bar chart compares the driving time to the idling time. Time is measured in hours.")

    driving_time_column = next((col for col in df.columns if 'driving time' in col.lower()), None)
    idling_time_column = next((col for col in df.columns if 'idling time' in col.lower()), None)

    if driving_time_column and idling_time_column:
        avg_times = df[[driving_time_column, idling_time_column]].mean()
        fig = px.bar(x=avg_times.index, y=avg_times.values,
                     title='Average Driving Time vs Idling Time')
        fig.update_xaxes(title='')
        fig.update_yaxes(title='Time (hours)')
        st.plotly_chart(fig)
    else:
        st.warning(
            "Couldn't find appropriate columns for Driving Time and Idling Time. Please check the CSV file structure.")

    # Display correlation matrix
    st.header("Correlation Matrix")
    st.write(
        "The correlation matrix shows the strength and direction of relationships between different numerical variables in the dataset.")
    correlation_matrix = df.select_dtypes(include=[float, int]).corr()
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto")
    fig.update_layout(title='Correlation Matrix')
    st.plotly_chart(fig)

    # Allow users to select columns for custom scatter plot
    st.header("Custom Scatter Plot")
    st.write("Create your own scatter plot by selecting variables for the X and Y axes.")
    numeric_columns = df.select_dtypes(include=[float, int]).columns
    col1, col2 = st.columns(2)
    with col1:
        x_column = st.selectbox("Select X-axis", numeric_columns)
    with col2:
        y_column = st.selectbox("Select Y-axis", numeric_columns)

    fig = px.scatter(df, x=x_column, y=y_column, title=f'{y_column} vs {x_column}')
    st.plotly_chart(fig)

else:
    st.write("""
    No CSV files found in the specified folder. Please check the folder path and ensure CSV files are present.

    Expected folder path: `{vehicle_data_folder}`

    If you're sure the files exist in this location, there might be a permission issue or a problem with file naming.
    """)

# Add a footer with additional information
st.markdown("""
---
### About This Dashboard

This dashboard is designed to help analyze and visualize electric vehicle (EV) data. It provides various 
visualizations and statistics to gain insights into EV performance, usage patterns, and energy consumption.
All distance measurements are in miles and time-based values are in hours.

For questions or support, please contact the development team.
""")
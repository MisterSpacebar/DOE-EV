import streamlit as st
import pandas as pd
import os
import json
from pages.fleetdata import EVDataAnalyzer
from pages.vehicledata import VehicleDataAnalyzer

# Set page configuration
st.set_page_config(page_title="EV Data Visualization", layout="wide", page_icon="🚗")


# Function to read CSV files from a folder
def read_csv_files(folder_path):
    csv_files = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                csv_files[filename] = df
            except Exception as e:
                st.error(f"Error reading {filename}: {e}")
    return csv_files


def main():
    # Set the folder paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vehicle_data_folder = os.path.join(current_dir, "EV_Data", "Reference", "vehicle")
    reference_data_file = os.path.join(current_dir, "EV_Data", "Reference", "data_reference",
                                       "vehicle_reference_json.json")

    # Load data
    try:
        csv_files = read_csv_files(vehicle_data_folder)
        with open(reference_data_file, 'r') as json_file:
            reference_data = json.load(json_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    analysis_mode = st.sidebar.radio(
        "Select Analysis Mode",
        ["Individual Vehicle", "Fleet Analysis"]
    )

    st.title("EV Data Visualization Dashboard")

    if analysis_mode == "Individual Vehicle":
        # Individual vehicle analysis
        selected_file = st.sidebar.selectbox(
            "Choose a Vehicle",
            list(csv_files.keys()) if csv_files else []
        )

        if selected_file:
            # Get vehicle data and info
            df = csv_files[selected_file]
            vehicle_code = selected_file.split('.')[1]
            vehicle_info = next(
                (item for item in reference_data if item['Vehicle ID'].lower() == vehicle_code.lower()),
                None
            )

            # Create vehicle analyzer instance
            vehicle_analyzer = VehicleDataAnalyzer(df, vehicle_info)

            # Display vehicle information
            st.header("Vehicle Information")
            info_display = vehicle_analyzer.get_vehicle_info_display()
            if info_display:
                col1, col2, col3 = st.columns(3)
                for col, items in zip([col1, col2, col3], info_display.values()):
                    with col:
                        for key, value in items.items():
                            st.write(f"**{key}:** {value}")

            # Display data preview
            st.header("Data Preview")
            st.write(df.head())

            # Display summary statistics
            st.header("Summary Statistics")
            summary_stats = vehicle_analyzer.calculate_summary_statistics()
            if summary_stats is not None:
                st.write(summary_stats)
            else:
                st.warning("No numeric data available for statistics")

            # Display visualizations
            visualizations = vehicle_analyzer.generate_visualizations()

            for title, fig in visualizations.items():
                st.plotly_chart(fig, use_container_width=True)

            # Custom scatter plot
            st.header("Custom Scatter Plot")
            numeric_columns = vehicle_analyzer.get_numeric_columns()
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Select X-axis", numeric_columns)
            with col2:
                y_column = st.selectbox("Select Y-axis", numeric_columns)

            custom_fig = vehicle_analyzer.create_custom_scatter(x_column, y_column)
            st.plotly_chart(custom_fig, use_container_width=True)



    else:
        # Fleet analysis
        fleet_analyzer = EVDataAnalyzer(csv_files, reference_data)

        if fleet_analyzer.aggregated_data.empty:
            st.error("No data was successfully aggregated. Please check your data files.")

            st.stop()

        try:
            # Display fleet summary
            summary = fleet_analyzer.get_fleet_summary()
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Vehicles", summary['Total_Vehicles'])
                st.metric("Total Manufacturers", summary['Total_Manufacturers'])

            with col2:
                st.metric("Regions Covered", summary['Regions_Covered'])
                st.metric("States Covered", summary['Total_States'])

            with col3:
                if 'Avg_Rated_Energy' in summary:
                    st.metric("Avg Rated Energy", f"{summary['Avg_Rated_Energy']:.1f} kWh")
                if 'Most_Common_Chemistry' in summary:
                    st.metric("Most Common Battery", summary['Most_Common_Chemistry'])

            # Category analysis
            st.subheader("Analysis by Category")

            category = st.selectbox(
                "Select Category for Analysis",
               ["Manufacturer", "Region", "Sector", "Vocation"]
            )

            category_stats = fleet_analyzer.analyze_by_category(category)

            if not category_stats.empty:
                st.write(category_stats)
            else:
                st.warning("No data available for the selected category.")

            # Visualizations
            st.subheader("Visual Analysis")
            visuals = fleet_analyzer.generate_visualizations()

            if not visuals:
                st.warning("No visualizations could be generated. This might be due to missing or invalid data.")
            else:
                viz_options = st.multiselect(
                    "Select Visualizations",
                    list(visuals.keys()),
                   default=list(visuals.keys())[0:2] if len(visuals) >= 2 else list(visuals.keys())
                )
                for viz in viz_options:
                    st.plotly_chart(visuals[viz], use_container_width=True)

        except Exception as e:
            st.error(f"Error displaying fleet analysis: {str(e)}")
            st.write("Debug information:")
            st.write(f"Number of files: {len(csv_files)}")
            st.write(f"Reference data shape: {fleet_analyzer.reference_data.shape}")

if __name__ == "__main__":
    main()
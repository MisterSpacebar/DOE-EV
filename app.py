import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px
from pages.fleetdata import EVDataAnalyzer
from pages.vehicledata import VehicleDataAnalyzer
from pages.categorydata import CategoryDataAnalyzer

# Set page configuration
st.set_page_config(page_title="EV Data Visualization", layout="wide", page_icon="truck")


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
        ["Category Analysis", "Individual Vehicle", "Fleet Analysis"],
        index=0  # Set Category Analysis as default
    )

    st.title("EV Data Visualization Dashboard")

    if analysis_mode == "Category Analysis":
        # Initialize category analyzer
        category_analyzer = CategoryDataAnalyzer(csv_files, reference_data)

        # Get available categories
        categories = category_analyzer.get_unique_categories()

        # Category selection
        st.sidebar.subheader("Filter Options")

        # Initialize session state for selections if not exists
        if 'prev_manufacturer' not in st.session_state:
            st.session_state.prev_manufacturer = "All"
        if 'prev_weight_class' not in st.session_state:
            st.session_state.prev_weight_class = "All"

        # Category selection
        selected_manufacturer = st.sidebar.selectbox(
            "Select Manufacturer",
            ["All"] + categories['manufacturers']
        )

        selected_weight_class = st.sidebar.selectbox(
            "Select Weight Class",
            ["All"] + categories['weight_classes']
        )

        # Reset logic
        if selected_manufacturer != st.session_state.prev_manufacturer:
            if selected_manufacturer != "All" and selected_weight_class != "All":
                selected_weight_class = "All"
                st.session_state.prev_weight_class = "All"
                st.rerun()
            st.session_state.prev_manufacturer = selected_manufacturer

        if selected_weight_class != st.session_state.prev_weight_class:
            if selected_weight_class != "All" and selected_manufacturer != "All":
                selected_manufacturer = "All"
                st.session_state.prev_manufacturer = "All"
                st.rerun()
            st.session_state.prev_weight_class = selected_weight_class

        # Convert "All" to None for filtering
        manufacturer_filter = None if selected_manufacturer == "All" else selected_manufacturer
        weight_filter = None if selected_weight_class == "All" else selected_weight_class

        # Analysis type selection
        analysis_type = st.sidebar.radio(
            "Select Analysis Type",
            ["Overview", "Performance Metrics", "Statistical Analysis"]
        )

        if analysis_type == "Overview":
            # Create display title based on selections
            if manufacturer_filter:
                category_title = f"for {selected_manufacturer}"
            elif weight_filter:
                category_title = f"for Weight Class {selected_weight_class}"
            else:
                category_title = "for All Vehicles"

            # Display summary metrics
            st.header(f"Category Summary {category_title}")
            summary = category_analyzer.get_category_summary(manufacturer_filter, weight_filter)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Vehicles", summary.get('total_vehicles', 0))
                st.metric("Total Manufacturers", summary.get('total_manufacturers', 0))
            with col2:
                st.metric("Total Distance (mi)", f"{summary.get('total_distance', 0):,.0f}")
                st.metric("Total Weight Classes", summary.get('total_weight_classes', 0))
            with col3:
                st.metric("Total Energy (kWh)", f"{summary.get('total_energy', 0):,.0f}")
                if summary.get('avg_temperature'):
                    st.metric("Avg Temperature (°F)", f"{summary['avg_temperature']:.1f}")

            # Add comprehensive statistics section
            st.header(f"Comprehensive Statistics {category_title}")
            detailed_stats = category_analyzer.calculate_detailed_stats(manufacturer_filter, weight_filter)

            if not detailed_stats.empty:
                # Display the statistics table
                st.dataframe(detailed_stats)

                # Create tabs for different metric visualizations
                metric_tabs = st.tabs([
                    "Energy Metrics",
                    "Distance Metrics",
                    "Temperature Metrics",
                    "Time Metrics"
                ])

                with metric_tabs[0]:
                    st.subheader(f"Energy Performance Statistics {category_title}")
                    energy_cols = detailed_stats.loc[
                        ['Energy Efficiency (kWh/mi)', 'Total Energy (kWh)', 'Avg Energy per Trip (kWh)']]
                    st.dataframe(energy_cols)

                with metric_tabs[1]:
                    st.subheader(f"Distance Statistics {category_title}")
                    distance_cols = detailed_stats.loc[['Total Distance (mi)', 'Avg Trip Distance (mi)']]
                    st.dataframe(distance_cols)

                with metric_tabs[2]:
                    st.subheader(f"Temperature Statistics {category_title}")
                    temp_cols = detailed_stats.loc[
                        ['Avg Temperature (°F)', 'Min Temperature (°F)', 'Max Temperature (°F)']]
                    st.dataframe(temp_cols)

                with metric_tabs[3]:
                    st.subheader(f"Time and Utilization Statistics {category_title}")
                    time_cols = detailed_stats.loc[
                        ['Total Drive Time (hrs)', 'Total Idle Time (hrs)', 'Avg Idle Time (%)']]
                    st.dataframe(time_cols)

                # Display statistical visualizations
                st.header(f"Statistical Distributions {category_title}")
                stat_visuals = category_analyzer.generate_stats_visualizations(detailed_stats)

                col1, col2 = st.columns(2)
                for i, (title, fig) in enumerate(stat_visuals.items()):
                    with col1 if i % 2 == 0 else col2:
                        st.plotly_chart(fig, use_container_width=True)

            # Display visualizations
            st.header(f"Performance Overview {category_title}")
            visuals = category_analyzer.generate_visualizations(manufacturer_filter, weight_filter)

            col1, col2 = st.columns(2)
            with col1:
                if 'efficiency' in visuals:
                    st.plotly_chart(visuals['efficiency'], use_container_width=True)
                if 'temperature' in visuals:
                    st.plotly_chart(visuals['temperature'], use_container_width=True)
            with col2:
                if 'energy_distance' in visuals:
                    st.plotly_chart(visuals['energy_distance'], use_container_width=True)
                if 'idle_time' in visuals:
                    st.plotly_chart(visuals['idle_time'], use_container_width=True)


        elif analysis_type == "Performance Metrics":
            st.header("Detailed Performance Metrics")

            # Calculate and display performance metrics
            metrics_df = category_analyzer.calculate_performance_metrics(manufacturer_filter, weight_filter)
            if not metrics_df.empty:
                st.dataframe(metrics_df)

            # Display trend analysis
            st.header("Performance Trends")
            trend_data = category_analyzer.perform_trend_analysis(manufacturer_filter, weight_filter)
            if trend_data and 'daily_metrics' in trend_data:
                st.subheader("Daily Performance Metrics")
                st.line_chart(trend_data['daily_metrics'].set_index('Trip_Date')[
                                  ['Total Distance', 'Total Energy Consumption']
                              ])

        else:  # Statistical Analysis
            st.header("Statistical Analysis")

            # Display statistical summary
            stats = category_analyzer.calculate_statistical_summary(manufacturer_filter, weight_filter)

            st.subheader("Basic Statistics")
            if 'basic_stats' in stats:
                st.dataframe(stats['basic_stats'])

            # Display correlation analysis
            st.subheader("Correlation Analysis")
            corr_matrix, heatmap = category_analyzer.calculate_correlations(manufacturer_filter, weight_filter)
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)

            # Display statistical comparisons
            st.subheader("Category Comparisons")
            comparison_results = category_analyzer.perform_category_comparison(
                "Manufacturer" if not manufacturer_filter else "Weight_Class"
            )
            if comparison_results:
                for metric, results in comparison_results.items():
                    st.write(f"**{metric}**")
                    st.write(f"- F-statistic: {results['f_statistic']:.3f}")
                    st.write(f"- p-value: {results['p_value']:.3f}")
                    st.write(f"- Effect size (η²): {results['effect_size']:.3f}")
                    st.write(f"- Statistically significant: {results['significant']}")

            # Display statistical visualizations
            st.header("Statistical Visualizations")
            stat_visuals = category_analyzer.generate_statistical_visualizations(
                manufacturer_filter, weight_filter
            )

            for title, fig in stat_visuals.items():
                st.plotly_chart(fig, use_container_width=True)

    elif analysis_mode == "Individual Vehicle":
        # [Keep existing Individual Vehicle analysis code]
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




    else:  # Fleet Analysis mode

        fleet_analyzer = EVDataAnalyzer(csv_files, reference_data)

        if fleet_analyzer.aggregated_data.empty:
            st.error("No data was successfully aggregated. Please check your data files.")

            st.stop()

        analysis_type = st.sidebar.radio(

            "Select Analysis Type",

            ["Fleet Overview", "Comparative Analysis", "Statistical Summary"]

        )

        if analysis_type == "Fleet Overview":
            st.header("Fleet Overview")

            # Display basic fleet metrics
            fleet_summary = fleet_analyzer.get_fleet_summary()
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Vehicles", fleet_summary['Total_Vehicles'])
                st.metric("Total Manufacturers", fleet_summary['Total_Manufacturers'])
            with col2:
                st.metric("Regions Covered", fleet_summary['Regions_Covered'])
                st.metric("Total States", fleet_summary['Total_States'])
            with col3:
                st.metric("Date Range", fleet_summary['Date_Range'])
                if 'Avg_Rated_Energy' in fleet_summary:
                    st.metric("Avg Rated Energy (kWh)", f"{fleet_summary['Avg_Rated_Energy']:.1f}")

            # Add category analysis section
            st.subheader("Fleet Category Analysis")
            category = st.selectbox(
                "Select Analysis Category",
                ["Manufacturer", "Weight Class"]
            )

            # Get analysis results - make sure this matches the actual method name in your EVDataAnalyzer class
            try:
                if hasattr(fleet_analyzer, 'analyze_manufacturers_and_weights'):  # Changed method name here
                    analysis_results = fleet_analyzer.analyze_manufacturers_and_weights()  # And here
                else:
                    st.error("Analysis method not found. Please check the implementation.")
                    st.write(f"Available methods: {dir(fleet_analyzer)}")  # Add debug info
                    analysis_results = {}

                if analysis_results:
                    if category == "Manufacturer":
                        st.write("### Manufacturer Analysis")
                        st.dataframe(analysis_results['manufacturer'])
                    else:
                        st.write("### Weight Class Analysis")
                        st.dataframe(analysis_results['weight_class'])

                    # Display visualizations
                    visuals = fleet_analyzer.generate_category_visualizations(
                        'Manufacturer' if category == "Manufacturer" else 'Weight_Class'
                    )

                    if visuals:
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'efficiency_temp' in visuals:
                                st.plotly_chart(visuals['efficiency_temp'], use_container_width=True)
                            if 'operation_patterns' in visuals:
                                st.plotly_chart(visuals['operation_patterns'], use_container_width=True)
                        with col2:
                            if 'performance_matrix' in visuals:
                                st.plotly_chart(visuals['performance_matrix'], use_container_width=True)
                            if 'temperature_impact' in visuals:
                                st.plotly_chart(visuals['temperature_impact'], use_container_width=True)
                else:
                    st.warning("No analysis data available for the selected category.")
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")



        elif analysis_type == "Comparative Analysis":

            st.header("Comparative Analysis")

            try:

                # Get comparative statistics with error handling

                with st.spinner("Analyzing manufacturer data..."):

                    vehicle_stats, manufacturer_stats = fleet_analyzer.compare_manufacturers()

                    # Check if we have valid data

                    if vehicle_stats.empty:
                        st.warning("No vehicle statistics are available.")

                        st.stop()

                    if manufacturer_stats.empty:
                        st.warning("No manufacturer summary statistics are available.")

                    # Display manufacturer summary

                    st.subheader("Manufacturer Summary Statistics")

                    if not manufacturer_stats.empty:
                        st.dataframe(manufacturer_stats)

                    # Display detailed comparisons

                    st.subheader("Detailed Vehicle Statistics")

                    metric_option = st.selectbox(

                        "Select Metric to Analyze",

                        ["Energy Efficiency", "Trip Patterns", "Idle Time", "Temperature Impact"]

                    )

                    if metric_option == "Energy Efficiency":

                        st.write("### Energy Efficiency Analysis")

                        # Check if required columns exist

                        if all(col in vehicle_stats.columns for col in
                               ['Manufacturer', 'Model', 'Vehicle_ID', 'Energy_per_Mile']):

                            efficiency_stats = vehicle_stats[

                                ['Manufacturer', 'Model', 'Vehicle_ID', 'Energy_per_Mile']

                            ].sort_values('Energy_per_Mile')

                            st.write("Energy Efficiency Rankings (kWh/mile):")

                            st.dataframe(efficiency_stats)

                            # Show visualization

                            visuals = fleet_analyzer.generate_comparative_visualizations()

                            if 'energy_efficiency' in visuals:
                                st.plotly_chart(visuals['energy_efficiency'], use_container_width=True)

                        else:

                            st.warning("Energy efficiency data is not available.")


                    elif metric_option == "Trip Patterns":

                        st.write("### Trip Pattern Analysis")

                        # Show trip statistics if available

                        trip_cols = [col for col in vehicle_stats.columns if 'Distance' in col]

                        if trip_cols:

                            trip_stats = vehicle_stats[

                                ['Manufacturer', 'Model', 'Vehicle_ID'] + trip_cols

                                ]

                            st.dataframe(trip_stats)

                            # Show visualization

                            visuals = fleet_analyzer.generate_comparative_visualizations()

                            if 'trip_distance' in visuals:
                                st.plotly_chart(visuals['trip_distance'], use_container_width=True)

                        else:

                            st.warning("Trip pattern data is not available.")


                    elif metric_option == "Idle Time":

                        st.write("### Idle Time Analysis")

                        # Check if idle percentage data is available

                        if 'Idle_Percentage' in vehicle_stats.columns:

                            idle_stats = vehicle_stats[

                                ['Manufacturer', 'Model', 'Vehicle_ID', 'Idle_Percentage']

                            ].sort_values('Idle_Percentage')

                            st.write("Idle Time Rankings (% of total time):")

                            st.dataframe(idle_stats)

                            # Show visualization

                            visuals = fleet_analyzer.generate_comparative_visualizations()

                            if 'idle_comparison' in visuals:
                                st.plotly_chart(visuals['idle_comparison'], use_container_width=True)

                        else:

                            st.warning("Idle time data is not available.")


                    else:  # Temperature Impact

                        st.write("### Temperature Impact Analysis")

                        temp_cols = [col for col in vehicle_stats.columns if 'Temperature' in col]

                        if temp_cols:

                            temp_stats = vehicle_stats[

                                ['Manufacturer', 'Model', 'Vehicle_ID'] + temp_cols

                                ]

                            st.dataframe(temp_stats)

                            # Show energy vs temperature relationship

                            visuals = fleet_analyzer.generate_comparative_visualizations()

                            if 'energy_distance_scatter' in visuals:

                                st.plotly_chart(visuals['energy_distance_scatter'], use_container_width=True)

                            else:

                                st.warning("Temperature impact visualization is not available.")

                        else:

                            st.warning("Temperature data is not available.")


            except Exception as e:

                st.error(f"An error occurred during comparative analysis. Please check the data and try again.")

                st.write(f"Error details: {str(e)}")




        else:  # Statistical Summary

            st.header("Statistical Summary")

            # Get comprehensive statistics

            stats = fleet_analyzer.generate_statistical_summary()

            # Display fleet-wide summary

            st.subheader("Fleet-Wide Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric("Total Vehicles", stats['fleet_summary']['total_vehicles'])

                st.metric("Total Trips", stats['fleet_summary']['total_trips'])

            with col2:

                st.metric("Total Distance (miles)", f"{stats['fleet_summary']['total_distance']:,.0f}")

                st.metric("Avg Trip Distance", f"{stats['fleet_summary']['avg_trip_distance']:.1f}")

            with col3:

                st.metric("Total Energy (kWh)", f"{stats['fleet_summary']['total_energy']:,.0f}")

            # Display percentile analysis

            st.subheader("Percentile Analysis")

            for metric, percentiles in stats['percentiles'].items():
                st.write(f"### {metric}")

                st.write(pd.DataFrame([percentiles]))

            # Show correlations with improved layout

            st.subheader("Correlation Analysis")

            # Get numeric columns and create correlation matrix

            numeric_data = fleet_analyzer.aggregated_data.select_dtypes(include=['float64', 'int64'])

            # Filter out columns that might not be useful for correlation

            excluded_columns = ['index', 'id', 'year']  # Add any columns you want to exclude

            correlation_columns = [col for col in numeric_data.columns if
                                   not any(excl in col.lower() for excl in excluded_columns)]

            correlation_matrix = numeric_data[correlation_columns].corr()

            # Create a larger and more readable correlation heatmap

            fig = px.imshow(

                correlation_matrix,

                title="Correlation Matrix of Key Metrics",

                # aspect='auto',  # This helps with the aspect ratio

                width=800,  # Increased width

                height=800,  # Increased height

                color_continuous_scale='RdBu_r',  # Better color scale for correlations

            )

            # Update layout for better readability

            fig.update_layout(

                title_x=0.5,  # Center the title

                title_y=0.95,  # Move title up slightly

                font=dict(size=10),  # Increase font size

                xaxis=dict(tickangle=45),  # Angle x-axis labels for better fit

            )

            # Add value annotations on the heatmap

            fig.update_traces(

                text=correlation_matrix.round(2),  # Show correlation values with 2 decimal places

                texttemplate='%{text}',

                textfont={"size": 10},  # Adjust text size

                showscale=True,

            )

            # Display the correlation matrix

            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
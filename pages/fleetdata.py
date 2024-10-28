from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
from scipy import stats
import logging


class EVDataAnalyzer:
    def __init__(self, csv_files: Dict[str, pd.DataFrame], reference_data: List[dict]):
        self.csv_files = csv_files
        self.reference_data = pd.DataFrame(reference_data)
        self.aggregated_data = self._aggregate_data()

        # Add logging for debugging
        print(f"CSV files count: {len(self.csv_files)}")
        print(f"Reference data shape: {self.reference_data.shape}")
        print(f"Aggregated data shape: {self.aggregated_data.shape}")
        print(f"Aggregated data columns: {self.aggregated_data.columns.tolist()}")

    def _aggregate_data(self) -> pd.DataFrame:
        """Combine all vehicle data into a single DataFrame with additional metadata"""
        all_data = []

        print("\nStarting data aggregation...")
        print(f"Processing {len(self.csv_files)} files")

        for filename, df in self.csv_files.items():
            try:
                vehicle_id = filename.split('.')[1]
                print(f"\nProcessing {vehicle_id}")

                # Try case-insensitive matching
                vehicle_info = self.reference_data[
                    self.reference_data['Vehicle ID'].str.lower() == vehicle_id.lower()
                    ]

                if not vehicle_info.empty:
                    df_copy = df.copy()

                    # Convert date/time columns if they exist
                    date_columns = [col for col in df_copy.columns if 'time' in col.lower() or 'date' in col.lower()]
                    for col in date_columns:
                        try:
                            df_copy[col] = pd.to_datetime(df_copy[col])
                        except Exception as e:
                            print(f"Could not convert column {col} to datetime: {e}")

                    # Add metadata
                    df_copy['Vehicle_ID'] = vehicle_id
                    metadata_columns = {
                        'Manufacturer': 'Manufacturer',
                        'Model_Name': 'Model Name',
                        'Weight_Class': 'Weight Class',
                        'Battery_Chemistry': 'Battery Chemistry',
                        'Rated_Energy': 'Rated Energy',
                        'State': 'State',
                        'Region': 'Region',
                        'Sector': 'Sector',
                        'Vocation': 'Vocation'
                    }

                    for col_name, ref_col in metadata_columns.items():
                        try:
                            df_copy[col_name] = vehicle_info.iloc[0][ref_col]
                        except Exception as e:
                            print(f"Error adding column {col_name}: {e}")
                            df_copy[col_name] = 'Unknown'

                    all_data.append(df_copy)
                    print(f"Successfully processed {vehicle_id}")
                else:
                    print(f"No reference data found for {vehicle_id}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        if not all_data:
            print("No data was aggregated!")
            return pd.DataFrame()

        try:
            result = pd.concat(all_data, ignore_index=True)
            print(f"\nSuccessfully aggregated {len(all_data)} vehicles")
            print(f"Final dataset shape: {result.shape}")
            return result
        except Exception as e:
            print(f"Error in final concatenation: {e}")
            return pd.DataFrame()

        def _generate_vehicle_summaries(self) -> pd.DataFrame:
            """Generate summary statistics for each individual vehicle"""
            print("\nGenerating vehicle summaries...")
            summaries = []

            for filename, df in self.csv_files.items():
                try:
                    vehicle_id = filename.split('.')[1]
                    print(f"Processing summary for {vehicle_id}")

                    summary = {}
                    summary['Vehicle_ID'] = vehicle_id

                    # Basic metrics to analyze
                    metrics = [
                        'Total Distance',
                        'Total Energy Consumption',
                        'SOC Used',
                        'Average Ambient Temperature',
                        'Driving Time',
                        'Idling Time'
                    ]

                    # Calculate statistics for each metric
                    for metric in metrics:
                        if metric in df.columns:
                            summary[f"{metric}_mean"] = df[metric].mean()
                            summary[f"{metric}_median"] = df[metric].median()
                            summary[f"{metric}_std"] = df[metric].std()
                            summary[f"{metric}_min"] = df[metric].min()
                            summary[f"{metric}_max"] = df[metric].max()

                    # Add metadata from reference data
                    vehicle_info = self.reference_data[
                        self.reference_data['Vehicle ID'].str.lower() == vehicle_id.lower()
                        ]
                    if not vehicle_info.empty:
                        summary['Manufacturer'] = vehicle_info.iloc[0]['Manufacturer']
                        summary['Model_Name'] = vehicle_info.iloc[0]['Model Name']
                        summary['Weight_Class'] = vehicle_info.iloc[0]['Weight Class']
                        summary['Battery_Chemistry'] = vehicle_info.iloc[0]['Battery Chemistry']
                        summary['Rated_Energy'] = vehicle_info.iloc[0]['Rated Energy']

                    summaries.append(summary)
                    print(f"Successfully generated summary for {vehicle_id}")

                except Exception as e:
                    print(f"Error generating summary for {filename}: {e}")
                    continue

            if not summaries:
                print("No summaries were generated")
                return pd.DataFrame()

            summaries_df = pd.DataFrame(summaries)
            print(f"Generated summaries for {len(summaries)} vehicles")
            return summaries_df

    def get_fleet_summary(self) -> Dict:
        """Generate summary statistics for the entire fleet"""
        summary = {
            'Total_Vehicles': len(self.csv_files),
            'Total_Manufacturers': self.reference_data['Manufacturer'].nunique(),
            'Regions_Covered': self.reference_data['Region'].nunique(),
            'Total_States': self.reference_data['State'].nunique(),
            'Date_Range': 'N/A'  # Default value
        }

        # Add conditional statistics
        if not self.reference_data.empty:
            rated_energy = self.reference_data['Rated Energy'].dropna()
            if not rated_energy.empty:
                summary['Avg_Rated_Energy'] = rated_energy.mean()
            else:
                summary['Avg_Rated_Energy'] = 0

            battery_chem = self.reference_data['Battery Chemistry'].dropna()
            if not battery_chem.empty:
                summary['Most_Common_Chemistry'] = battery_chem.mode().iloc[0]
            else:
                summary['Most_Common_Chemistry'] = 'Unknown'

        # Try to get date range if available
        try:
            if not self.aggregated_data.empty and 'Local Trip Start Time' in self.aggregated_data.columns:
                # Convert to datetime if not already
                start_times = pd.to_datetime(self.aggregated_data['Local Trip Start Time'])
                end_times = pd.to_datetime(self.aggregated_data['Local Trip End Time'])

                summary[
                    'Date_Range'] = f"{start_times.min().strftime('%Y-%m-%d')} to {end_times.max().strftime('%Y-%m-%d')}"
        except Exception as e:
            print(f"Error calculating date range: {e}")
            summary['Date_Range'] = 'Error calculating date range'

        return summary

    def generate_visualizations(self):
        """Generate a set of standard visualizations for the dashboard"""
        visualizations = {}

        if self.aggregated_data.empty:
            return visualizations

        try:
            # Energy consumption by manufacturer
            if 'Manufacturer' in self.aggregated_data.columns and 'Total Energy Consumption' in self.aggregated_data.columns:
                visualizations['energy_by_manufacturer'] = px.box(
                    self.aggregated_data,
                    x='Manufacturer',
                    y='Total Energy Consumption',
                    title='Energy Consumption Distribution by Manufacturer'
                )

            # Distance by region
            if 'Region' in self.aggregated_data.columns and 'Total Distance' in self.aggregated_data.columns:
                visualizations['distance_by_region'] = px.violin(
                    self.aggregated_data,
                    x='Region',
                    y='Total Distance',
                    title='Distance Distribution by Region'
                )

            # Temperature impact
            if all(col in self.aggregated_data.columns for col in
                   ['Average Ambient Temperature', 'Total Energy Consumption', 'Region']):
                visualizations['temperature_impact'] = px.scatter(
                    self.aggregated_data,
                    x='Average Ambient Temperature',
                    y='Total Energy Consumption',
                    color='Region',
                    title='Temperature Impact on Energy Consumption'
                )

            # Battery chemistry distribution
            if 'Battery_Chemistry' in self.aggregated_data.columns:
                visualizations['chemistry_distribution'] = px.pie(
                    self.aggregated_data,
                    names='Battery_Chemistry',
                    title='Distribution of Battery Chemistry Types'
                )
        except Exception as e:
            print(f"Error generating visualizations: {e}")

        return visualizations

    def analyze_by_category(self, category: str) -> pd.DataFrame:
        """Generate statistics grouped by a specific category"""
        if self.aggregated_data.empty or category not in self.aggregated_data.columns:
            return pd.DataFrame()

        try:
            required_columns = ['Total Distance', 'Total Energy Consumption', 'SOC Used', 'Average Ambient Temperature']
            available_columns = [col for col in required_columns if col in self.aggregated_data.columns]

            if not available_columns:
                return pd.DataFrame()

            return self.aggregated_data.groupby(category)[available_columns].agg([
                'mean', 'std', 'count'
            ]).round(2)
        except Exception as e:
            print(f"Error in analyze_by_category: {e}")
            return pd.DataFrame()

    def perform_statistical_test(self, category: str, metric: str) -> Dict:
        """Perform statistical test to compare groups within a category"""
        if (self.aggregated_data.empty or
                category not in self.aggregated_data.columns or
                metric not in self.aggregated_data.columns):
            return {'error': 'Invalid category or metric'}

        try:
            categories = self.aggregated_data[category].unique()
            if len(categories) < 2:
                return {'error': 'Need at least 2 groups for comparison'}

            groups = [
                self.aggregated_data[
                    self.aggregated_data[category] == cat
                    ][metric].dropna()
                for cat in categories
            ]

            if any(len(group) == 0 for group in groups):
                return {'error': 'Some groups have no valid data'}

            f_stat, p_value = stats.f_oneway(*groups)
            return {
                'category': category,
                'metric': metric,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except Exception as e:
            return {'error': f'Statistical test failed: {str(e)}'}

        def compare_vehicles(self, metric: str, stat_type: str = 'mean') -> pd.DataFrame:
            """Compare specific metrics across vehicles"""
            summary_col = f"{metric}_{stat_type}"
            vehicle_stats = self._generate_vehicle_summaries()

            if not vehicle_stats.empty and summary_col in vehicle_stats.columns:
                comparison = vehicle_stats[[
                    'Vehicle_ID', 'Manufacturer', 'Model_Name', summary_col
                ]].sort_values(summary_col, ascending=False)

                # Calculate relative differences
                mean_value = comparison[summary_col].mean()
                comparison['Percent_Difference_from_Mean'] = (
                        (comparison[summary_col] - mean_value) / mean_value * 100
                ).round(2)

                return comparison
            return pd.DataFrame()

        def compare_manufacturers(self, metric: str) -> pd.DataFrame:
            """Compare metrics aggregated by manufacturer"""
            vehicle_stats = self._generate_vehicle_summaries()

            if not vehicle_stats.empty:
                stats_to_compute = ['mean', 'median', 'std', 'min', 'max']
                manufacturer_stats = []

                for manufacturer in vehicle_stats['Manufacturer'].unique():
                    mfg_data = vehicle_stats[vehicle_stats['Manufacturer'] == manufacturer]
                    stats = {
                        'Manufacturer': manufacturer,
                        'Vehicle_Count': len(mfg_data)
                    }

                    for stat in stats_to_compute:
                        col = f"{metric}_{stat}"
                        if col in mfg_data.columns:
                            stats[f"{stat}"] = mfg_data[col].mean()

                    manufacturer_stats.append(stats)

                return pd.DataFrame(manufacturer_stats)
            return pd.DataFrame()

        def generate_comparative_visualizations(self, metric: str):
            """Generate visualizations comparing vehicles and manufacturers"""
            vehicle_stats = self._generate_vehicle_summaries()
            visuals = {}

            if not vehicle_stats.empty:
                mean_col = f"{metric}_mean"

                if mean_col in vehicle_stats.columns:
                    # Box plot by manufacturer
                    visuals['manufacturer_distribution'] = px.box(
                        vehicle_stats,
                        x='Manufacturer',
                        y=mean_col,
                        points='all',
                        title=f'{metric} Distribution by Manufacturer'
                    )

                    # Scatter plot with error bars
                    visuals['vehicle_comparison'] = px.scatter(
                        vehicle_stats,
                        x='Vehicle_ID',
                        y=mean_col,
                        error_y=f"{metric}_std",
                        color='Manufacturer',
                        title=f'Vehicle {metric} Comparison'
                    )

                    # Violin plot split by manufacturer
                    visuals['manufacturer_violin'] = px.violin(
                        vehicle_stats,
                        x='Manufacturer',
                        y=mean_col,
                        box=True,
                        points='all',
                        title=f'{metric} Distribution by Manufacturer'
                    )

            return visuals
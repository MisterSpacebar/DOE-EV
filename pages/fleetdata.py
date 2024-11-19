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
        for filename, df in self.csv_files.items():
            try:
                vehicle_id = filename.split('.')[1]
                vehicle_info = self.reference_data[
                    self.reference_data['Vehicle ID'].str.lower() == vehicle_id.lower()
                    ]

                if not vehicle_info.empty:
                    df_copy = df.copy()

                    # Handle time columns first
                    time_columns = ['Driving Time', 'Idling Time']
                    for col in time_columns:
                        if col in df_copy.columns:
                            try:
                                # If it's already a timedelta or can be converted to one
                                if df_copy[col].dtype == 'timedelta64[ns]':
                                    df_copy[col] = df_copy[col].dt.total_seconds() / 3600
                                else:
                                    df_copy[col] = pd.to_timedelta(df_copy[col]).dt.total_seconds() / 3600
                            except Exception as e:
                                print(f"Could not convert {col} to hours for {vehicle_id}: {e}")
                                # If conversion fails, try to extract numeric values if possible
                                try:
                                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                                except:
                                    print(f"Could not extract numeric values from {col}")

                    # Handle other date/time columns
                    date_columns = [col for col in df_copy.columns
                                    if ('time' in col.lower() or 'date' in col.lower())
                                    and col not in time_columns]
                    for col in date_columns:
                        try:
                            df_copy[col] = pd.to_datetime(df_copy[col])
                        except Exception as e:
                            print(f"Could not convert column {col} to datetime: {e}")

                    df_copy['Vehicle_ID'] = vehicle_id

                    # Add metadata columns
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
                        df_copy[col_name] = vehicle_info.iloc[0][ref_col]

                    all_data.append(df_copy)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

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

            battery_chem = self.reference_data['Battery Chemistry'].dropna()
            if not battery_chem.empty:
                summary['Most_Common_Chemistry'] = battery_chem.mode().iloc[0]

        # Try to get date range if available
        if not self.aggregated_data.empty:
            date_cols = [col for col in self.aggregated_data.columns
                         if 'time' in col.lower() or 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                try:
                    dates = pd.to_datetime(self.aggregated_data[date_col])
                    summary['Date_Range'] = f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"
                except Exception as e:
                    print(f"Error calculating date range: {e}")

        return summary

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

    def compare_manufacturers(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compare comprehensive statistics across manufacturers"""
        if self.aggregated_data.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Calculate vehicle-level statistics
        vehicle_stats = []
        for vehicle_id in self.aggregated_data['Vehicle_ID'].unique():
            vehicle_data = self.aggregated_data[self.aggregated_data['Vehicle_ID'] == vehicle_id]

            stats = {
                'Vehicle_ID': vehicle_id,
                'Manufacturer': vehicle_data['Manufacturer'].iloc[0],
                'Model': vehicle_data['Model_Name'].iloc[0]
            }

            # Energy efficiency
            if 'Total Energy Consumption' in vehicle_data.columns and 'Total Distance' in vehicle_data.columns:
                total_distance = vehicle_data['Total Distance'].sum()
                if total_distance > 0:  # Prevent division by zero
                    stats['Energy_per_Mile'] = (
                            vehicle_data['Total Energy Consumption'].sum() /
                            total_distance
                    )

            # Idle time calculation
            if 'Driving Time' in vehicle_data.columns and 'Idling Time' in vehicle_data.columns:
                try:
                    # Convert time columns to hours if they're datetime
                    driving_time = vehicle_data['Driving Time']
                    idling_time = vehicle_data['Idling Time']

                    # Convert if datetime
                    if driving_time.dtype == 'datetime64[ns]':
                        driving_time = pd.to_timedelta(driving_time).dt.total_seconds() / 3600
                    if idling_time.dtype == 'datetime64[ns]':
                        idling_time = pd.to_timedelta(idling_time).dt.total_seconds() / 3600

                    # Calculate total time and idle percentage
                    total_time = driving_time.sum() + idling_time.sum()
                    if total_time > 0:  # Prevent division by zero
                        stats['Idle_Percentage'] = (idling_time.sum() / total_time * 100)
                    else:
                        stats['Idle_Percentage'] = 0
                except Exception as e:
                    print(f"Error calculating idle time for vehicle {vehicle_id}: {e}")
                    stats['Idle_Percentage'] = 0

            vehicle_stats.append(stats)

        # Create vehicle stats DataFrame
        vehicle_stats_df = pd.DataFrame(vehicle_stats)

        # Calculate manufacturer summary only if we have valid data
        if not vehicle_stats_df.empty and 'Manufacturer' in vehicle_stats_df.columns:
            # Get numeric columns for aggregation
            numeric_cols = vehicle_stats_df.select_dtypes(include=['float64', 'int64']).columns
            agg_columns = {}

            if 'Vehicle_ID' in vehicle_stats_df.columns:
                agg_columns['Vehicle_ID'] = 'count'
            if 'Energy_per_Mile' in numeric_cols:
                agg_columns['Energy_per_Mile'] = ['mean', 'std', 'min', 'max']
            if 'Idle_Percentage' in numeric_cols:
                agg_columns['Idle_Percentage'] = ['mean', 'std', 'min', 'max']

            manufacturer_summary = vehicle_stats_df.groupby('Manufacturer').agg(agg_columns).round(2)

            # Flatten column names if we have multi-level columns
            if isinstance(manufacturer_summary.columns, pd.MultiIndex):
                manufacturer_summary.columns = [
                    'Vehicle_Count' if col == ('Vehicle_ID', 'count')
                    else f"{col[0]}_{col[1].capitalize()}"
                    for col in manufacturer_summary.columns
                ]
        else:
            manufacturer_summary = pd.DataFrame()

        return vehicle_stats_df, manufacturer_summary

    def generate_statistical_summary(self) -> Dict:
        """Generate comprehensive statistical summary"""
        if self.aggregated_data.empty:
            return {'error': 'No data available'}

        stats = {
            'fleet_summary': {
                'total_vehicles': len(self.aggregated_data['Vehicle_ID'].unique()),
                'total_trips': len(self.aggregated_data),
                'total_distance': self.aggregated_data['Total Distance'].sum()
                if 'Total Distance' in self.aggregated_data.columns else 0,
                'avg_trip_distance': self.aggregated_data['Total Distance'].mean()
                if 'Total Distance' in self.aggregated_data.columns else 0,
                'total_energy': self.aggregated_data['Total Energy Consumption'].sum()
                if 'Total Energy Consumption' in self.aggregated_data.columns else 0
            }
        }

        # Calculate percentiles for key metrics
        key_metrics = ['Total Distance', 'Total Energy Consumption', 'SOC Used']
        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        stats['percentiles'] = {}
        for metric in key_metrics:
            if metric in self.aggregated_data.columns:
                stats['percentiles'][metric] = {
                    f'p{int(p * 100)}': self.aggregated_data[metric].quantile(p)
                    for p in percentiles
                }

        return stats

    def generate_comparative_visualizations(self):
        """Generate comparative visualizations"""
        if self.aggregated_data.empty:
            return {}

        visuals = {}
        aggregated_data = self.aggregated_data.copy()

        # Convert datetime columns to numeric hours if needed
        time_columns = ['Driving Time', 'Idling Time']
        for col in time_columns:
            if col in aggregated_data.columns and aggregated_data[col].dtype == 'datetime64[ns]':
                try:
                    aggregated_data[col] = pd.to_timedelta(aggregated_data[col]).dt.total_seconds() / 3600
                except Exception as e:
                    print(f"Error converting {col} to hours: {e}")
                    continue

        # Energy efficiency by manufacturer
        if all(col in aggregated_data.columns for col in
               ['Manufacturer', 'Total Energy Consumption', 'Total Distance']):
            try:
                efficiency_data = (
                    aggregated_data.groupby(['Manufacturer', 'Vehicle_ID'])
                    .agg({
                        'Total Energy Consumption': 'sum',
                        'Total Distance': 'sum'
                    })
                    .reset_index()
                )
                efficiency_data['Energy_per_Mile'] = (
                        efficiency_data['Total Energy Consumption'] /
                        efficiency_data['Total Distance']
                )

                visuals['energy_efficiency'] = px.box(
                    efficiency_data,
                    x='Manufacturer',
                    y='Energy_per_Mile',
                    title='Energy Efficiency by Manufacturer'
                )
            except Exception as e:
                print(f"Error generating energy efficiency visualization: {e}")

        # Trip distance distribution
        if 'Total Distance' in aggregated_data.columns:
            try:
                visuals['trip_distance'] = px.violin(
                    aggregated_data,
                    x='Manufacturer',
                    y='Total Distance',
                    title='Trip Distance Distribution by Manufacturer',
                    box=True
                )
            except Exception as e:
                print(f"Error generating trip distance visualization: {e}")

        # Idle time comparison
        if all(col in aggregated_data.columns for col in ['Driving Time', 'Idling Time', 'Manufacturer']):
            try:
                idle_data = (
                    aggregated_data.groupby(['Manufacturer', 'Vehicle_ID'])
                    .agg({
                        'Driving Time': 'sum',
                        'Idling Time': 'sum'
                    })
                    .reset_index()
                )
                idle_data['Idle_Percentage'] = (
                        (idle_data['Idling Time'] / (idle_data['Driving Time'] + idle_data['Idling Time'])) * 100
                )

                visuals['idle_comparison'] = px.box(
                    idle_data,
                    x='Manufacturer',
                    y='Idle_Percentage',
                    title='Idle Time Percentage by Manufacturer'
                )
            except Exception as e:
                print(f"Error generating idle time visualization: {e}")

        # Energy vs Distance scatter
        if all(col in aggregated_data.columns for col in
               ['Total Energy Consumption', 'Total Distance', 'Average Ambient Temperature']):
            try:
                visuals['energy_distance_scatter'] = px.scatter(
                    aggregated_data,
                    x='Total Distance',
                    y='Total Energy Consumption',
                    color='Average Ambient Temperature',
                    title='Energy Consumption vs Distance (colored by Temperature)',
                    trendline="ols"
                )
            except Exception as e:
                print(f"Error generating energy-distance scatter plot: {e}")

        return visuals
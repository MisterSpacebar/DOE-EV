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

    def analyze_by_manufacturer_and_weight(self) -> Dict:
        """Enhanced analysis by manufacturer and weight class"""
        print("\nStarting manufacturer and weight class analysis...")

        try:
            if self.aggregated_data is None or self.aggregated_data.empty:
                print("No data available for analysis")
                return {}

            results = {}

            def safe_numeric_conversion(df: pd.DataFrame, column: str) -> pd.Series:
                """Safely convert a column to numeric values"""
                try:
                    return pd.to_numeric(df[column], errors='coerce')
                except Exception as e:
                    print(f"Error converting {column}: {e}")
                    return pd.Series([0] * len(df))

            def calculate_category_metrics(category_col: str) -> Optional[pd.DataFrame]:
                """Inner function to calculate metrics for a category"""
                if category_col not in self.aggregated_data.columns:
                    print(f"Column {category_col} not found in data")
                    return None

                try:
                    df = self.aggregated_data.copy()
                    metrics = {}

                    # Group by category
                    grouped = df.groupby(category_col)

                    # Vehicle count
                    metrics['Total_Vehicles'] = grouped['Vehicle_ID'].nunique()

                    # Distance metrics
                    if 'Total Distance' in df.columns:
                        df['Total Distance'] = safe_numeric_conversion(df, 'Total Distance')
                        metrics['Total_Distance'] = grouped['Total Distance'].sum()
                        metrics['Avg_Distance_Per_Trip'] = grouped['Total Distance'].mean()

                    # Energy metrics
                    if 'Total Energy Consumption' in df.columns:
                        df['Total Energy Consumption'] = safe_numeric_conversion(df, 'Total Energy Consumption')
                        metrics['Total_Energy'] = grouped['Total Energy Consumption'].sum()

                        # Calculate efficiency
                        energy_sums = grouped['Total Energy Consumption'].sum()
                        distance_sums = grouped['Total Distance'].sum()
                        metrics['Energy_Efficiency'] = (
                            energy_sums.div(distance_sums)
                            .where(distance_sums > 0, 0)
                            .round(3)
                        )

                    # Time metrics
                    time_cols = ['Driving Time', 'Idling Time']
                    if all(col in df.columns for col in time_cols):
                        for col in time_cols:
                            df[col] = safe_numeric_conversion(df, col)

                        metrics['Driving_Hours'] = grouped['Driving Time'].sum()
                        metrics['Idling_Hours'] = grouped['Idling Time'].sum()
                        metrics['Total_Operating_Hours'] = metrics['Driving_Hours'] + metrics['Idling_Hours']

                        # Calculate idle percentage
                        metrics['Idle_Percentage'] = (
                            (metrics['Idling_Hours'] / metrics['Total_Operating_Hours'] * 100)
                            .where(metrics['Total_Operating_Hours'] > 0, 0)
                            .round(2)
                        )

                    # Temperature metrics
                    if 'Average Ambient Temperature' in df.columns:
                        df['Average Ambient Temperature'] = safe_numeric_conversion(df, 'Average Ambient Temperature')
                        temp_stats = grouped['Average Ambient Temperature'].agg(['mean', 'min', 'max'])
                        metrics['Avg_Temperature'] = temp_stats['mean'].round(2)
                        metrics['Min_Temperature'] = temp_stats['min'].round(2)
                        metrics['Max_Temperature'] = temp_stats['max'].round(2)

                    # Convert all metrics to DataFrame
                    metrics_df = pd.DataFrame(metrics)

                    # Handle any NaN values
                    metrics_df = metrics_df.fillna(0)

                    return metrics_df.round(2)

                except Exception as e:
                    print(f"Error calculating metrics for {category_col}: {e}")
                    return None

            # Calculate metrics for manufacturers
            print("Calculating manufacturer metrics...")
            manufacturer_metrics = calculate_category_metrics('Manufacturer')
            if manufacturer_metrics is not None:
                results['manufacturer'] = manufacturer_metrics
                print("Manufacturer metrics calculated successfully")

            # Calculate metrics for weight classes
            print("Calculating weight class metrics...")
            weight_metrics = calculate_category_metrics('Weight_Class')
            if weight_metrics is not None:
                results['weight_class'] = weight_metrics
                print("Weight class metrics calculated successfully")

            return results

        except Exception as e:
            print(f"Critical error in analyze_by_manufacturer_and_weight: {e}")
            return {}

    def generate_category_visualizations(self, category: str):
        """Enhanced visualizations for manufacturer or weight class analysis"""
        if self.aggregated_data.empty or category not in ['Manufacturer', 'Weight_Class']:
            return {}

        visuals = {}
        df = self.aggregated_data.copy()

        # Convert numeric columns
        numeric_cols = ['Total Energy Consumption', 'Total Distance', 'Average Ambient Temperature',
                        'Driving Time', 'Idling Time']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        try:
            # Calculate efficiency metrics by group
            efficiency_data = []
            for name, group in df.groupby([category, 'Vehicle_ID']):
                try:
                    energy_sum = group['Total Energy Consumption'].sum()
                    distance_sum = group['Total Distance'].sum()
                    temp_mean = group['Average Ambient Temperature'].mean()

                    # Only add if we have valid energy and distance values
                    if energy_sum > 0 and distance_sum > 0:
                        efficiency_data.append({
                            category: name[0],
                            'Vehicle_ID': name[1],
                            'Total Energy Consumption': energy_sum,
                            'Total Distance': distance_sum,
                            'Average Ambient Temperature': temp_mean,
                            'Energy_per_Mile': energy_sum / distance_sum,
                            'Miles_per_kWh': distance_sum / energy_sum
                        })
                except Exception as e:
                    print(f"Error processing group {name}: {e}")
                    continue

            if efficiency_data:
                efficiency_df = pd.DataFrame(efficiency_data)
                if not efficiency_df.empty:
                    # 1. Energy Efficiency vs Temperature scatter plot
                    visuals['efficiency_temp'] = px.scatter(
                        efficiency_df,
                        x='Average Ambient Temperature',
                        y='Energy_per_Mile',
                        color=category,
                        title=f'Energy Efficiency vs Temperature by {category}',
                        trendline="ols"
                    )

                    # 2. Average Miles per kWh bar chart
                    avg_efficiency = efficiency_df.groupby(category)['Miles_per_kWh'].mean().reset_index()
                    avg_efficiency = avg_efficiency.sort_values('Miles_per_kWh', ascending=False)

                    # Calculate max value and add 10% padding
                    max_value = avg_efficiency['Miles_per_kWh'].max()
                    y_max = max_value * 1.1  # Add 10% padding

                    visuals['temperature_impact'] = px.bar(
                        avg_efficiency,
                        x=category,
                        y='Miles_per_kWh',
                        title=f'Average Energy Efficiency (mi/kWh) by {category}'
                    ).update_layout(
                        yaxis_title='Miles per kWh',
                        xaxis_title=category,
                        xaxis_tickangle=45 if len(efficiency_df[category].unique()) > 6 else 0,
                        showlegend=False,
                        yaxis_range=[0, y_max]  # Set y-axis range from 0 to max+10%
                    ).update_traces(
                        text=avg_efficiency['Miles_per_kWh'].round(2),
                        textposition='outside'
                    )

            # Rest of the visualizations remain the same...
            # 3. Operational Patterns
            operational_data = []
            for cat, group in df.groupby(category):
                try:
                    operational_data.append({
                        category: cat,
                        'Driving Time': group['Driving Time'].sum(),
                        'Idling Time': group['Idling Time'].sum(),
                        'Total Distance': group['Total Distance'].sum()
                    })
                except Exception as e:
                    print(f"Error processing operational data for {cat}: {e}")
                    continue

            if operational_data:
                operational_df = pd.DataFrame(operational_data)
                if not operational_df.empty:
                    visuals['operation_patterns'] = px.bar(
                        operational_df,
                        x=category,
                        y=['Driving Time', 'Idling Time'],
                        title=f'Operational Time Distribution by {category}',
                        barmode='stack'
                    )

            # 4. Performance Matrix
            if efficiency_data:
                efficiency_df = pd.DataFrame(efficiency_data)
                if not efficiency_df.empty:
                    performance_stats = efficiency_df.groupby(category).agg({
                        'Total Distance': 'sum',
                        'Total Energy Consumption': 'sum',
                        'Miles_per_kWh': 'mean'
                    }).reset_index()

                    visuals['performance_matrix'] = px.scatter(
                        performance_stats,
                        x='Total Distance',
                        y='Total Energy Consumption',
                        size='Miles_per_kWh',
                        size_max=50,
                        color=category,
                        title=f'Performance Matrix by {category}',
                        labels={
                            'Total Distance': 'Total Distance (miles)',
                            'Total Energy Consumption': 'Total Energy Consumption (kWh)',
                            'Miles_per_kWh': 'Average Miles per kWh'
                        }
                    )

        except Exception as e:
            print(f"Error generating visualizations: {e}")

        return visuals

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
        try:
            # Validate input data
            if self.aggregated_data is None or self.aggregated_data.empty:
                print("No aggregated data available")
                return pd.DataFrame(), pd.DataFrame()

            # Ensure required columns exist
            required_cols = ['Vehicle_ID', 'Manufacturer', 'Model_Name']
            if not all(col in self.aggregated_data.columns for col in required_cols):
                print(f"Missing required columns. Available columns: {self.aggregated_data.columns.tolist()}")
                return pd.DataFrame(), pd.DataFrame()

            # Calculate vehicle-level statistics
            vehicle_stats = []

            # Group by Vehicle_ID first to avoid duplicate processing
            vehicle_groups = self.aggregated_data.groupby('Vehicle_ID')

            for vehicle_id, vehicle_data in vehicle_groups:
                try:
                    # Basic vehicle info with safe access
                    stats = {
                        'Vehicle_ID': vehicle_id,
                        'Manufacturer': vehicle_data['Manufacturer'].iloc[0],
                        'Model': vehicle_data['Model_Name'].iloc[0]
                    }

                    # Energy efficiency calculation
                    if all(col in vehicle_data.columns for col in ['Total Energy Consumption', 'Total Distance']):
                        try:
                            energy = pd.to_numeric(vehicle_data['Total Energy Consumption'], errors='coerce').sum()
                            distance = pd.to_numeric(vehicle_data['Total Distance'], errors='coerce').sum()
                            stats['Energy_per_Mile'] = energy / distance if distance > 0 else 0
                        except Exception as e:
                            print(f"Energy calculation error for {vehicle_id}: {e}")
                            stats['Energy_per_Mile'] = 0

                    # Time calculations
                    if all(col in vehicle_data.columns for col in ['Driving Time', 'Idling Time']):
                        try:
                            # Convert time data to numeric hours if needed
                            driving_time = vehicle_data['Driving Time']
                            idling_time = vehicle_data['Idling Time']

                            # Handle different time formats
                            for time_series in [driving_time, idling_time]:
                                if time_series.dtype == 'datetime64[ns]':
                                    time_series = pd.to_timedelta(time_series).dt.total_seconds() / 3600
                                elif time_series.dtype == 'object':
                                    time_series = pd.to_numeric(time_series, errors='coerce')

                            total_time = driving_time.sum() + idling_time.sum()
                            stats['Idle_Percentage'] = (idling_time.sum() / total_time * 100) if total_time > 0 else 0
                        except Exception as e:
                            print(f"Time calculation error for {vehicle_id}: {e}")
                            stats['Idle_Percentage'] = 0

                    vehicle_stats.append(stats)
                except Exception as e:
                    print(f"Error processing vehicle {vehicle_id}: {e}")
                    continue

            # Create vehicle stats DataFrame
            if not vehicle_stats:
                print("No vehicle statistics were generated")
                return pd.DataFrame(), pd.DataFrame()

            vehicle_stats_df = pd.DataFrame(vehicle_stats)

            # Calculate manufacturer summary
            try:
                if not vehicle_stats_df.empty and 'Manufacturer' in vehicle_stats_df.columns:
                    agg_dict = {'Vehicle_ID': 'count'}

                    if 'Energy_per_Mile' in vehicle_stats_df.columns:
                        agg_dict['Energy_per_Mile'] = ['mean', 'std', 'min', 'max']
                    if 'Idle_Percentage' in vehicle_stats_df.columns:
                        agg_dict['Idle_Percentage'] = ['mean', 'std', 'min', 'max']

                    manufacturer_summary = vehicle_stats_df.groupby('Manufacturer').agg(agg_dict).round(2)

                    # Flatten column names
                    if isinstance(manufacturer_summary.columns, pd.MultiIndex):
                        manufacturer_summary.columns = [
                            'Vehicle_Count' if col == ('Vehicle_ID', 'count')
                            else f"{col[0]}_{col[1].capitalize()}"
                            for col in manufacturer_summary.columns
                        ]

                    return vehicle_stats_df, manufacturer_summary
                else:
                    print("Cannot create manufacturer summary - missing required data")
                    return vehicle_stats_df, pd.DataFrame()

            except Exception as e:
                print(f"Error creating manufacturer summary: {e}")
                return vehicle_stats_df, pd.DataFrame()

        except Exception as e:
            print(f"Critical error in compare_manufacturers: {e}")
            return pd.DataFrame(), pd.DataFrame()

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
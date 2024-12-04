import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go


class CategoryDataAnalyzer:
    def __init__(self, csv_files: Dict[str, pd.DataFrame], reference_data: List[dict]):
        self.csv_files = csv_files
        self.reference_data = pd.DataFrame(reference_data)
        self.file_metadata = self.parse_filenames()
        self.aggregated_data = self._aggregate_data()

    def parse_filenames(self) -> pd.DataFrame:
        """Parse metadata from filenames into a DataFrame"""
        metadata = []

        for filename in self.csv_files.keys():
            try:
                parts = filename.split('.')
                if len(parts) >= 6:
                    metadata.append({
                        'filename': filename,
                        'vehicle_id': parts[1],
                        'fleet_id': parts[2],
                        'model_year': parts[3],
                        'manufacturer': parts[4],
                        'weight_class': parts[5]
                    })
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
                continue

        return pd.DataFrame(metadata)

    def get_unique_categories(self) -> Dict[str, List[str]]:
        """Get unique manufacturers and weight classes from filenames"""
        if self.file_metadata.empty:
            return {'manufacturers': [], 'weight_classes': []}

        return {
            'manufacturers': sorted(self.file_metadata['manufacturer'].unique()),
            'weight_classes': sorted(self.file_metadata['weight_class'].unique())
        }

    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert common numeric columns to proper numeric type"""
        numeric_columns = {
            'Total Distance': 2,  # number of decimal places
            'Total Energy Consumption': 2,
            'Average Ambient Temperature': 2,
            'SOC Used': 1,
            'Initial SOC': 1,
            'Final SOC': 1,
            'Percent Idling Time': 0
        }

        df_copy = df.copy()
        for col, decimals in numeric_columns.items():
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').round(decimals)

        return df_copy

    def _aggregate_data(self) -> pd.DataFrame:
        """Combine all vehicle data into a single DataFrame with metadata"""
        all_data = []

        for filename, df in self.csv_files.items():
            try:
                file_info = self.file_metadata[self.file_metadata['filename'] == filename].iloc[0]
                df_copy = df.copy()
                df_copy = df_copy[df_copy['Total Distance'] > 0]

                if df_copy.empty:
                    continue

                # Calculate trip time with fallback to Total Run Time
                try:
                    df_copy['Local Trip Start Time'] = pd.to_datetime(df_copy['Local Trip Start Time'])
                    df_copy['Local Trip End Time'] = pd.to_datetime(df_copy['Local Trip End Time'])
                    df_copy['Total Trip Time'] = (df_copy['Local Trip End Time'] - df_copy[
                        'Local Trip Start Time']).dt.total_seconds() / 3600
                    # If calculation results in NaT or negative values, use Total Run Time
                    invalid_times = df_copy['Total Trip Time'].isna() | (df_copy['Total Trip Time'] <= 0)
                    df_copy.loc[invalid_times, 'Total Trip Time'] = df_copy.loc[invalid_times, 'Total Run Time']
                except Exception as e:
                    print(f"Using Total Run Time for {filename}: {e}")
                    df_copy['Total Trip Time'] = df_copy['Total Run Time']

                # Calculate average speed using the determined total time
                df_copy['Average Speed'] = df_copy.apply(
                    lambda row: row['Total Distance'] / row['Total Trip Time']
                    if row['Total Trip Time'] > 0 else 0,
                    axis=1
                ).round(2)

                # Handle other time columns
                time_columns = ['Driving Time', 'Idling Time', 'Total Trip Time']
                for col in time_columns:
                    if col in df_copy.columns:
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').round(4)

                # Add metadata
                df_copy['Vehicle_ID'] = file_info['vehicle_id']
                df_copy['Manufacturer'] = file_info['manufacturer']
                df_copy['Weight_Class'] = file_info['weight_class']
                df_copy['Model_Year'] = file_info['model_year']
                df_copy['Fleet_ID'] = file_info['fleet_id']

                all_data.append(df_copy)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def get_category_summary(self, manufacturer: Optional[str] = None, weight_class: Optional[str] = None) -> Dict:
        """Generate summary statistics for filtered data"""
        if self.aggregated_data.empty:
            return {}

        filtered_data = self.aggregated_data.copy()
        if manufacturer:
            filtered_data = filtered_data[
                filtered_data['Manufacturer'].str.lower() == manufacturer.lower()
                ]
        if weight_class:
            filtered_data = filtered_data[
                filtered_data['Weight_Class'] == weight_class
                ]

        filtered_data = self._convert_numeric_columns(filtered_data)

        summary = {
            'total_vehicles': len(filtered_data['Vehicle_ID'].unique()),
            'total_manufacturers': len(filtered_data['Manufacturer'].unique()),
            'total_weight_classes': len(filtered_data['Weight_Class'].unique()),
            'total_distance': filtered_data['Total Distance'].sum() if 'Total Distance' in filtered_data.columns else 0,
            'total_energy': filtered_data[
                'Total Energy Consumption'].sum() if 'Total Energy Consumption' in filtered_data.columns else 0,
            'avg_temperature': filtered_data[
                'Average Ambient Temperature'].mean() if 'Average Ambient Temperature' in filtered_data.columns else None
        }

        # Convert any numpy types to Python native types for JSON serialization
        for key, value in summary.items():
            if isinstance(value, (np.integer, np.floating)):
                summary[key] = float(value)

        return summary

    def calculate_performance_metrics(self, manufacturer: Optional[str] = None,
                                      weight_class: Optional[str] = None) -> pd.DataFrame:
        """Calculate detailed performance metrics for filtered data"""
        if self.aggregated_data.empty:
            return pd.DataFrame()

        filtered_data = self.aggregated_data.copy()
        if manufacturer:
            filtered_data = filtered_data[
                filtered_data['Manufacturer'].str.lower() == manufacturer.lower()
                ]
        if weight_class:
            filtered_data = filtered_data[
                filtered_data['Weight_Class'] == weight_class
                ]

        metrics = []
        for vehicle_id in filtered_data['Vehicle_ID'].unique():
            vehicle_data = filtered_data[filtered_data['Vehicle_ID'] == vehicle_id]
            valid_time_data = vehicle_data.dropna(subset=['Driving Time', 'Idling Time', 'Total Trip Time'])

            metric = {
                'Vehicle_ID': vehicle_id,
                'Manufacturer': vehicle_data['Manufacturer'].iloc[0],
                'Weight_Class': vehicle_data['Weight_Class'].iloc[0],
                'Model_Year': vehicle_data['Model_Year'].iloc[0],
                'Total_Trips': len(vehicle_data),
                'Total_Distance': vehicle_data['Total Distance'].sum(),
                'Total_Energy': vehicle_data[
                    'Total Energy Consumption'].sum() if 'Total Energy Consumption' in vehicle_data.columns else 0,
                'Total_Trip_Hours': round(vehicle_data['Total Trip Time'].sum(), 4),
                'Average_Speed': round(
                    vehicle_data['Total Distance'].sum() / vehicle_data['Total Trip Time'].sum(), 2
                ) if vehicle_data['Total Trip Time'].sum() > 0 else 0
            }

            # Calculate efficiency
            if metric['Total_Distance'] > 0 and metric['Total_Energy'] > 0:
                metric['Energy_Efficiency'] = metric['Total_Energy'] / metric['Total_Distance']
            else:
                metric['Energy_Efficiency'] = 0

            # Calculate time metrics
            if not valid_time_data.empty:
                metric['Total_Driving_Hours'] = round(valid_time_data['Driving Time'].sum(), 4)
                metric['Total_Idle_Hours'] = round(valid_time_data['Idling Time'].sum(), 4)
                total_hours = metric['Total_Driving_Hours'] + metric['Total_Idle_Hours']
                metric['Idle_Percentage'] = round((metric['Total_Idle_Hours'] / total_hours * 100),
                                                  2) if total_hours > 0 else 0

            # Calculate temperature metrics
            if 'Average Ambient Temperature' in vehicle_data.columns:
                # Convert to numeric, coerce errors to NaN
                temps = pd.to_numeric(vehicle_data['Average Ambient Temperature'], errors='coerce')
                valid_temps = temps.dropna()
                if not valid_temps.empty:
                    metric['Avg_Temperature'] = round(valid_temps.mean(), 2)
                    metric['Min_Temperature'] = round(valid_temps.min(), 2)
                    metric['Max_Temperature'] = round(valid_temps.max(), 2)

            metrics.append(metric)

        return pd.DataFrame(metrics)

    def calculate_detailed_stats(self, manufacturer: Optional[str] = None,
                                 weight_class: Optional[str] = None) -> pd.DataFrame:
        """Calculate comprehensive statistics for the selected category"""
        if self.aggregated_data.empty:
            return pd.DataFrame()

        filtered_data = self.aggregated_data.copy()
        if manufacturer:
            filtered_data = filtered_data[
                filtered_data['Manufacturer'].str.lower() == manufacturer.lower()
                ]
        if weight_class:
            filtered_data = filtered_data[
                filtered_data['Weight_Class'] == weight_class
                ]

        # Group metrics by vehicle
        vehicle_metrics = []
        for vehicle_id in filtered_data['Vehicle_ID'].unique():
            vehicle_data = filtered_data[filtered_data['Vehicle_ID'] == vehicle_id]

            try:
                total_distance = vehicle_data['Total Distance'].sum()
                total_time = vehicle_data['Total Trip Time'].sum()

                metrics = {
                    'Total Trips': len(vehicle_data),
                    'Total Distance (mi)': total_distance,
                    'Avg Trip Distance (mi)': vehicle_data['Total Distance'].mean(),
                    'Total Energy (kWh)': vehicle_data['Total Energy Consumption'].sum(),
                    'Avg Energy per Trip (kWh)': vehicle_data['Total Energy Consumption'].mean(),
                    'Energy Efficiency (kWh/mi)': (vehicle_data['Total Energy Consumption'].sum() /
                                                   total_distance) if total_distance > 0 else 0,
                    'Average Speed (mph)': (total_distance / total_time) if total_time > 0 else 0,
                    'Avg Temperature (°F)': vehicle_data['Average Ambient Temperature'].mean(),
                    'Min Temperature (°F)': vehicle_data['Average Ambient Temperature'].min(),
                    'Max Temperature (°F)': vehicle_data['Average Ambient Temperature'].max(),
                    'Total Drive Time (hrs)': vehicle_data['Driving Time'].sum(),
                    'Total Idle Time (hrs)': vehicle_data['Idling Time'].sum(),
                    'Avg Idle Time (%)': vehicle_data['Percent Idling Time'].mean(),
                    'Avg Initial SOC (%)': vehicle_data[
                        'Initial SOC'].mean() if 'Initial SOC' in vehicle_data.columns else None,
                    'Avg Final SOC (%)': vehicle_data[
                        'Final SOC'].mean() if 'Final SOC' in vehicle_data.columns else None,
                    'Avg SOC Used (%)': vehicle_data['SOC Used'].mean() if 'SOC Used' in vehicle_data.columns else None
                }

                # Remove None values
                metrics = {k: v for k, v in metrics.items() if v is not None}
                vehicle_metrics.append(metrics)
            except Exception as e:
                print(f"Error calculating metrics for vehicle {vehicle_id}: {e}")
                continue

        if not vehicle_metrics:
            return pd.DataFrame()

        # Convert to DataFrame and calculate aggregate statistics
        metrics_df = pd.DataFrame(vehicle_metrics)

        # Calculate aggregate statistics only for numeric columns
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        agg_stats = pd.DataFrame({
            'Minimum': metrics_df[numeric_cols].min(),
            'Maximum': metrics_df[numeric_cols].max(),
            'Mean': metrics_df[numeric_cols].mean(),
            'Median': metrics_df[numeric_cols].median(),
            'Std Dev': metrics_df[numeric_cols].std()
        }).round(2)

        return agg_stats

    def generate_stats_visualizations(self, stats_df: pd.DataFrame) -> Dict:
        """Generate visualizations for comprehensive statistics"""
        visuals = {}

        # Only consider numeric columns that exist in the stats DataFrame
        available_metrics = [
            'Energy Efficiency (kWh/mi)',
            'Avg Trip Distance (mi)',
            'Avg Temperature (°F)',
            'Avg Idle Time (%)'
        ]

        key_metrics = [metric for metric in available_metrics if metric in stats_df.index]

        for metric in key_metrics:
            try:
                data = pd.DataFrame({
                    'Metric': [metric] * 5,
                    'Statistic': ['Minimum', 'Maximum', 'Mean', 'Median', 'Std Dev'],
                    'Value': stats_df.loc[metric]
                })

                visuals[f'{metric.lower().replace(" ", "_").replace("(", "").replace(")", "")}'] = px.bar(
                    data,
                    x='Statistic',
                    y='Value',
                    title=f'{metric} Statistics',
                    labels={'Value': metric}
                )
            except Exception as e:
                print(f"Error creating visualization for {metric}: {e}")
                continue

        return visuals

    def generate_visualizations(self, manufacturer: Optional[str] = None, weight_class: Optional[str] = None) -> Dict:
        """Generate visualizations for filtered data"""
        if self.aggregated_data.empty:
            return {}

        filtered_data = self.aggregated_data.copy()
        if manufacturer:
            filtered_data = filtered_data[
                filtered_data['Manufacturer'].str.lower() == manufacturer.lower()
                ]
        if weight_class:
            filtered_data = filtered_data[
                filtered_data['Weight_Class'] == weight_class
                ]

        filtered_data = self._convert_numeric_columns(filtered_data)
        visuals = {}

        # Energy Efficiency by Vehicle
        metrics_df = self.calculate_performance_metrics(manufacturer, weight_class)
        if not metrics_df.empty and 'Energy_Efficiency' in metrics_df.columns:
            visuals['efficiency'] = px.bar(
                metrics_df,
                x='Vehicle_ID',
                y='Energy_Efficiency',
                color='Manufacturer',
                title='Energy Efficiency by Vehicle',
                labels={'Energy_Efficiency': 'Energy per Mile (kWh/mi)'}
            )

        # Distance vs Energy Scatter
        if all(col in filtered_data.columns for col in ['Total Distance', 'Total Energy Consumption']):
            visuals['energy_distance'] = px.scatter(
                filtered_data,
                x='Total Distance',
                y='Total Energy Consumption',
                color='Manufacturer' if not manufacturer else 'Weight_Class',
                title='Energy Consumption vs Distance',
                trendline='ols'
            )

        # Temperature Impact
        if 'Average Ambient Temperature' in filtered_data.columns:
            visuals['temperature'] = px.scatter(
                filtered_data,
                x='Average Ambient Temperature',
                y='Total Energy Consumption',
                color='Manufacturer' if not manufacturer else 'Weight_Class',
                title='Temperature Impact on Energy Consumption',
                trendline='ols'
            )

        # Idle Time Analysis
        if 'Percent Idling Time' in filtered_data.columns:
            visuals['idle_time'] = px.box(
                filtered_data,
                x='Manufacturer' if not manufacturer else 'Weight_Class',
                y='Percent Idling Time',
                title='Idle Time Distribution'
            )

        return visuals

    def calculate_statistical_summary(self, manufacturer: Optional[str] = None,
                                      weight_class: Optional[str] = None) -> Dict:
        if self.aggregated_data.empty:
            return {}

        filtered_data = self.aggregated_data.copy()
        if manufacturer:
            filtered_data = filtered_data[
                filtered_data['Manufacturer'].str.lower() == manufacturer.lower()
                ]
        if weight_class:
            filtered_data = filtered_data[
                filtered_data['Weight_Class'] == weight_class
                ]

        # Add Average Speed to the statistics
        filtered_data['Average Speed'] = filtered_data.apply(
            lambda row: row['Total Distance'] / row['Total Trip Time']
            if row['Total Trip Time'] > 0 else 0,
            axis=1
        )

        numeric_cols = [
            'Total Distance', 'Total Energy Consumption',
            'Average Ambient Temperature', 'Total Trip Time',
            'Average Speed', 'Driving Time', 'Idling Time'
        ]
        basic_stats_data = {}

        for col in numeric_cols:
            if col in filtered_data.columns:
                valid_data = pd.to_numeric(filtered_data[col], errors='coerce').dropna()
                if not valid_data.empty:
                    basic_stats_data[col] = {
                        'Count': len(valid_data),
                        'Mean': valid_data.mean(),
                        'Std': valid_data.std(),
                        'Min': valid_data.min(),
                        '25%': valid_data.quantile(0.25),
                        'Median': valid_data.median(),
                        '75%': valid_data.quantile(0.75),
                        'Max': valid_data.max()
                    }

        basic_stats = pd.DataFrame(basic_stats_data)

        round_dict = {
            'Total Distance': 2,
            'Total Energy Consumption': 2,
            'Average Ambient Temperature': 2,
            'Total Trip Time': 4,
            'Driving Time': 4,
            'Idling Time': 4,
            'Average Speed': 2
        }

        for col in basic_stats.columns:
            if col in round_dict:
                basic_stats[col] = basic_stats[col].round(round_dict[col])

        fleet_stats = {
            'total_vehicles': len(filtered_data['Vehicle_ID'].unique()),
            'total_trips': len(filtered_data),
            'total_distance': filtered_data['Total Distance'].sum(),
            'avg_trip_distance': filtered_data['Total Distance'].mean(),
            'total_energy': filtered_data[
                'Total Energy Consumption'].sum() if 'Total Energy Consumption' in filtered_data.columns else 0,
            'avg_energy_per_trip': filtered_data[
                'Total Energy Consumption'].mean() if 'Total Energy Consumption' in filtered_data.columns else 0,
            'avg_speed': filtered_data['Average Speed'].mean()
        }

        return {
            'basic_stats': basic_stats,
            'fleet_stats': fleet_stats
        }

    def perform_category_comparison(self, category: str) -> Dict:
        """Perform statistical tests comparing different categories"""
        if self.aggregated_data.empty or category not in ['Manufacturer', 'Weight_Class']:
            return {}

        comparison_results = {}
        metrics = ['Total Distance', 'Total Energy Consumption', 'Average Ambient Temperature']

        for metric in metrics:
            if metric not in self.aggregated_data.columns:
                continue

            # Group data by category
            groups = []
            labels = []
            for cat_value in self.aggregated_data[category].unique():
                group_data = self.aggregated_data[self.aggregated_data[category] == cat_value][metric].dropna()
                if len(group_data) > 0:
                    groups.append(group_data)
                    labels.append(cat_value)

            if len(groups) < 2:
                continue

            # Perform ANOVA
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                comparison_results[metric] = {
                    'test': 'ANOVA',
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

                # Add effect size (eta-squared)
                total_data = np.concatenate(groups)
                grand_mean = np.mean(total_data)
                ss_total = np.sum((total_data - grand_mean) ** 2)
                ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
                eta_squared = ss_between / ss_total
                comparison_results[metric]['effect_size'] = eta_squared

            except Exception as e:
                print(f"Error performing ANOVA for {metric}: {e}")
                continue

        return comparison_results

    def calculate_correlations(self, manufacturer: Optional[str] = None, weight_class: Optional[str] = None) -> Tuple[
        pd.DataFrame, go.Figure]:
        """Calculate correlation matrix and generate heatmap for numeric variables"""
        if self.aggregated_data.empty:
            return pd.DataFrame(), None

        # Filter data
        filtered_data = self.aggregated_data.copy()
        if manufacturer:
            filtered_data = filtered_data[
                filtered_data['Manufacturer'].str.lower() == manufacturer.lower()
                ]
        if weight_class:
            filtered_data = filtered_data[
                filtered_data['Weight_Class'] == weight_class
                ]

        # Select numeric columns
        numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = filtered_data[numeric_cols].corr().round(3)

        # Create correlation heatmap
        heatmap = px.imshow(
            correlation_matrix,
            title="Correlation Matrix of Key Metrics",
            aspect='auto',
            color_continuous_scale='RdBu_r'
        )

        # Update layout for better readability
        heatmap.update_layout(
            title_x=0.5,
            font=dict(size=10),
            xaxis=dict(tickangle=45)
        )

        # Add value annotations
        heatmap.update_traces(
            text=correlation_matrix.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        )

        return correlation_matrix, heatmap

    def perform_trend_analysis(self, manufacturer: Optional[str] = None, weight_class: Optional[str] = None) -> Dict:
        """Analyze trends over time for key metrics"""
        if self.aggregated_data.empty:
            return {}

        filtered_data = self.aggregated_data.copy()
        if manufacturer:
            filtered_data = filtered_data[
                filtered_data['Manufacturer'].str.lower() == manufacturer.lower()
                ]
        if weight_class:
            filtered_data = filtered_data[
                filtered_data['Weight_Class'] == weight_class
                ]

        # Handle datetime with flexible format
        if 'Local Trip Start Time' in filtered_data.columns:
            try:
                filtered_data['Trip_Date'] = pd.to_datetime(
                    filtered_data['Local Trip Start Time'],
                    format='mixed'
                ).dt.date

                daily_metrics = filtered_data.groupby('Trip_Date').agg({
                    'Total Distance': 'sum',
                    'Total Energy Consumption': 'sum',
                    'Average Ambient Temperature': 'mean',
                    'Vehicle_ID': 'nunique'
                }).reset_index()

                window = 7
                rolling_metrics = daily_metrics.set_index('Trip_Date').rolling(window=window).mean()

                return {
                    'daily_metrics': daily_metrics,
                    'rolling_averages': rolling_metrics,
                    'total_days': len(daily_metrics),
                    'avg_daily_distance': daily_metrics['Total Distance'].mean(),
                    'avg_daily_energy': daily_metrics['Total Energy Consumption'].mean(),
                    'avg_daily_vehicles': daily_metrics['Vehicle_ID'].mean()
                }
            except Exception as e:
                print(f"Error in trend analysis: {e}")
                return {}

        return {}

    def generate_statistical_visualizations(self, manufacturer: Optional[str] = None,
                                            weight_class: Optional[str] = None) -> Dict:
        """Generate visualizations for statistical analysis"""
        visuals = {}

        # Filter data
        filtered_data = self.aggregated_data.copy()
        if manufacturer:
            filtered_data = filtered_data[
                filtered_data['Manufacturer'].str.lower() == manufacturer.lower()
                ]
        if weight_class:
            filtered_data = filtered_data[
                filtered_data['Weight_Class'] == weight_class
                ]

        # Distribution plots for key metrics
        metrics = ['Total Distance', 'Total Energy Consumption', 'Average Ambient Temperature']
        for metric in metrics:
            if metric in filtered_data.columns:
                visuals[f'{metric.lower()}_dist'] = px.histogram(
                    filtered_data,
                    x=metric,
                    title=f'{metric} Distribution',
                    marginal='box'
                )

        # Time series analysis
        trend_data = self.perform_trend_analysis(manufacturer, weight_class)
        if trend_data and 'daily_metrics' in trend_data:
            visuals['time_series'] = px.line(
                trend_data['daily_metrics'],
                x='Trip_Date',
                y=['Total Distance', 'Total Energy Consumption'],
                title='Daily Metrics Over Time'
            )

        # Box plots for category comparisons
        category = 'Manufacturer' if not manufacturer else 'Weight_Class'
        for metric in metrics:
            if metric in filtered_data.columns:
                visuals[f'{metric.lower()}_box'] = px.box(
                    filtered_data,
                    x=category,
                    y=metric,
                    title=f'{metric} by {category}'
                )

        return visuals
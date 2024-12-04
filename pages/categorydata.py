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

    def validate_trip_data(self, row: pd.Series) -> Dict[str, bool]:
        """Validate trip data for consistency and reasonableness"""
        validations = {
            'valid_distance': False,
            'valid_times': False,
            'matching_times': False,
            'reasonable_speed': False
        }

        try:
            # Distance validation (allow smaller distances)
            distance = pd.to_numeric(row['Total Distance'], errors='coerce')
            validations['valid_distance'] = 0 <= distance < 1000

            # Time validation
            drive_time = pd.to_numeric(row['Driving Time'], errors='coerce')
            idle_time = pd.to_numeric(row['Idling Time'], errors='coerce')
            run_time = pd.to_numeric(row['Total Run Time'], errors='coerce')

            validations['valid_times'] = (
                    pd.notnull(drive_time) and drive_time >= 0 and
                    pd.notnull(idle_time) and idle_time >= 0 and
                    pd.notnull(run_time) and run_time >= 0
            )

            # Time components validation (increase tolerance to 5%)
            if validations['valid_times']:
                calculated_total = drive_time + idle_time
                if run_time > 0:
                    time_diff_percent = abs(calculated_total - run_time) / run_time * 100
                    validations['matching_times'] = time_diff_percent <= 5
                else:
                    validations['matching_times'] = calculated_total == 0

            # Speed validation
            if validations['valid_distance'] and validations['valid_times'] and run_time > 0:
                speed = distance / run_time
                validations['reasonable_speed'] = 0 <= speed <= 80

            # Log specific validation failures
            if not all(validations.values()):
                failed_checks = {k: v for k, v in validations.items() if not v}
                print(f"Row validation failed: {failed_checks}")
                print(f"Distance: {distance}, Drive Time: {drive_time}, Idle Time: {idle_time}, Run Time: {run_time}")
                if 'reasonable_speed' in failed_checks and run_time > 0:
                    print(f"Calculated speed: {distance / run_time} mph")

        except Exception as e:
            print(f"Validation error: {e}")

        return validations

    def _aggregate_data(self) -> pd.DataFrame:
        """Combine all vehicle data into a single DataFrame with metadata"""
        all_data = []

        for filename, df in self.csv_files.items():
            try:
                file_info = self.file_metadata[self.file_metadata['filename'] == filename].iloc[0]
                df_copy = df.copy()

                # Convert numeric columns
                numeric_cols = ['Total Distance', 'Total Run Time', 'Driving Time', 'Idling Time',
                                'Total Energy Consumption']
                for col in numeric_cols:
                    if col in df_copy.columns:
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

                # Add Total Trip Time based on Total Run Time
                df_copy['Total Trip Time'] = df_copy['Total Run Time']

                # Filter rows with valid data
                validated_rows = []
                for _, row in df_copy.iterrows():
                    validations = self.validate_trip_data(row)
                    if all(validations.values()):
                        # Ensure distance is reasonable (> 0.5 miles to avoid inflated ratios)
                        if row['Total Distance'] > 0.5:
                            row['Average Speed'] = row['Total Distance'] / row['Total Run Time']
                            validated_rows.append(row)

                if validated_rows:
                    validated_df = pd.DataFrame(validated_rows)

                    # Calculate Energy Efficiency for each row
                    validated_df['Energy_Efficiency'] = validated_df['Total Energy Consumption'] / validated_df[
                        'Total Distance']

                    # Append metadata columns
                    validated_df['Vehicle_ID'] = file_info['vehicle_id']
                    validated_df['Manufacturer'] = file_info['manufacturer']
                    validated_df['Weight_Class'] = file_info['weight_class']
                    validated_df['Model_Year'] = file_info['model_year']
                    validated_df['Fleet_ID'] = file_info['fleet_id']
                    all_data.append(validated_df)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        aggregated_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

        # Filter using z-score or IQR to remove outliers
        if not aggregated_df.empty:
            # Use IQR to remove outliers for Energy Efficiency
            q1 = aggregated_df['Energy_Efficiency'].quantile(0.25)
            q3 = aggregated_df['Energy_Efficiency'].quantile(0.75)
            iqr = q3 - q1

            # Define lower and upper bounds for outlier detection
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Filter out rows that fall outside the bounds
            filtered_df = aggregated_df[
                (aggregated_df['Energy_Efficiency'] >= lower_bound) & (
                            aggregated_df['Energy_Efficiency'] <= upper_bound)
                ]

            print(f"Number of records removed due to outliers: {len(aggregated_df) - len(filtered_df)}")
            aggregated_df = filtered_df

        return aggregated_df

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
        if self.aggregated_data.empty:
            return pd.DataFrame()

        filtered_data = self.aggregated_data.copy()

        # Apply filters
        if manufacturer:
            filtered_data = filtered_data[filtered_data['Manufacturer'].str.lower() == manufacturer.lower()]
        if weight_class:
            filtered_data = filtered_data[filtered_data['Weight_Class'] == weight_class]

        vehicle_metrics = []
        for vehicle_id in filtered_data['Vehicle_ID'].unique():
            vehicle_data = filtered_data[filtered_data['Vehicle_ID'] == vehicle_id]

            try:
                metrics = {
                    'Total Trips': len(vehicle_data),
                    'Total Distance (mi)': vehicle_data['Total Distance'].sum(),
                    'Avg Trip Distance (mi)': vehicle_data['Total Distance'].mean(),
                    'Total Energy (kWh)': vehicle_data['Total Energy Consumption'].sum(),
                    'Avg Energy per Trip (kWh)': vehicle_data['Total Energy Consumption'].mean(),
                    'Energy Efficiency (kWh/mi)': vehicle_data['Total Energy Consumption'].sum() / vehicle_data[
                        'Total Distance'].sum() if vehicle_data['Total Distance'].sum() > 0 else 0,
                    'Average Speed (mph)': vehicle_data['Average Speed'].mean(),
                    'Total Drive Time (hrs)': vehicle_data['Driving Time'].sum(),
                    'Total Idle Time (hrs)': vehicle_data['Idling Time'].sum(),
                    'Total Run Time (hrs)': vehicle_data['Total Trip Time'].sum(),
                    'Avg Temperature (°F)': pd.to_numeric(vehicle_data['Average Ambient Temperature'],
                                                          errors='coerce').mean(),
                    'Min Temperature (°F)': pd.to_numeric(vehicle_data['Average Ambient Temperature'],
                                                          errors='coerce').min(),
                    'Max Temperature (°F)': pd.to_numeric(vehicle_data['Average Ambient Temperature'],
                                                          errors='coerce').max(),
                    'Avg Idle Time (%)': vehicle_data['Percent Idling Time'].mean()
                }

                # Remove None/NaN values
                metrics = {k: v for k, v in metrics.items() if pd.notnull(v)}
                vehicle_metrics.append(metrics)

            except Exception as e:
                print(f"Error calculating metrics for vehicle {vehicle_id}: {e}")
                continue

        if not vehicle_metrics:
            return pd.DataFrame()

        metrics_df = pd.DataFrame(vehicle_metrics)

        # Calculate aggregate statistics
        agg_stats = pd.DataFrame({
            'Minimum': metrics_df.min(),
            'Maximum': metrics_df.max(),
            'Mean': metrics_df.mean(),
            'Median': metrics_df.median(),
            'Std Dev': metrics_df.std()
        }).round(2)

        return agg_stats

    def generate_stats_visualizations(self, stats: Dict) -> Dict:
        """Generate visualizations for comprehensive statistics"""
        visuals = {}
        warnings = []

        # Extract 'basic_stats' from the provided dictionary
        if 'basic_stats' not in stats:
            warnings.append("No basic statistics available for visualization.")
            return {'visuals': visuals, 'warnings': warnings}

        stats_df = stats['basic_stats']

        # Ensure that stats_df is a DataFrame and not empty
        if not isinstance(stats_df, pd.DataFrame) or stats_df.empty:
            warnings.append("The basic stats DataFrame is either missing or empty.")
            return {'visuals': visuals, 'warnings': warnings}

        # Only consider numeric columns that exist in the stats DataFrame
        available_metrics = [
            'Energy Efficiency (kWh/mi)',
            'Avg Trip Distance (mi)',
            'Avg Temperature (°F)',
            'Avg Idle Time (%)'
        ]

        # Correctly access the index of the DataFrame
        key_metrics = [metric for metric in available_metrics if metric in stats_df.index]

        if not key_metrics:
            warnings.append("No key metrics are available in the provided data for visualization.")
            return {'visuals': visuals, 'warnings': warnings}

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
                warnings.append(f"Error creating visualization for {metric}: {e}")

        return {'visuals': visuals, 'warnings': warnings}

    def generate_visualizations(self, manufacturer: Optional[str] = None, weight_class: Optional[str] = None) -> Dict:
        if self.aggregated_data.empty:
            return {}

        filtered_data = self.aggregated_data.copy()
        if manufacturer:
            filtered_data = filtered_data[filtered_data['Manufacturer'].str.lower() == manufacturer.lower()]
        if weight_class:
            filtered_data = filtered_data[filtered_data['Weight_Class'] == weight_class]

        # Apply reasonable bounds filtering
        max_efficiency = 100  # Based on the statistical maximum of ~13 kWh/mi

        energy_data = filtered_data[
            (filtered_data['Total Energy Consumption'] > 0) & (filtered_data['Total Distance'] > 0)
            ].copy()

        energy_data['Energy_Efficiency'] = energy_data['Total Energy Consumption'] / energy_data['Total Distance']
        energy_data = energy_data[energy_data['Energy_Efficiency'] <= max_efficiency]

        visuals = {}

        # Energy Efficiency by Vehicle
        vehicle_metrics = energy_data.groupby('Vehicle_ID').agg({
            'Energy_Efficiency': 'mean',
            'Manufacturer': 'first'
        }).reset_index()

        if not vehicle_metrics.empty:
            visuals['efficiency'] = px.bar(
                vehicle_metrics,
                x='Vehicle_ID',
                y='Energy_Efficiency',
                color='Manufacturer',
                title='Energy Efficiency by Vehicle',
                labels={'Energy_Efficiency': 'Energy per Mile (kWh/mi)'}
            )

        # Distance vs Energy Scatter
        visuals['energy_distance'] = px.scatter(
            energy_data,
            x='Total Distance',
            y='Total Energy Consumption',
            color='Manufacturer' if not manufacturer else 'Weight_Class',
            title='Energy Consumption vs Distance',
            trendline='ols'
        )

        # Temperature Impact on Energy Efficiency
        visuals['temperature'] = px.scatter(
            energy_data,
            x='Average Ambient Temperature',
            y='Energy_Efficiency',
            color='Manufacturer' if not manufacturer else 'Weight_Class',
            title='Temperature Impact on Energy Efficiency',
            labels={'Energy_Efficiency': 'Energy per Mile (kWh/mi)'},
            trendline='ols'
        )

        # New Plot: kWh/mi vs Average Speed
        if 'Energy_Efficiency' in energy_data.columns and 'Average Speed' in energy_data.columns:
            visuals['efficiency_speed'] = px.scatter(
                energy_data,
                x='Average Speed',
                y='Energy_Efficiency',
                color='Manufacturer' if not manufacturer else 'Weight_Class',
                title='Energy Efficiency vs Average Speed',
                labels={
                    'Average Speed': 'Average Speed (mph)',
                    'Energy_Efficiency': 'Energy per Mile (kWh/mi)'
                },
                trendline='ols'
            )

        # Retain Idle Time Distribution if desired (optional)
        # Idle Time Analysis
        if 'Percent Idling Time' in filtered_data.columns:
            idle_data = filtered_data[filtered_data['Percent Idling Time'] <= 100]
            visuals['idle_time'] = px.box(
                idle_data,
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
            filtered_data = filtered_data[filtered_data['Manufacturer'].str.lower() == manufacturer.lower()]
        if weight_class:
            filtered_data = filtered_data[filtered_data['Weight_Class'] == weight_class]

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
                    q1 = valid_data.quantile(0.25)
                    q3 = valid_data.quantile(0.75)
                    iqr = q3 - q1
                    basic_stats_data[col] = {
                        'Count': len(valid_data),
                        'Mean': valid_data.mean(),
                        'Std': valid_data.std(),
                        'Min': valid_data.min(),
                        '25%': q1,
                        'Median': valid_data.median(),
                        '75%': q3,
                        'Max': valid_data.max(),
                        'IQR': iqr
                    }

        basic_stats = pd.DataFrame(basic_stats_data)

        # Ensure rounding is applied consistently
        round_dict = {
            'Total Distance': 2,
            'Total Energy Consumption': 2,
            'Average Ambient Temperature': 2,
            'Total Trip Time': 4,
            'Driving Time': 4,
            'Idling Time': 4,
            'Average Speed': 2,
            '25%': 2,
            '75%': 2,
            'IQR': 2
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
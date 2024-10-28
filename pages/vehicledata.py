import pandas as pd
import plotly.express as px
from typing import Dict, Optional


class VehicleDataAnalyzer:
    def __init__(self, df: pd.DataFrame, vehicle_info: Dict):
        self.df = df
        self.vehicle_info = vehicle_info
        self.df = self.convert_time_to_hours(self.df)

    def convert_time_to_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert time columns to hours"""
        df_copy = df.copy()
        time_columns = [col for col in df_copy.columns if 'time' in col.lower() and 'date' not in col.lower()]

        for col in time_columns:
            if df_copy[col].dtype == 'float64':
                df_copy[col] = df_copy[col] * 24  # Assuming original values are in days
            elif df_copy[col].dtype == 'object':
                try:
                    df_copy[col] = pd.to_timedelta(df_copy[col]).dt.total_seconds() / 3600
                except:
                    pass  # Silent fail as we'll handle warnings in the UI
        return df_copy

    def get_vehicle_info_display(self) -> Dict[str, Dict[str, str]]:
        """Organize vehicle information for display"""
        if self.vehicle_info:
            return {
                'column1': {
                    'Vehicle ID': self.vehicle_info['Vehicle ID'],
                    'Model Year': self.vehicle_info['Model Year'],
                    'Manufacturer': self.vehicle_info['Manufacturer'],
                    'Model Name': self.vehicle_info['Model Name']
                },
                'column2': {
                    'Weight Class': self.vehicle_info['Weight Class'],
                    'Battery Chemistry': self.vehicle_info['Battery Chemistry'],
                    'Rated Energy': f"{self.vehicle_info['Rated Energy']} kWh",
                    'Max Charge Rate': f"{self.vehicle_info['Max Charge Rate']} kW"
                },
                'column3': {
                    'State': self.vehicle_info['State'],
                    'Body Style': self.vehicle_info['Body Style'],
                    'Sector': self.vehicle_info['Sector'],
                    'Vocation': self.vehicle_info['Vocation']
                }
            }
        return {}

    def calculate_summary_statistics(self) -> Optional[pd.DataFrame]:
        """Calculate summary statistics for the vehicle"""
        exclude_cols = ['Initial Odometer', 'Final Odometer']
        numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        stat_columns = [col for col in numeric_columns if col not in exclude_cols]

        if not stat_columns:
            return None

        # Calculate basic statistics
        summary_stats = self.df[stat_columns].describe()

        # Add median and variance
        summary_stats.loc['median'] = summary_stats.loc['50%']
        summary_stats.loc['variance'] = self.df[stat_columns].var()

        # Reorder rows
        new_index = ['count', 'mean', 'median', 'std', 'variance', 'min', '25%', '50%', '75%', 'max']
        summary_stats = summary_stats.reindex(new_index)

        # Add trip distance statistics if available
        if all(col in self.df.columns for col in ['Initial Odometer', 'Final Odometer']):
            valid_readings = self.df[['Initial Odometer', 'Final Odometer']].dropna()
            if len(valid_readings) > 0:
                distances = valid_readings['Final Odometer'] - valid_readings['Initial Odometer']
                distance_stats = pd.DataFrame({
                    'Trip Distance': {
                        'count': len(distances),
                        'mean': distances.mean(),
                        'median': distances.median(),
                        'std': distances.std(),
                        'variance': distances.var(),
                        'min': distances.min(),
                        '25%': distances.quantile(0.25),
                        '50%': distances.median(),
                        '75%': distances.quantile(0.75),
                        'max': distances.max()
                    }
                })
                summary_stats = pd.concat([summary_stats, distance_stats], axis=1)

        return summary_stats.round(4)

    def generate_visualizations(self):
        """Generate all visualizations for the vehicle"""
        visualizations = {}

        # Distance over time
        date_column = next((col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
        distance_column = next((col for col in self.df.columns if 'distance' in col.lower()), None)

        if date_column and distance_column:
            visualizations['distance_time'] = px.line(
                self.df,
                x=date_column,
                y=distance_column,
                title='Total Distance Over Time'
            ).update_layout(
                xaxis_title='Date',
                yaxis_title='Total Distance (miles)'
            )

        # Energy vs Distance
        energy_column = next(
            (col for col in self.df.columns if 'energy' in col.lower() or 'consumption' in col.lower()), None)
        if distance_column and energy_column:
            visualizations['energy_distance'] = px.scatter(
                self.df,
                x=distance_column,
                y=energy_column,
                title='Total Energy Consumption vs Total Distance'
            ).update_layout(
                xaxis_title='Total Distance (miles)',
                yaxis_title='Total Energy Consumption'
            )

        # Driving vs Idling Time
        driving_time_column = next((col for col in self.df.columns if 'driving time' in col.lower()), None)
        idling_time_column = next((col for col in self.df.columns if 'idling time' in col.lower()), None)

        if driving_time_column and idling_time_column:
            avg_times = self.df[[driving_time_column, idling_time_column]].mean()
            visualizations['time_comparison'] = px.bar(
                x=avg_times.index,
                y=avg_times.values,
                title='Average Driving Time vs Idling Time'
            ).update_layout(
                xaxis_title='',
                yaxis_title='Time (hours)'
            )

        # Correlation matrix
        correlation_matrix = self.df.select_dtypes(include=[float, int]).corr()
        visualizations['correlation'] = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title='Correlation Matrix'
        )

        return visualizations

    def create_custom_scatter(self, x_column: str, y_column: str):
        """Create custom scatter plot based on user-selected columns"""
        return px.scatter(
            self.df,
            x=x_column,
            y=y_column,
            title=f'{y_column} vs {x_column}'
        )

    def get_numeric_columns(self):
        """Get list of numeric columns for custom plotting"""
        return list(self.df.select_dtypes(include=[float, int]).columns)
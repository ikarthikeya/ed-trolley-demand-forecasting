#!/usr/bin/env python3
"""
UNIFIED VISUALIZER MODULE
=========================

Unified visualization module that consolidates all visualization functionality.
Provides comprehensive visualization capabilities for Emergency Department trolley demand forecasting.

Features:
- Exploratory data analysis visualizations
- Model performance comparisons
- Forecast visualizations with confidence intervals
- SHAP explainability plots
- Regional analysis dashboards
- Research-quality publication-ready charts
- Comprehensive evaluation dashboards

Author: Unified Pipeline
Date: December 28, 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any

# Optional dependencies
try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Some time series plots will be disabled.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')


class Visualizer:
    """
    Unified visualization class that handles all visualization logic.
    """
    
    def __init__(self, results_dir: str = None, style: str = 'healthcare'):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory to save visualization results
            style: Visualization style ('healthcare', 'academic', 'publication')
        """
        self.results_dir = Path(results_dir) if results_dir else Path("results/visualizations")
        self.style = style
        
        # Create subdirectories
        self.charts_dir = self.results_dir / "charts"
        self.dashboards_dir = self.results_dir / "dashboards"
        self.plots_dir = self.results_dir / "plots"
        self.exports_dir = self.results_dir / "exports"
        
        # Create directories
        for directory in [self.charts_dir, self.dashboards_dir, self.plots_dir, self.exports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Set style
        self._set_style()
        
        # Healthcare color palette
        self.colors = {
            'primary': '#2E86AB',      # Medical blue
            'secondary': '#A23B72',    # Clinical purple
            'accent': '#F18F01',       # Healthcare orange
            'positive': '#52A57A',     # Positive outcome green
            'negative': '#C73E1D',     # Risk factor red
            'neutral': '#6C757D',      # Neutral gray
            'background': '#F8F9FA',   # Light background
            'grid': '#E9ECEF'          # Grid color
        }
        
        print(f"🎨 Visualizer initialized")
        print(f"📂 Results directory: {self.results_dir}")
        print(f"🎨 Style: {style}")
    
    def _set_style(self):
        """Set matplotlib style based on selected theme."""
        if self.style == 'healthcare':
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 12,
                'figure.titlesize': 18,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'figure.facecolor': 'white',
                'axes.facecolor': 'white'
            })
        elif self.style == 'academic':
            plt.style.use('seaborn-v0_8-paper')
            plt.rcParams.update({
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'font.size': 10,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'legend.fontsize': 10,
                'figure.titlesize': 16
            })
        elif self.style == 'publication':
            plt.style.use('seaborn-v0_8-paper')
            plt.rcParams.update({
                'figure.dpi': 600,
                'savefig.dpi': 600,
                'font.size': 8,
                'axes.labelsize': 10,
                'axes.titlesize': 12,
                'legend.fontsize': 8,
                'figure.titlesize': 14,
                'font.family': 'serif'
            })
    
    # ========================================================================================
    # TIME SERIES VISUALIZATIONS
    # ========================================================================================
    
    def plot_time_series(self, data: pd.DataFrame, column: str = 'Total_Trolleys',
                        title: str = None, figsize: Tuple[int, int] = (15, 7),
                        save_name: str = None) -> None:
        """
        Plot time series data.
        
        Args:
            data: DataFrame with time series data (index should be datetime)
            column: Column name to plot
            title: Custom title for the plot
            figsize: Figure size tuple
            save_name: Custom filename for saving
        """
        plt.figure(figsize=figsize)
        
        # Plot the time series
        plt.plot(data.index, data[column], linewidth=2, color=self.colors['primary'])
        
        # Add trend line if data is long enough
        if len(data) > 30:
            z = np.polyfit(range(len(data)), data[column], 1)
            p = np.poly1d(z)
            plt.plot(data.index, p(range(len(data))), 
                    linestyle='--', color=self.colors['negative'], alpha=0.8, linewidth=1)
        
        # Formatting
        title = title or f'Time Series: {column}'
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel(column.replace('_', ' '), fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = save_name or f"{column.lower().replace(' ', '_')}_time_series.png"
        plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Time series plot saved: {filename}")
    
    def plot_seasonal_decomposition(self, data: pd.DataFrame, column: str = 'Total_Trolleys',
                                   model: str = 'additive', period: int = 7,
                                   save_name: str = None) -> None:
        """
        Plot seasonal decomposition of time series.
        
        Args:
            data: DataFrame with time series data
            column: Column to decompose
            model: 'additive' or 'multiplicative'
            period: Seasonal period (7 for weekly, 365 for yearly)
            save_name: Custom filename for saving
        """
        if not STATSMODELS_AVAILABLE:
            print("⚠️ statsmodels not available for seasonal decomposition")
            return
        
        try:
            # Ensure data has datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                print("⚠️ Data index is not datetime, attempting conversion")
                if 'Date' in data.columns:
                    data = data.set_index('Date')
                    data.index = pd.to_datetime(data.index)
                else:
                    print("⚠️ Cannot convert to datetime index, skipping seasonal decomposition")
                    return
            else:
                # Ensure datetime index is properly formatted
                if not hasattr(data.index, 'freq') or data.index.freq is None:
                    data.index = pd.to_datetime(data.index)
            
            # Ensure regular frequency
            if data.index.freq is None:
                try:
                    data = data.asfreq('D')  # Daily frequency
                except:
                    # If asfreq fails, try resampling
                    data = data.resample('D').mean()
            
            # Remove any categorical columns that might interfere
            numeric_data = data.select_dtypes(include=[np.number])
            if column not in numeric_data.columns:
                print(f"⚠️ Column {column} not found in numeric data")
                return
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                numeric_data[column].dropna(), 
                model=model, 
                period=period
            )
            
            # Create subplot
            fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
            
            # Plot components with proper datetime handling
            decomposition.observed.plot(ax=axes[0], title='Observed', color=self.colors['primary'])
            decomposition.trend.plot(ax=axes[1], title='Trend', color=self.colors['secondary'])
            decomposition.seasonal.plot(ax=axes[2], title='Seasonality', color=self.colors['accent'])
            decomposition.resid.plot(ax=axes[3], title='Residuals', color=self.colors['neutral'])
            
            # Format subplots
            for ax in axes:
                ax.grid(True, alpha=0.3)
                # Ensure x-axis handles datetime properly
                try:
                    ax.tick_params(axis='x', rotation=45)
                except:
                    pass
            
            plt.suptitle(f'Seasonal Decomposition: {column}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            filename = save_name or f"{column.lower().replace(' ', '_')}_seasonal_decomposition.png"
            plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Seasonal decomposition plot saved: {filename}")
            
        except Exception as e:
            print(f"⚠️ Error in seasonal decomposition: {e}")
            plt.close('all')  # Clean up any partial plots
    
    def plot_autocorrelation(self, data: pd.DataFrame, column: str = 'Total_Trolleys',
                           lags: int = 40, save_name: str = None) -> None:
        """
        Plot autocorrelation and partial autocorrelation functions.
        
        Args:
            data: DataFrame with time series data
            column: Column to analyze
            lags: Number of lags to plot
            save_name: Custom filename for saving
        """
        if not STATSMODELS_AVAILABLE:
            print("⚠️ statsmodels not available for autocorrelation plots")
            return
        
        try:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # ACF plot
            plot_acf(data[column], lags=lags, ax=axes[0], color=self.colors['primary'])
            axes[0].set_title(f'Autocorrelation Function (ACF) for {column}', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            
            # PACF plot
            plot_pacf(data[column], lags=lags, ax=axes[1], color=self.colors['secondary'])
            axes[1].set_title(f'Partial Autocorrelation Function (PACF) for {column}', fontsize=14)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            filename = save_name or f"{column.lower().replace(' ', '_')}_autocorrelation.png"
            plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Autocorrelation plot saved: {filename}")
            
        except Exception as e:
            print(f"⚠️ Error in autocorrelation plot: {e}")
    
    def plot_distribution(self, data: pd.DataFrame, column: str = 'Total_Trolleys',
                         save_name: str = None) -> None:
        """
        Plot distribution of a column with histogram and box plot.
        
        Args:
            data: DataFrame containing the data
            column: Column to plot
            save_name: Custom filename for saving
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram with KDE
        axes[0].hist(data[column], bins=30, alpha=0.7, color=self.colors['primary'],
                    edgecolor='black', density=True)
        
        # Add KDE if possible
        try:
            from scipy import stats
            x = np.linspace(data[column].min(), data[column].max(), 100)
            kde = stats.gaussian_kde(data[column].dropna())
            axes[0].plot(x, kde(x), color=self.colors['negative'], linewidth=2, label='KDE')
            axes[0].legend()
        except ImportError:
            pass
        
        axes[0].set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(column.replace('_', ' '), fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(data[column], patch_artist=True,
                       boxprops=dict(facecolor=self.colors['primary'], alpha=0.7),
                       medianprops=dict(color=self.colors['negative'], linewidth=2))
        axes[1].set_title(f'Box Plot of {column}', fontsize=14, fontweight='bold')
        axes[1].set_ylabel(column.replace('_', ' '), fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = save_name or f"{column.lower().replace(' ', '_')}_distribution.png"
        plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Distribution plot saved: {filename}")
    
    def plot_correlation_matrix(self, data: pd.DataFrame, figsize: Tuple[int, int] = (12, 10),
                               save_name: str = None) -> None:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data: DataFrame with numeric columns
            figsize: Figure size tuple
            save_name: Custom filename for saving
        """
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save plot
        filename = save_name or "correlation_matrix.png"
        plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Correlation matrix saved: {filename}")
    
    # ========================================================================================
    # MODEL PERFORMANCE VISUALIZATIONS
    # ========================================================================================
    
    def plot_model_comparison(self, results_df: pd.DataFrame, metric: str = 'rmse',
                             title: str = None, save_name: str = None) -> None:
        """
        Plot model performance comparison.
        
        Args:
            results_df: DataFrame with model results
            metric: Metric to compare ('rmse', 'mae', 'r2', 'mape')
            title: Custom title
            save_name: Custom filename for saving
        """
        # Ensure metric column exists
        metric_col = metric.upper() if metric.upper() in results_df.columns else metric
        if metric_col not in results_df.columns:
            print(f"⚠️ Metric '{metric}' not found in results")
            return
        
        # Sort by metric (ascending for error metrics, descending for R²)
        ascending = metric.lower() != 'r2'
        results_sorted = results_df.sort_values(metric_col, ascending=ascending)
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot
        bars = plt.barh(range(len(results_sorted)), results_sorted[metric_col],
                       color=self.colors['primary'], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, results_sorted[metric_col])):
            plt.text(value + (max(results_sorted[metric_col]) * 0.01), bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center', fontweight='bold')
        
        # Formatting
        plt.yticks(range(len(results_sorted)), results_sorted['Model'])
        plt.xlabel(f'{metric_col} Value', fontsize=12, fontweight='bold')
        plt.title(title or f'Model Performance Comparison ({metric_col})', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Save plot
        filename = save_name or f"model_comparison_{metric.lower()}.png"
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Model comparison plot saved: {filename}")
    
    def plot_performance_metrics(self, results_df: pd.DataFrame, save_name: str = None) -> None:
        """
        Plot comprehensive performance metrics dashboard.
        
        Args:
            results_df: DataFrame with model results
            save_name: Custom filename for saving
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Metrics Dashboard', fontsize=18, fontweight='bold')
        
        metrics = ['RMSE', 'MAE', 'R²', 'MAPE']
        colors = [self.colors['negative'], self.colors['accent'], 
                 self.colors['positive'], self.colors['secondary']]
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            if metric not in results_df.columns:
                continue
                
            ax = axes[i//2, i%2]
            
            # Sort appropriately
            ascending = metric != 'R²'
            data_sorted = results_df.sort_values(metric, ascending=ascending)
            
            # Create bar plot
            bars = ax.bar(range(len(data_sorted)), data_sorted[metric],
                         color=color, alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar, value in zip(bars, data_sorted[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Formatting
            ax.set_xticks(range(len(data_sorted)))
            ax.set_xticklabels(data_sorted['Model'], rotation=45, ha='right')
            ax.set_ylabel(metric, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        filename = save_name or "performance_metrics_dashboard.png"
        plt.savefig(self.dashboards_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Performance metrics dashboard saved: {filename}")
    
    # ========================================================================================
    # FORECAST VISUALIZATIONS
    # ========================================================================================
    
    def plot_forecast(self, historical_data: pd.DataFrame = None, 
                     forecast_data: pd.DataFrame = None,
                     model_name: str = "Model", confidence_intervals: pd.DataFrame = None,
                     figsize: Tuple[int, int] = (15, 8), save_name: str = None) -> None:
        """
        Plot forecast with historical data and confidence intervals.
        
        Args:
            historical_data: Historical time series data
            forecast_data: Forecast data with dates and predictions
            model_name: Name of the model
            confidence_intervals: DataFrame with lower and upper bounds
            figsize: Figure size tuple
            save_name: Custom filename for saving
        """
        plt.figure(figsize=figsize)
        
        # Plot historical data
        if historical_data is not None:
            plt.plot(historical_data.index, historical_data.values,
                    label='Historical Data', color=self.colors['primary'], linewidth=2)
        
        # Plot forecast
        if forecast_data is not None:
            plt.plot(forecast_data.index, forecast_data.values,
                    label=f'{model_name} Forecast', color=self.colors['negative'],
                    linewidth=2, linestyle='--', marker='o', markersize=4)
        
        # Plot confidence intervals
        if confidence_intervals is not None:
            plt.fill_between(forecast_data.index, 
                           confidence_intervals.iloc[:, 0],
                           confidence_intervals.iloc[:, 1],
                           color=self.colors['neutral'], alpha=0.3,
                           label='Confidence Interval')
        
        # Formatting
        plt.title(f'{model_name} Forecast vs Historical Data', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Trolley Count', fontsize=12, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = save_name or f"{model_name.lower().replace(' ', '_')}_forecast.png"
        plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Forecast plot saved: {filename}")
    
    def plot_multiple_forecasts(self, historical_data: pd.DataFrame = None,
                               forecasts_dict: Dict[str, pd.DataFrame] = None,
                               figsize: Tuple[int, int] = (15, 8), save_name: str = None) -> None:
        """
        Plot multiple model forecasts on the same chart.
        
        Args:
            historical_data: Historical time series data
            forecasts_dict: Dictionary of {model_name: forecast_data}
            figsize: Figure size tuple
            save_name: Custom filename for saving
        """
        plt.figure(figsize=figsize)
        
        # Plot historical data
        if historical_data is not None:
            # Ensure only numeric data is plotted to avoid matplotlib categorical conversion errors
            if hasattr(historical_data, 'select_dtypes'):
                # If it's a DataFrame, select only numeric columns
                numeric_data = historical_data.select_dtypes(include=[np.number])
                if not numeric_data.empty:
                    # Use the first numeric column for visualization
                    plot_data = numeric_data.iloc[:, 0] if len(numeric_data.columns) > 0 else historical_data
                else:
                    print("⚠️ No numeric data found for historical plot, skipping...")
                    plot_data = None
            else:
                # If it's a Series, check if it's numeric
                if pd.api.types.is_numeric_dtype(historical_data):
                    plot_data = historical_data
                else:
                    print("⚠️ Historical data is not numeric, skipping plot...")
                    plot_data = None
            
            if plot_data is not None:
                plt.plot(plot_data.index, plot_data.values,
                        label='Historical Data', color=self.colors['primary'], 
                        linewidth=2, alpha=0.8)
        
        # Plot forecasts
        if forecasts_dict is not None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts_dict)))
            
            for i, (model_name, forecast_data) in enumerate(forecasts_dict.items()):
                plt.plot(forecast_data.index, forecast_data.values,
                        label=f'{model_name} Forecast', color=colors[i],
                        linewidth=2, linestyle='--', marker='o', markersize=3)
        
        # Formatting
        plt.title('Multiple Model Forecasts Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Trolley Count', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = save_name or "multiple_forecasts_comparison.png"
        plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Multiple forecasts plot saved: {filename}")
    
    def plot_residuals_analysis(self, residuals: np.ndarray, model_name: str = "Model",
                               predictions: np.ndarray = None, save_name: str = None) -> None:
        """
        Plot residuals analysis for model diagnostics.
        
        Args:
            residuals: Model residuals
            model_name: Name of the model
            predictions: Model predictions (optional)
            save_name: Custom filename for saving
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Residuals Analysis: {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Residuals vs Fitted
        if predictions is not None:
            axes[0, 0].scatter(predictions, residuals, alpha=0.6, color=self.colors['primary'])
            axes[0, 0].axhline(y=0, color=self.colors['negative'], linestyle='--')
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Fitted')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color=self.colors['secondary'],
                       edgecolor='black', density=True)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')
            axes[1, 0].grid(True, alpha=0.3)
        except ImportError:
            axes[1, 0].text(0.5, 0.5, 'Q-Q plot requires scipy', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. Time series of residuals
        axes[1, 1].plot(residuals, color=self.colors['accent'], linewidth=1)
        axes[1, 1].axhline(y=0, color=self.colors['negative'], linestyle='--')
        axes[1, 1].set_xlabel('Observation')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Time Series')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = save_name or f"{model_name.lower().replace(' ', '_')}_residuals_analysis.png"
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Residuals analysis saved: {filename}")
    
    # ========================================================================================
    # EXPLAINABILITY VISUALIZATIONS
    # ========================================================================================
    
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                               model_name: str = "Model", top_n: int = 20,
                               save_name: str = None) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            model_name: Name of the model
            top_n: Number of top features to show
            save_name: Custom filename for saving
        """
        # Convert to DataFrame and sort
        if isinstance(feature_importance, dict):
            fi_df = pd.DataFrame(list(feature_importance.items()), 
                               columns=['feature', 'importance'])
        else:
            fi_df = feature_importance
        
        fi_df = fi_df.sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(12, max(8, top_n * 0.4)))
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(fi_df)), fi_df['importance'],
                       color=self.colors['primary'], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, fi_df['importance'])):
            plt.text(bar.get_width() + max(fi_df['importance'])*0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # Formatting
        plt.yticks(range(len(fi_df)), fi_df['feature'])
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title(f'Feature Importance: {model_name}', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Save plot
        filename = save_name or f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Feature importance plot saved: {filename}")
    
    def plot_shap_summary(self, shap_values: np.ndarray, feature_names: List[str],
                         model_name: str = "Model", save_name: str = None) -> None:
        """
        Plot SHAP summary visualization.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            model_name: Name of the model
            save_name: Custom filename for saving
        """
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP library not available for SHAP plots")
            return
        
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            plt.title(f'{model_name} - SHAP Summary Plot', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            filename = save_name or f"{model_name.lower().replace(' ', '_')}_shap_summary.png"
            plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ SHAP summary plot saved: {filename}")
            
        except Exception as e:
            print(f"⚠️ Error creating SHAP plot: {e}")
    
    # ========================================================================================
    # REGIONAL ANALYSIS VISUALIZATIONS
    # ========================================================================================
    
    def plot_regional_comparison(self, regional_data: Dict[str, pd.DataFrame],
                                metric: str = 'mean', save_name: str = None) -> None:
        """
        Plot regional comparison of trolley demand.
        
        Args:
            regional_data: Dictionary of {region_name: data_df}
            metric: Metric to compare ('mean', 'max', 'std')
            save_name: Custom filename for saving
        """
        # Calculate metric for each region
        region_metrics = {}
        for region_name, data in regional_data.items():
            if 'Total_Trolleys' in data.columns:
                if metric == 'mean':
                    region_metrics[region_name] = data['Total_Trolleys'].mean()
                elif metric == 'max':
                    region_metrics[region_name] = data['Total_Trolleys'].max()
                elif metric == 'std':
                    region_metrics[region_name] = data['Total_Trolleys'].std()
        
        if not region_metrics:
            print("⚠️ No regional data available for comparison")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Sort regions by metric
        sorted_regions = sorted(region_metrics.items(), key=lambda x: x[1], reverse=True)
        regions, values = zip(*sorted_regions)
        
        # Create bar plot
        bars = plt.bar(range(len(regions)), values, color=self.colors['primary'],
                      alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Formatting
        plt.xticks(range(len(regions)), regions, rotation=45, ha='right')
        plt.ylabel(f'{metric.title()} Trolley Count', fontsize=12, fontweight='bold')
        plt.title(f'Regional Trolley Demand Comparison ({metric.title()})', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save plot
        filename = save_name or f"regional_comparison_{metric}.png"
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Regional comparison plot saved: {filename}")
    
    # ========================================================================================
    # COMPREHENSIVE DASHBOARDS
    # ========================================================================================
    
    def create_comprehensive_dashboard(self, data: pd.DataFrame,
                                     results_df: pd.DataFrame = None,
                                     forecasts_dict: Dict[str, pd.DataFrame] = None,
                                     save_name: str = None) -> None:
        """
        Create comprehensive analysis dashboard.
        
        Args:
            data: Main dataset
            results_df: Model performance results
            forecasts_dict: Dictionary of forecasts
            save_name: Custom filename for saving
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Emergency Department Trolley Demand Analysis Dashboard', 
                     fontsize=20, fontweight='bold', y=0.96)
        
        # 1. Time series (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'Total_Trolleys' in data.columns:
            ax1.plot(data.index, data['Total_Trolleys'], color=self.colors['primary'], linewidth=2)
            ax1.set_title('Time Series Overview', fontweight='bold')
            ax1.set_ylabel('Trolley Count')
            ax1.grid(True, alpha=0.3)
        
        # 2. Distribution (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'Total_Trolleys' in data.columns:
            ax2.hist(data['Total_Trolleys'], bins=30, alpha=0.7, color=self.colors['secondary'],
                    edgecolor='black')
            ax2.set_title('Demand Distribution', fontweight='bold')
            ax2.set_xlabel('Trolley Count')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
        
        # 3. Model performance (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if results_df is not None and 'RMSE' in results_df.columns:
            bars = ax3.bar(range(len(results_df)), results_df['RMSE'],
                          color=self.colors['negative'], alpha=0.8)
            ax3.set_title('Model Performance (RMSE)', fontweight='bold')
            ax3.set_ylabel('RMSE')
            ax3.set_xticks(range(len(results_df)))
            ax3.set_xticklabels(results_df['Model'], rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # 4. Seasonal pattern (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        if 'Total_Trolleys' in data.columns:
            try:
                # Weekly pattern - ensure numeric indexing
                if isinstance(data.index, pd.DatetimeIndex):
                    weekly_pattern = data.groupby(data.index.dayofweek)['Total_Trolleys'].mean()
                    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    # Use numeric indices for plotting to avoid categorical issues
                    ax4.plot(range(7), weekly_pattern.values, marker='o', color=self.colors['accent'], linewidth=2)
                    ax4.set_xticks(range(7))
                    ax4.set_xticklabels(days)
                else:
                    # Fallback for non-datetime index
                    ax4.text(0.5, 0.5, 'Weekly pattern\n(requires datetime index)', 
                            ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Weekly Pattern', fontweight='bold')
                ax4.set_ylabel('Average Trolleys')
                ax4.grid(True, alpha=0.3)
            except Exception as e:
                print(f"⚠️ Error plotting weekly pattern: {e}")
                ax4.text(0.5, 0.5, 'Weekly pattern\n(data error)', 
                        ha='center', va='center', transform=ax4.transAxes)
        
        # 5. Monthly pattern (middle center)
        ax5 = fig.add_subplot(gs[1, 1])
        if 'Total_Trolleys' in data.columns:
            try:
                if isinstance(data.index, pd.DatetimeIndex):
                    monthly_pattern = data.groupby(data.index.month)['Total_Trolleys'].mean()
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    # Ensure we have data for all 12 months, fill missing with 0
                    full_monthly = pd.Series(index=range(1, 13), data=0.0)
                    full_monthly.update(monthly_pattern)
                    
                    # Use numeric indices for bar plot
                    ax5.bar(range(12), full_monthly.values, color=self.colors['positive'], alpha=0.8)
                    ax5.set_xticks(range(12))
                    ax5.set_xticklabels(months, rotation=45)
                else:
                    # Fallback for non-datetime index
                    ax5.text(0.5, 0.5, 'Monthly pattern\n(requires datetime index)', 
                            ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Monthly Pattern', fontweight='bold')
                ax5.set_ylabel('Average Trolleys')
                ax5.grid(True, alpha=0.3)
            except Exception as e:
                print(f"⚠️ Error plotting monthly pattern: {e}")
                ax5.text(0.5, 0.5, 'Monthly pattern\n(data error)', 
                        ha='center', va='center', transform=ax5.transAxes)
        
        # 6. Forecasts comparison (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        if forecasts_dict is not None:
            try:
                colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts_dict)))
                for i, (model_name, forecast_data) in enumerate(forecasts_dict.items()):
                    # Ensure forecast data is properly formatted
                    if isinstance(forecast_data, pd.Series):
                        # Use numeric range for x-axis to avoid datetime issues
                        x_vals = range(len(forecast_data))
                        y_vals = pd.to_numeric(forecast_data.values, errors='coerce')
                        y_vals = np.nan_to_num(y_vals, nan=0.0)  # Replace NaN with 0
                        
                        ax6.plot(x_vals, y_vals, label=model_name, 
                                color=colors[i], marker='o', markersize=3)
                    elif isinstance(forecast_data, pd.DataFrame) and not forecast_data.empty:
                        # Handle DataFrame forecasts
                        value_col = 'values' if 'values' in forecast_data.columns else forecast_data.columns[0]
                        x_vals = range(len(forecast_data))
                        y_vals = pd.to_numeric(forecast_data[value_col], errors='coerce')
                        y_vals = np.nan_to_num(y_vals, nan=0.0)
                        
                        ax6.plot(x_vals, y_vals, label=model_name, 
                                color=colors[i], marker='o', markersize=3)
                
                ax6.set_title('Forecast Comparison', fontweight='bold')
                ax6.set_ylabel('Predicted Trolleys')
                ax6.set_xlabel('Forecast Steps')
                ax6.legend(fontsize=8)
                ax6.grid(True, alpha=0.3)
            except Exception as e:
                print(f"⚠️ Error plotting forecasts: {e}")
                ax6.text(0.5, 0.5, 'Forecast comparison\n(data error)', 
                        ha='center', va='center', transform=ax6.transAxes)
        else:
            ax6.text(0.5, 0.5, 'No forecast data\navailable', 
                    ha='center', va='center', transform=ax6.transAxes)
        
        # 7. Summary statistics (bottom span)
        ax7 = fig.add_subplot(gs[2, :])
        if 'Total_Trolleys' in data.columns:
            stats_text = f"""
            SUMMARY STATISTICS
            
            • Dataset Period: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}
            • Total Observations: {len(data):,}
            • Mean Daily Trolleys: {data['Total_Trolleys'].mean():.1f}
            • Maximum Daily Trolleys: {data['Total_Trolleys'].max():.0f}
            • Standard Deviation: {data['Total_Trolleys'].std():.1f}
            """
            
            if results_df is not None:
                champion = results_df.loc[results_df['RMSE'].idxmin()]
                stats_text += f"""
            • Champion Model: {champion['Model']}
            • Best RMSE: {champion['RMSE']:.3f}
            • Best R²: {champion.get('R²', 'N/A')}
                """
            
            ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['background']))
            ax7.axis('off')
        
        plt.tight_layout()
        
        # Save dashboard
        filename = save_name or "comprehensive_dashboard.png"
        plt.savefig(self.dashboards_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Comprehensive dashboard saved: {filename}")
    
    def save_all_plots(self, data: pd.DataFrame, results_df: pd.DataFrame = None,
                      forecasts_dict: Dict[str, pd.DataFrame] = None) -> None:
        """
        Generate and save all standard plots.
        
        Args:
            data: Main dataset
            results_df: Model performance results
            forecasts_dict: Dictionary of forecasts
        """
        print("🎨 Generating all visualization plots...")
        
        # Time series analysis
        if 'Total_Trolleys' in data.columns:
            self.plot_time_series(data, 'Total_Trolleys')
            self.plot_distribution(data, 'Total_Trolleys')
            
            if STATSMODELS_AVAILABLE:
                self.plot_seasonal_decomposition(data, 'Total_Trolleys')
                self.plot_autocorrelation(data, 'Total_Trolleys')
        
        # Correlation analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            self.plot_correlation_matrix(data[numeric_cols])
        
        # Model performance
        if results_df is not None:
            for metric in ['RMSE', 'MAE', 'R²', 'MAPE']:
                if metric in results_df.columns:
                    self.plot_model_comparison(results_df, metric.lower())
            
            self.plot_performance_metrics(results_df)
        
        # Forecasts
        if forecasts_dict is not None:
            self.plot_multiple_forecasts(data, forecasts_dict)
        
        # Comprehensive dashboard
        self.create_comprehensive_dashboard(data, results_df, forecasts_dict)
        
        print("✅ All visualization plots generated successfully!")

    def _safe_categorical_plot(self, data: pd.Series, labels: List[str], 
                              plot_type: str = 'bar', ax=None, **kwargs) -> bool:
        """
        Safely plot categorical data by ensuring proper numeric indexing.
        
        Args:
            data: Data series to plot
            labels: Category labels
            plot_type: Type of plot ('bar', 'line', etc.)
            ax: Matplotlib axis to plot on
            **kwargs: Additional plotting arguments
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure data is numeric and properly indexed
            numeric_data = pd.to_numeric(data, errors='coerce').fillna(0)
            
            if plot_type == 'bar':
                if ax is not None:
                    bars = ax.bar(range(len(numeric_data)), numeric_data.values, **kwargs)
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45)
                else:
                    bars = plt.bar(range(len(numeric_data)), numeric_data.values, **kwargs)
                    plt.xticks(range(len(labels)), labels, rotation=45)
            elif plot_type == 'line':
                if ax is not None:
                    ax.plot(range(len(numeric_data)), numeric_data.values, **kwargs)
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45)
                else:
                    plt.plot(range(len(numeric_data)), numeric_data.values, **kwargs)
                    plt.xticks(range(len(labels)), labels, rotation=45)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error in categorical plot: {e}")
            return False


# Example usage and testing
def main():
    """Example usage of the Visualizer."""
    print("🎨 Visualizer Demo")
    print("=" * 30)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    
    # Sample time series with trend and seasonality
    trend = np.linspace(10, 15, 365)
    seasonal = 3 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly pattern
    noise = np.random.normal(0, 2, 365)
    trolleys = trend + seasonal + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'Total_Trolleys': trolleys,
        'Day_of_Week': [d.weekday() for d in dates],
        'Month': [d.month for d in dates],
        'Is_Weekend': [1 if d.weekday() >= 5 else 0 for d in dates]
    }, index=dates)
    
    # Sample model results
    results_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'LSTM', 'Prophet'],
        'RMSE': [2.1, 1.8, 2.3, 2.0],
        'MAE': [1.6, 1.4, 1.8, 1.5],
        'R²': [0.85, 0.89, 0.82, 0.87],
        'MAPE': [12.5, 10.8, 14.2, 11.9]
    })
    
    # Sample forecasts
    forecast_dates = pd.date_range('2024-01-01', periods=14, freq='D')
    forecasts_dict = {
        'Random Forest': pd.Series(np.random.normal(12, 1.5, 14), index=forecast_dates),
        'XGBoost': pd.Series(np.random.normal(11.8, 1.3, 14), index=forecast_dates),
        'Prophet': pd.Series(np.random.normal(12.2, 1.4, 14), index=forecast_dates)
    }
    
    # Initialize visualizer
    visualizer = Visualizer(results_dir="results/visualizer_demo", style='healthcare')
    
    # Generate all plots
    visualizer.save_all_plots(data, results_df, forecasts_dict)
    
    print(f"\n📊 Visualization Summary:")
    print(f"   Charts saved to: {visualizer.charts_dir}")
    print(f"   Dashboards saved to: {visualizer.dashboards_dir}")
    print(f"   Plots saved to: {visualizer.plots_dir}")


if __name__ == "__main__":
    main()

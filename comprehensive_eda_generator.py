#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) Generator
=======================================================

Generates comprehensive exploratory data analysis figures for Emergency Department 
trolley demand forecasting research. Creates publication-quality visualizations
covering all aspects of the dataset.

Features:
- Dataset overview and summary statistics
- Temporal patterns and seasonality analysis
- Regional variation analysis
- Correlation analysis
- Distribution analysis
- Missing data patterns
- Weather impact analysis
- Holiday effect analysis
- Capacity utilization analysis

Author: Comprehensive EDA Generator
Date: August 9, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.figsize': (12, 8)
})

class ComprehensiveEDA:
    """Comprehensive Exploratory Data Analysis Generator."""
    
    def __init__(self, data_path: str, output_dir: str = "eda_figures"):
        """Initialize EDA generator."""
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 Comprehensive EDA Generator Initialized")
        print(f"📊 Data path: {data_path}")
        print(f"📁 Output directory: {output_dir}")
        
        self.data = None
        self.regional_data = None
        
    def load_and_prepare_data(self):
        """Load and prepare data for analysis."""
        print("\n📈 Loading and preparing data...")
        
        # Load main dataset
        self.data = pd.read_csv(self.data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Create regional aggregated data
        self.regional_data = self.data.groupby(['Date', 'Region']).agg({
            'Total_Trolleys': 'sum',
            'ED_Trolleys': 'sum',
            'Ward_Trolleys': 'sum',
            'Precipitation': 'mean',
            'Is_Weekend': 'first',
            'Is_Public_Holiday': 'first',
            'Season': 'first',
            'Month': 'first',
            'Day_Of_Week': 'first'
        }).reset_index()
        
        print(f"✅ Data loaded: {len(self.data):,} records")
        print(f"✅ Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        print(f"✅ Regions: {self.data['Region'].nunique()}")
        print(f"✅ Hospitals: {self.data['Hospital'].nunique()}")
        
    def generate_dataset_overview(self):
        """Generate dataset overview and summary statistics."""
        print("\n📊 Generating dataset overview...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Dataset Summary Table
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('tight')
        ax1.axis('off')
        
        summary_stats = {
            'Metric': ['Total Records', 'Date Range', 'Hospitals', 'Regions', 
                      'Mean Daily Trolleys', 'Max Daily Trolleys', 'Missing Data %'],
            'Value': [
                f"{len(self.data):,}",
                f"{self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')}",
                f"{self.data['Hospital'].nunique()}",
                f"{self.data['Region'].nunique()}",
                f"{self.data['Total_Trolleys'].mean():.1f}",
                f"{self.data['Total_Trolleys'].max():.0f}",
                f"{(self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns)) * 100):.1f}%"
            ]
        }
        
        table = ax1.table(cellText=list(zip(summary_stats['Metric'], summary_stats['Value'])),
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         bbox=[0.2, 0.2, 0.6, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        ax1.set_title('Dataset Overview Summary', fontsize=16, fontweight='bold', pad=20)
        
        # 2. Distribution of Total Trolleys
        ax2 = fig.add_subplot(gs[1, 0])
        self.data['Total_Trolleys'].hist(bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Total Trolleys')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Total Trolleys')
        ax2.grid(True, alpha=0.3)
        
        # 3. Daily Trolley Counts by Region
        ax3 = fig.add_subplot(gs[1, 1])
        region_daily = self.regional_data.groupby('Region')['Total_Trolleys'].mean().sort_values(ascending=False)
        region_daily.plot(kind='bar', ax=ax3, color='lightcoral')
        ax3.set_xlabel('Region')
        ax3.set_ylabel('Average Daily Trolleys')
        ax3.set_title('Average Daily Trolleys by Region')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Monthly Patterns
        ax4 = fig.add_subplot(gs[1, 2])
        monthly_avg = self.data.groupby('Month')['Total_Trolleys'].mean()
        monthly_avg.plot(kind='line', marker='o', ax=ax4, color='green', linewidth=2)
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Average Trolleys')
        ax4.set_title('Average Trolleys by Month')
        ax4.grid(True, alpha=0.3)
        
        # 5. Daily Patterns
        ax5 = fig.add_subplot(gs[2, 0])
        daily_avg = self.data.groupby('Day_Of_Week')['Total_Trolleys'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_avg.plot(kind='bar', ax=ax5, color='orange')
        ax5.set_xlabel('Day of Week')
        ax5.set_ylabel('Average Trolleys')
        ax5.set_title('Average Trolleys by Day of Week')
        ax5.set_xticklabels(days, rotation=45)
        
        # 6. Weekend vs Weekday
        ax6 = fig.add_subplot(gs[2, 1])
        weekend_comparison = self.data.groupby('Is_Weekend')['Total_Trolleys'].mean()
        weekend_labels = ['Weekday', 'Weekend']
        weekend_comparison.plot(kind='bar', ax=ax6, color=['lightblue', 'lightpink'])
        ax6.set_xlabel('Day Type')
        ax6.set_ylabel('Average Trolleys')
        ax6.set_title('Weekend vs Weekday Comparison')
        ax6.set_xticklabels(weekend_labels, rotation=0)
        
        # 7. Holiday Effect
        ax7 = fig.add_subplot(gs[2, 2])
        holiday_comparison = self.data.groupby('Is_Public_Holiday')['Total_Trolleys'].mean()
        holiday_labels = ['Regular Day', 'Public Holiday']
        holiday_comparison.plot(kind='bar', ax=ax7, color=['lightgreen', 'gold'])
        ax7.set_xlabel('Day Type')
        ax7.set_ylabel('Average Trolleys')
        ax7.set_title('Public Holiday Effect')
        ax7.set_xticklabels(holiday_labels, rotation=0)
        
        # 8. Missing Data Heatmap
        ax8 = fig.add_subplot(gs[3, :2])
        missing_data = self.data.isnull().sum()
        missing_pct = (missing_data / len(self.data) * 100).sort_values(ascending=False)
        missing_pct[missing_pct > 0].plot(kind='bar', ax=ax8, color='red', alpha=0.7)
        ax8.set_xlabel('Columns')
        ax8.set_ylabel('Missing Data %')
        ax8.set_title('Missing Data by Column')
        ax8.tick_params(axis='x', rotation=45)
        
        # 9. Records per Hospital
        ax9 = fig.add_subplot(gs[3, 2])
        hospital_counts = self.data['Hospital'].value_counts().head(10)
        hospital_counts.plot(kind='barh', ax=ax9, color='purple', alpha=0.7)
        ax9.set_xlabel('Number of Records')
        ax9.set_ylabel('Hospital')
        ax9.set_title('Top 10 Hospitals by Record Count')
        
        plt.suptitle('Comprehensive Dataset Overview', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_temporal_analysis(self):
        """Generate comprehensive temporal analysis."""
        print("\n⏰ Generating temporal analysis...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 2, hspace=0.4, wspace=0.3)
        
        # Prepare time series data
        daily_totals = self.regional_data.groupby('Date')['Total_Trolleys'].sum().reset_index()
        daily_totals = daily_totals.set_index('Date')
        
        # 1. Complete Time Series
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(daily_totals.index, daily_totals['Total_Trolleys'], 
                color='blue', linewidth=1, alpha=0.8)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Trolleys')
        ax1.set_title('Complete Time Series: Daily Total Trolleys Across All Regions')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(range(len(daily_totals)), daily_totals['Total_Trolleys'], 1)
        p = np.poly1d(z)
        ax1.plot(daily_totals.index, p(range(len(daily_totals))), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend')
        ax1.legend()
        
        # 2. Seasonal Decomposition (if possible)
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Resample to ensure regular frequency
            daily_regular = daily_totals.resample('D').sum().fillna(method='ffill')
            
            if len(daily_regular) > 365:  # Need at least 1 year for yearly seasonality
                decomposition = seasonal_decompose(daily_regular['Total_Trolleys'], 
                                                 model='additive', period=365)
                
                # Plot decomposition
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.plot(decomposition.trend.dropna(), color='red', linewidth=1)
                ax2.set_title('Trend Component')
                ax2.set_ylabel('Trolleys')
                ax2.grid(True, alpha=0.3)
                
                ax3 = fig.add_subplot(gs[1, 1])
                ax3.plot(decomposition.seasonal[:365], color='green', linewidth=1)
                ax3.set_title('Seasonal Component (First Year)')
                ax3.set_ylabel('Trolleys')
                ax3.grid(True, alpha=0.3)
                
        except ImportError:
            print("⚠️ Seasonal decomposition skipped (statsmodels not available)")
            
        # 3. Monthly Box Plots
        ax4 = fig.add_subplot(gs[2, 0])
        monthly_data = []
        month_labels = []
        for month in range(1, 13):
            month_data = self.data[self.data['Month'] == month]['Total_Trolleys']
            if len(month_data) > 0:
                monthly_data.append(month_data)
                month_labels.append(month)
        
        ax4.boxplot(monthly_data, labels=month_labels)
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Total Trolleys')
        ax4.set_title('Monthly Distribution of Trolley Counts')
        ax4.grid(True, alpha=0.3)
        
        # 4. Day of Week Box Plots
        ax5 = fig.add_subplot(gs[2, 1])
        daily_data = []
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for day in range(7):
            day_data = self.data[self.data['Day_Of_Week'] == day]['Total_Trolleys']
            if len(day_data) > 0:
                daily_data.append(day_data)
        
        ax5.boxplot(daily_data, labels=day_labels)
        ax5.set_xlabel('Day of Week')
        ax5.set_ylabel('Total Trolleys')
        ax5.set_title('Daily Distribution of Trolley Counts')
        ax5.grid(True, alpha=0.3)
        
        # 5. Hourly patterns (if hour data available)
        ax6 = fig.add_subplot(gs[3, 0])
        # Since we don't have hourly data, show seasonal patterns
        seasonal_avg = self.data.groupby('Season')['Total_Trolleys'].mean()
        seasonal_avg.plot(kind='bar', ax=ax6, color=['lightblue', 'lightgreen', 'orange', 'pink'])
        ax6.set_xlabel('Season')
        ax6.set_ylabel('Average Trolleys')
        ax6.set_title('Seasonal Patterns')
        ax6.tick_params(axis='x', rotation=45)
        
        # 6. Year-over-Year Comparison
        ax7 = fig.add_subplot(gs[3, 1])
        yearly_avg = self.data.groupby('Year')['Total_Trolleys'].mean()
        yearly_avg.plot(kind='bar', ax=ax7, color='purple', alpha=0.7)
        ax7.set_xlabel('Year')
        ax7.set_ylabel('Average Trolleys')
        ax7.set_title('Year-over-Year Comparison')
        ax7.tick_params(axis='x', rotation=0)
        
        plt.suptitle('Comprehensive Temporal Analysis', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_regional_analysis(self):
        """Generate comprehensive regional analysis."""
        print("\n🗺️ Generating regional analysis...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, hspace=0.4, wspace=0.3)
        
        # 1. Regional Time Series
        ax1 = fig.add_subplot(gs[0, :])
        for region in self.regional_data['Region'].unique():
            region_data = self.regional_data[self.regional_data['Region'] == region]
            ax1.plot(region_data['Date'], region_data['Total_Trolleys'], 
                    label=region, linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Trolleys')
        ax1.set_title('Regional Time Series Comparison')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Regional Statistics Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        regional_stats = self.regional_data.groupby('Region')['Total_Trolleys'].agg(['mean', 'std', 'max']).round(1)
        regional_stats['mean'].plot(kind='bar', ax=ax2, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Region')
        ax2.set_ylabel('Average Trolleys')
        ax2.set_title('Average Trolleys by Region')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Regional Variability
        ax3 = fig.add_subplot(gs[1, 1])
        regional_stats['std'].plot(kind='bar', ax=ax3, color='lightcoral', alpha=0.7)
        ax3.set_xlabel('Region')
        ax3.set_ylabel('Standard Deviation')
        ax3.set_title('Regional Variability')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Regional Maximum
        ax4 = fig.add_subplot(gs[1, 2])
        regional_stats['max'].plot(kind='bar', ax=ax4, color='lightgreen', alpha=0.7)
        ax4.set_xlabel('Region')
        ax4.set_ylabel('Maximum Trolleys')
        ax4.set_title('Regional Maximum Trolleys')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Regional Box Plot
        ax5 = fig.add_subplot(gs[2, :])
        regions = self.regional_data['Region'].unique()
        regional_data_for_box = [self.regional_data[self.regional_data['Region'] == region]['Total_Trolleys'] 
                                for region in regions]
        ax5.boxplot(regional_data_for_box, labels=regions)
        ax5.set_xlabel('Region')
        ax5.set_ylabel('Total Trolleys')
        ax5.set_title('Regional Distribution Comparison')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. Regional Monthly Heatmap
        ax6 = fig.add_subplot(gs[3, :])
        pivot_data = self.regional_data.groupby(['Region', 'Month'])['Total_Trolleys'].mean().unstack()
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax6)
        ax6.set_xlabel('Month')
        ax6.set_ylabel('Region')
        ax6.set_title('Regional Monthly Average Heatmap')
        
        plt.suptitle('Comprehensive Regional Analysis', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'regional_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_correlation_analysis(self):
        """Generate correlation analysis."""
        print("\n🔗 Generating correlation analysis...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        # Prepare numeric data for correlation
        numeric_cols = ['Total_Trolleys', 'ED_Trolleys', 'Ward_Trolleys', 'Precipitation', 
                       'Month', 'Day_Of_Week', 'Is_Weekend', 'Is_Public_Holiday']
        
        # Filter columns that exist in the data
        available_cols = [col for col in numeric_cols if col in self.data.columns]
        correlation_data = self.data[available_cols]
        
        # 1. Main Correlation Heatmap
        ax1 = fig.add_subplot(gs[0, :])
        correlation_matrix = correlation_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax1)
        ax1.set_title('Feature Correlation Matrix')
        
        # 2. Weather vs Trolleys
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(self.data['Precipitation'], self.data['Total_Trolleys'], 
                   alpha=0.5, color='blue')
        ax2.set_xlabel('Precipitation (mm)')
        ax2.set_ylabel('Total Trolleys')
        ax2.set_title('Precipitation vs Total Trolleys')
        
        # Add correlation coefficient
        corr_coef = self.data['Precipitation'].corr(self.data['Total_Trolleys'])
        ax2.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.grid(True, alpha=0.3)
        
        # 3. ED vs Ward Trolleys
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(self.data['ED_Trolleys'], self.data['Ward_Trolleys'], 
                   alpha=0.5, color='red')
        ax3.set_xlabel('ED Trolleys')
        ax3.set_ylabel('Ward Trolleys')
        ax3.set_title('ED vs Ward Trolleys')
        
        # Add correlation coefficient
        corr_coef_ed_ward = self.data['ED_Trolleys'].corr(self.data['Ward_Trolleys'])
        ax3.text(0.05, 0.95, f'r = {corr_coef_ed_ward:.3f}', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Correlation Analysis', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_distribution_analysis(self):
        """Generate distribution analysis."""
        print("\n📊 Generating distribution analysis...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, hspace=0.4, wspace=0.3)
        
        # 1. Total Trolleys Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self.data['Total_Trolleys'].hist(bins=50, alpha=0.7, color='skyblue', 
                                        density=True, ax=ax1)
        ax1.set_xlabel('Total Trolleys')
        ax1.set_ylabel('Density')
        ax1.set_title('Total Trolleys Distribution')
        ax1.axvline(self.data['Total_Trolleys'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.data["Total_Trolleys"].mean():.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ED Trolleys Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self.data['ED_Trolleys'].hist(bins=50, alpha=0.7, color='lightcoral', 
                                     density=True, ax=ax2)
        ax2.set_xlabel('ED Trolleys')
        ax2.set_ylabel('Density')
        ax2.set_title('ED Trolleys Distribution')
        ax2.axvline(self.data['ED_Trolleys'].mean(), color='red', linestyle='--',
                   label=f'Mean: {self.data["ED_Trolleys"].mean():.1f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Ward Trolleys Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ward_data = self.data['Ward_Trolleys'].dropna()
        if len(ward_data) > 0:
            ward_data.hist(bins=50, alpha=0.7, color='lightgreen', density=True, ax=ax3)
            ax3.axvline(ward_data.mean(), color='red', linestyle='--',
                       label=f'Mean: {ward_data.mean():.1f}')
            ax3.legend()
        ax3.set_xlabel('Ward Trolleys')
        ax3.set_ylabel('Density')
        ax3.set_title('Ward Trolleys Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Precipitation Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        self.data['Precipitation'].hist(bins=50, alpha=0.7, color='blue', 
                                       density=True, ax=ax4)
        ax4.set_xlabel('Precipitation (mm)')
        ax4.set_ylabel('Density')
        ax4.set_title('Precipitation Distribution')
        ax4.axvline(self.data['Precipitation'].mean(), color='red', linestyle='--',
                   label=f'Mean: {self.data["Precipitation"].mean():.1f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Log-transformed Total Trolleys
        ax5 = fig.add_subplot(gs[1, 1])
        log_trolleys = np.log1p(self.data['Total_Trolleys'])
        log_trolleys.hist(bins=50, alpha=0.7, color='purple', density=True, ax=ax5)
        ax5.set_xlabel('Log(Total Trolleys + 1)')
        ax5.set_ylabel('Density')
        ax5.set_title('Log-Transformed Total Trolleys')
        ax5.grid(True, alpha=0.3)
        
        # 6. Q-Q Plot for normality check
        try:
            from scipy import stats
            ax6 = fig.add_subplot(gs[1, 2])
            stats.probplot(self.data['Total_Trolleys'], dist="norm", plot=ax6)
            ax6.set_title('Q-Q Plot: Total Trolleys vs Normal')
            ax6.grid(True, alpha=0.3)
        except ImportError:
            print("⚠️ Q-Q plot skipped (scipy not available)")
            
        # 7. Regional Distribution Comparison
        ax7 = fig.add_subplot(gs[2, :])
        regions = self.data['Region'].unique()[:6]  # Top 6 regions
        for i, region in enumerate(regions):
            region_data = self.data[self.data['Region'] == region]['Total_Trolleys']
            ax7.hist(region_data, bins=30, alpha=0.5, label=region, density=True)
        ax7.set_xlabel('Total Trolleys')
        ax7.set_ylabel('Density')
        ax7.set_title('Distribution Comparison by Region')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle('Distribution Analysis', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_weather_impact_analysis(self):
        """Generate weather impact analysis."""
        print("\n🌦️ Generating weather impact analysis...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Precipitation vs Trolleys Scatter
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(self.data['Precipitation'], self.data['Total_Trolleys'], 
                             c=self.data['Month'], cmap='viridis', alpha=0.6)
        ax1.set_xlabel('Precipitation (mm)')
        ax1.set_ylabel('Total Trolleys')
        ax1.set_title('Precipitation vs Trolleys (colored by month)')
        plt.colorbar(scatter, ax=ax1, label='Month')
        ax1.grid(True, alpha=0.3)
        
        # 2. Precipitation Categories
        ax2 = fig.add_subplot(gs[0, 1])
        if 'Precipitation_Category' in self.data.columns:
            precip_cat_avg = self.data.groupby('Precipitation_Category')['Total_Trolleys'].mean()
            precip_cat_avg.plot(kind='bar', ax=ax2, color='lightblue', alpha=0.7)
            ax2.set_xlabel('Precipitation Category')
            ax2.set_ylabel('Average Trolleys')
            ax2.set_title('Average Trolleys by Precipitation Category')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Seasonal Weather Patterns
        ax3 = fig.add_subplot(gs[1, 0])
        seasonal_weather = self.data.groupby('Season')['Precipitation'].mean()
        seasonal_weather.plot(kind='bar', ax=ax3, color=['lightblue', 'lightgreen', 'orange', 'pink'])
        ax3.set_xlabel('Season')
        ax3.set_ylabel('Average Precipitation (mm)')
        ax3.set_title('Seasonal Precipitation Patterns')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Weather vs Regional Trolleys
        ax4 = fig.add_subplot(gs[1, 1])
        for region in self.regional_data['Region'].unique()[:4]:  # Top 4 regions
            region_data = self.regional_data[self.regional_data['Region'] == region]
            ax4.scatter(region_data['Precipitation'], region_data['Total_Trolleys'], 
                       label=region, alpha=0.6)
        ax4.set_xlabel('Precipitation (mm)')
        ax4.set_ylabel('Total Trolleys')
        ax4.set_title('Weather Impact by Region')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Weather Impact Analysis', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weather_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_capacity_analysis(self):
        """Generate capacity utilization analysis."""
        print("\n🏥 Generating capacity analysis...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. ED vs Ward Trolleys
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(self.data['ED_Trolleys'], self.data['Ward_Trolleys'], 
                   alpha=0.5, color='blue')
        ax1.set_xlabel('ED Trolleys')
        ax1.set_ylabel('Ward Trolleys')
        ax1.set_title('ED vs Ward Trolleys Relationship')
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal line
        max_val = max(self.data['ED_Trolleys'].max(), 
                     self.data['Ward_Trolleys'].max())
        ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal line')
        ax1.legend()
        
        # 2. ED Occupancy Distribution (if available)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'ED_Occupancy_Pct' in self.data.columns:
            occupancy_data = self.data['ED_Occupancy_Pct'].dropna()
            if len(occupancy_data) > 0:
                occupancy_data.hist(bins=30, alpha=0.7, color='orange', ax=ax2)
                ax2.axvline(100, color='red', linestyle='--', linewidth=2, 
                           label='100% Occupancy')
                ax2.legend()
        ax2.set_xlabel('ED Occupancy %')
        ax2.set_ylabel('Frequency')
        ax2.set_title('ED Occupancy Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trolleys by Hospital Type/Size
        ax3 = fig.add_subplot(gs[1, 0])
        hospital_avg = self.data.groupby('Hospital')['Total_Trolleys'].mean().sort_values(ascending=False)
        hospital_avg.head(15).plot(kind='barh', ax=ax3, color='lightcoral')
        ax3.set_xlabel('Average Total Trolleys')
        ax3.set_ylabel('Hospital')
        ax3.set_title('Top 15 Hospitals by Average Trolleys')
        
        # 4. Regional Capacity Utilization
        ax4 = fig.add_subplot(gs[1, 1])
        regional_ed = self.regional_data.groupby('Region')['ED_Trolleys'].mean()
        regional_ward = self.regional_data.groupby('Region')['Ward_Trolleys'].mean()
        
        x = np.arange(len(regional_ed))
        width = 0.35
        
        ax4.bar(x - width/2, regional_ed, width, label='ED Trolleys', color='skyblue')
        ax4.bar(x + width/2, regional_ward, width, label='Ward Trolleys', color='lightcoral')
        
        ax4.set_xlabel('Region')
        ax4.set_ylabel('Average Trolleys')
        ax4.set_title('Regional ED vs Ward Trolley Distribution')
        ax4.set_xticks(x)
        ax4.set_xticklabels(regional_ed.index, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Capacity Utilization Analysis', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'capacity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_summary_dashboard(self):
        """Generate a comprehensive summary dashboard."""
        print("\n📋 Generating summary dashboard...")
        
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, hspace=0.4, wspace=0.3)
        
        # Key metrics
        total_records = len(self.data)
        avg_daily_trolleys = self.data['Total_Trolleys'].mean()
        max_daily_trolleys = self.data['Total_Trolleys'].max()
        total_regions = self.data['Region'].nunique()
        total_hospitals = self.data['Hospital'].nunique()
        date_range = f"{self.data['Date'].min().strftime('%Y-%m-%d')} to {self.data['Date'].max().strftime('%Y-%m-%d')}"
        
        # 1. Key Metrics Text
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('off')
        metrics_text = f"""
        DATASET SUMMARY METRICS
        ========================
        Total Records: {total_records:,}
        Date Range: {date_range}
        Hospitals: {total_hospitals}
        Regions: {total_regions}
        Average Daily Trolleys: {avg_daily_trolleys:.1f}
        Maximum Daily Trolleys: {max_daily_trolleys:.0f}
        """
        ax1.text(0.1, 0.5, metrics_text, transform=ax1.transAxes, fontsize=14,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # 2. Time Series Overview
        ax2 = fig.add_subplot(gs[0, 2:])
        daily_totals = self.data.groupby('Date')['Total_Trolleys'].sum()
        ax2.plot(daily_totals.index, daily_totals.values, color='blue', linewidth=1)
        ax2.set_title('Daily Total Trolleys - Complete Time Series')
        ax2.set_ylabel('Total Trolleys')
        ax2.grid(True, alpha=0.3)
        
        # 3. Regional Comparison
        ax3 = fig.add_subplot(gs[1, :2])
        regional_avg = self.regional_data.groupby('Region')['Total_Trolleys'].mean().sort_values(ascending=False)
        regional_avg.plot(kind='bar', ax=ax3, color='lightcoral')
        ax3.set_title('Average Trolleys by Region')
        ax3.set_ylabel('Average Trolleys')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Monthly Patterns
        ax4 = fig.add_subplot(gs[1, 2:])
        monthly_pattern = self.data.groupby('Month')['Total_Trolleys'].mean()
        monthly_pattern.plot(kind='line', marker='o', ax=ax4, color='green', linewidth=2)
        ax4.set_title('Monthly Seasonal Patterns')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Average Trolleys')
        ax4.grid(True, alpha=0.3)
        
        # 5. Distribution Overview
        ax5 = fig.add_subplot(gs[2, 0])
        self.data['Total_Trolleys'].hist(bins=30, alpha=0.7, color='skyblue', ax=ax5)
        ax5.set_title('Trolley Count Distribution')
        ax5.set_xlabel('Total Trolleys')
        ax5.set_ylabel('Frequency')
        
        # 6. Weather Impact
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.scatter(self.data['Precipitation'], self.data['Total_Trolleys'], alpha=0.5)
        ax6.set_title('Weather vs Trolleys')
        ax6.set_xlabel('Precipitation (mm)')
        ax6.set_ylabel('Total Trolleys')
        
        # 7. Day of Week Patterns
        ax7 = fig.add_subplot(gs[2, 2])
        dow_avg = self.data.groupby('Day_Of_Week')['Total_Trolleys'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_avg.plot(kind='bar', ax=ax7, color='orange')
        ax7.set_title('Day of Week Patterns')
        ax7.set_xticklabels(days, rotation=45)
        ax7.set_ylabel('Average Trolleys')
        
        # 8. Holiday Effect
        ax8 = fig.add_subplot(gs[2, 3])
        holiday_effect = self.data.groupby('Is_Public_Holiday')['Total_Trolleys'].mean()
        holiday_labels = ['Regular Day', 'Public Holiday']
        holiday_effect.plot(kind='bar', ax=ax8, color=['lightgreen', 'gold'])
        ax8.set_title('Holiday Effect')
        ax8.set_xticklabels(holiday_labels, rotation=0)
        ax8.set_ylabel('Average Trolleys')
        
        # 9. Top Hospitals
        ax9 = fig.add_subplot(gs[3, :2])
        top_hospitals = self.data.groupby('Hospital')['Total_Trolleys'].mean().sort_values(ascending=False).head(10)
        top_hospitals.plot(kind='barh', ax=ax9, color='purple', alpha=0.7)
        ax9.set_title('Top 10 Hospitals by Average Trolleys')
        ax9.set_xlabel('Average Trolleys')
        
        # 10. Seasonal Heatmap
        ax10 = fig.add_subplot(gs[3, 2:])
        seasonal_monthly = self.data.groupby(['Season', 'Month'])['Total_Trolleys'].mean().unstack()
        if not seasonal_monthly.empty:
            sns.heatmap(seasonal_monthly, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax10)
            ax10.set_title('Seasonal-Monthly Heatmap')
        
        plt.suptitle('Emergency Department Trolley Demand - Comprehensive EDA Summary', 
                    fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_eda_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def run_complete_analysis(self):
        """Run the complete EDA analysis."""
        print("\n🚀 Starting Comprehensive EDA Analysis")
        print("="*50)
        
        # Load data
        self.load_and_prepare_data()
        
        # Generate all analyses
        self.generate_dataset_overview()
        self.generate_temporal_analysis()
        self.generate_regional_analysis()
        self.generate_correlation_analysis()
        self.generate_distribution_analysis()
        self.generate_weather_impact_analysis()
        self.generate_capacity_analysis()
        self.generate_summary_dashboard()
        
        print(f"\n✅ Complete EDA Analysis Finished!")
        print(f"📁 All figures saved to: {self.output_dir}")
        print("\nGenerated figures:")
        for png_file in sorted(self.output_dir.glob("*.png")):
            print(f"  📊 {png_file.name}")

def main():
    """Main execution function."""
    print("🔍 Comprehensive EDA Generator")
    print("="*50)
    
    # Set paths
    data_path = "/Users/karthik/dissertaion-backup/data/master_trolley_data.csv"
    output_dir = "/Users/karthik/dissertaion-backup/src/results/eda_analysis"
    
    # Create and run EDA
    eda = ComprehensiveEDA(data_path, output_dir)
    eda.run_complete_analysis()
    
    print("\n🎉 Comprehensive EDA Complete!")

if __name__ == "__main__":
    main()

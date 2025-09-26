import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration
CSV_FILENAME = 'bank-additional.csv'  
SAVE_PLOTS = True  

class DataAnalyzer:
    """
    A class to handle data analysis tasks for the assignment
    """
    
    def __init__(self, filename):
        """Initialize the analyzer with a CSV filename"""
        self.filename = filename
        self.df = None
        self.df_cleaned = None
        
    def load_and_explore_data(self):
        """Load CSV data and perform initial exploration"""
        print(" TASK 1: DATA LOADING AND EXPLORATION")
        print("-" * 50)
        
        try:
            print(f"Loading dataset: {self.filename}")
            self.df = pd.read_csv(self.filename)
            
            print(f" Dataset loaded successfully!")
            print(f" Dataset shape: {self.df.shape} (rows, columns)")
            
            # Display first few rows
            print("\n First 5 rows of the dataset:")
            print(self.df.head())
            
            # Display dataset info
            print("\n Dataset Information:")
            self.df.info()
            
            # Check for missing values
            print("\n Missing Values Summary:")
            missing_values = self.df.isnull().sum()
            if missing_values.sum() > 0:
                print(missing_values[missing_values > 0])
            else:
                print("No missing values found! ")
            
            # Display data types
            print("\n Data Types:")
            for dtype in self.df.dtypes.items():
                print(f"  {dtype[0]}: {dtype[1]}")
            
            return True
        
        except FileNotFoundError:
            print(f" Error: File '{self.filename}' not found.")
            print("Please make sure the file exists in the current directory.")
            return False
        except pd.errors.EmptyDataError:
            print(" Error: The file is empty or corrupted.")
            return False
        except Exception as e:
            print(f" Error loading data: {str(e)}")
            return False
    
    def clean_data(self):
        """Clean the dataset by handling missing values"""
        print("\n DATA CLEANING")
        print("-" * 30)
        
        if self.df is None:
            return False
        
        self.df_cleaned = self.df.copy()
        original_shape = self.df_cleaned.shape
        
        # Handle missing values
        missing_count = self.df_cleaned.isnull().sum().sum()
        
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            # For numerical columns, fill with median
            numeric_columns = self.df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if self.df_cleaned[col].isnull().sum() > 0:
                    median_val = self.df_cleaned[col].median()
                    self.df_cleaned[col].fillna(median_val, inplace=True)
                    print(f"  • Filled {col} missing values with median: {median_val:.2f}")
            
            # For categorical columns, fill with mode
            categorical_columns = self.df_cleaned.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if self.df_cleaned[col].isnull().sum() > 0:
                    mode_val = self.df_cleaned[col].mode()[0]
                    self.df_cleaned[col].fillna(mode_val, inplace=True)
                    print(f"  • Filled {col} missing values with mode: {mode_val}")
        else:
            print("No missing values to clean ")
        
        print(f" Data cleaning complete. Shape: {original_shape} → {self.df_cleaned.shape}")
        return True
    
    def perform_basic_analysis(self):
        """Perform basic statistical analysis"""
        print("\n TASK 2: BASIC DATA ANALYSIS")
        print("-" * 50)
        
        if self.df_cleaned is None:
            return
        
        # Basic statistics for numerical columns
        numeric_columns = self.df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            print("\n Basic Statistics (Numerical Columns):")
            print(self.df_cleaned[numeric_columns].describe())
            
            # Save statistics to file
            if SAVE_PLOTS:
                self.df_cleaned[numeric_columns].describe().to_csv('basic_statistics.csv')
                print(" Basic statistics saved to 'basic_statistics.csv'")
        
        # Group analysis
        categorical_columns = self.df_cleaned.select_dtypes(include=['object']).columns
        
        if len(categorical_columns) > 0 and len(numeric_columns) > 0:
            # Use first categorical and first numerical column for grouping
            cat_col = categorical_columns[0]
            num_col = numeric_columns[0]
            
            print(f"\n Group Analysis: Statistics for {num_col} by {cat_col}")
            grouped_analysis = self.df_cleaned.groupby(cat_col)[num_col].agg(['mean', 'count', 'std']).round(2)
            print(grouped_analysis)
            
            # Additional analysis if more columns available
            if len(numeric_columns) > 1:
                print(f"\n Correlation Matrix (Numerical Columns):")
                correlation_matrix = self.df_cleaned[numeric_columns].corr()
                print(correlation_matrix.round(3))
                
                if SAVE_PLOTS:
                    correlation_matrix.to_csv('correlation_matrix.csv')
                    print(" Correlation matrix saved to 'correlation_matrix.csv'")
    
    def create_visualizations(self):
        """Create four different types of visualizations"""
        print("\n TASK 3: DATA VISUALIZATION")
        print("-" * 50)
        
        if self.df_cleaned is None:
            return
        
        # Set style for better-looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Analysis Visualizations', fontsize=18, fontweight='bold', y=0.98)
        
        numeric_columns = self.df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_columns = self.df_cleaned.select_dtypes(include=['object']).columns
        
        # Plot 1: Line Chart (Time Series or Index-based)
        ax1 = axes[0, 0]
        if len(numeric_columns) > 0:
            self._create_line_chart(ax1, numeric_columns)
        
        # Plot 2: Bar Chart (Categorical vs Numerical)
        ax2 = axes[0, 1]
        if len(categorical_columns) > 0 and len(numeric_columns) > 0:
            self._create_bar_chart(ax2, categorical_columns[0], numeric_columns[0])
        
        # Plot 3: Histogram (Distribution)
        ax3 = axes[1, 0]
        if len(numeric_columns) > 0:
            hist_col = numeric_columns[0] if len(numeric_columns) == 1 else numeric_columns[1]
            self._create_histogram(ax3, hist_col)
        
        # Plot 4: Scatter Plot (Correlation between two numerical columns)
        ax4 = axes[1, 1]
        if len(numeric_columns) >= 2:
            self._create_scatter_plot(ax4, numeric_columns[0], numeric_columns[1])
        
        plt.tight_layout()
        
        if SAVE_PLOTS:
            plt.savefig('data_visualizations.png', dpi=300, bbox_inches='tight')
            print(" Visualizations saved to 'data_visualizations.png'")
        
        plt.show()
        
        print(" All four visualization types created successfully!")
    
    def _create_line_chart(self, ax, numeric_columns):
        """Create line chart visualization"""
        # Check for date columns
        date_columns = [col for col in self.df_cleaned.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_columns:
            try:
                self.df_cleaned[date_columns[0]] = pd.to_datetime(self.df_cleaned[date_columns[0]])
                df_sorted = self.df_cleaned.sort_values(date_columns[0])
                ax.plot(df_sorted[date_columns[0]], df_sorted[numeric_columns[0]], 
                       marker='o', linewidth=2, markersize=4)
                ax.set_xlabel(date_columns[0])
                ax.tick_params(axis='x', rotation=45)
            except:
                ax.plot(self.df_cleaned.index, self.df_cleaned[numeric_columns[0]], 
                       marker='o', linewidth=2, markersize=4)
                ax.set_xlabel('Index')
        else:
            ax.plot(self.df_cleaned.index, self.df_cleaned[numeric_columns[0]], 
                   marker='o', linewidth=2, markersize=4)
            ax.set_xlabel('Index')
        
        ax.set_ylabel(numeric_columns[0])
        ax.set_title(f'Line Chart: {numeric_columns[0]} Trend', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _create_bar_chart(self, ax, cat_col, num_col):
        """Create bar chart visualization"""
        grouped_data = self.df_cleaned.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
        colors = plt.cm.viridis(np.linspace(0, 1, len(grouped_data)))
        bars = ax.bar(range(len(grouped_data)), grouped_data.values, color=colors)
        
        ax.set_xlabel(cat_col)
        ax.set_ylabel(f'Average {num_col}')
        ax.set_title(f'Bar Chart: Average {num_col} by {cat_col}', fontweight='bold')
        ax.set_xticks(range(len(grouped_data)))
        ax.set_xticklabels(grouped_data.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    def _create_histogram(self, ax, col):
        """Create histogram visualization"""
        data = self.df_cleaned[col].dropna()
        n, bins, patches = ax.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram: Distribution of {col}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend()
    
    def _create_scatter_plot(self, ax, x_col, y_col):
        """Create scatter plot visualization"""
        x_data = self.df_cleaned[x_col].dropna()
        y_data = self.df_cleaned[y_col].dropna()
        
        # Ensure we have matching data points
        common_index = x_data.index.intersection(y_data.index)
        x_data = x_data.loc[common_index]
        y_data = y_data.loc[common_index]
        
        scatter = ax.scatter(x_data, y_data, alpha=0.6, s=50, c=plt.cm.viridis(0.5))
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'Scatter Plot: {x_col} vs {y_col}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = x_data.corr(y_data)
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Add trend line
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2, label='Trend line')
            ax.legend()
    
    def generate_insights(self):
        """Generate insights and findings from the analysis"""
        print("\n FINDINGS AND INSIGHTS")
        print("-" * 50)
        
        if self.df_cleaned is None:
            return
        
        insights = []
        
        # Dataset overview insights
        insights.append(f" Dataset contains {self.df_cleaned.shape[0]} rows and {self.df_cleaned.shape[1]} columns")
        
        # Missing values insight
        original_missing = self.df.isnull().sum().sum() if self.df is not None else 0
        missing_percentage = (original_missing / (self.df.shape[0] * self.df.shape[1])) * 100 if self.df is not None else 0
        insights.append(f" Original missing values: {missing_percentage:.1f}% of total data (cleaned)")
        
        # Numerical columns insights
        numeric_columns = self.df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            # Find columns with highest/lowest variance
            variances = self.df_cleaned[numeric_columns].var().sort_values(ascending=False)
            insights.append(f" Most variable numerical feature: {variances.index[0]} (variance: {variances.iloc[0]:.2f})")
            
            if len(variances) > 1:
                insights.append(f" Least variable numerical feature: {variances.index[-1]} (variance: {variances.iloc[-1]:.2f})")
            
            # Statistical insights
            for col in numeric_columns[:3]:  # Limit to first 3 columns
                col_data = self.df_cleaned[col]
                insights.append(f" {col}: Mean={col_data.mean():.2f}, Std={col_data.std():.2f}, Range=[{col_data.min():.2f}, {col_data.max():.2f}]")
            
            # Correlation insights
            if len(numeric_columns) >= 2:
                corr_matrix = self.df_cleaned[numeric_columns].corr()
                corr_matrix_no_diag = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))
                
                if not corr_matrix_no_diag.isna().all().all():
                    max_corr_idx = corr_matrix_no_diag.abs().stack().idxmax()
                    max_corr_val = corr_matrix.loc[max_corr_idx]
                    insights.append(f" Strongest correlation: {max_corr_idx[0]} & {max_corr_idx[1]} (r = {max_corr_val:.3f})")
        
        # Categorical columns insights
        categorical_columns = self.df_cleaned.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            for col in categorical_columns[:3]:  # Limit to first 3 categorical columns
                unique_count = self.df_cleaned[col].nunique()
                most_common = self.df_cleaned[col].mode()[0]
                most_common_count = self.df_cleaned[col].value_counts().iloc[0]
                percentage = (most_common_count / len(self.df_cleaned)) * 100
                insights.append(f" {col}: {unique_count} unique values, most common: '{most_common}' ({percentage:.1f}%)")
        
        # Data quality insights
        if len(numeric_columns) > 0:
            # Check for potential outliers using IQR method
            outlier_counts = {}
            for col in numeric_columns:
                Q1 = self.df_cleaned[col].quantile(0.25)
                Q3 = self.df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df_cleaned[(self.df_cleaned[col] < lower_bound) | (self.df_cleaned[col] > upper_bound)][col]
                outlier_counts[col] = len(outliers)
            
            max_outliers = max(outlier_counts.values())
            if max_outliers > 0:
                max_outlier_col = max(outlier_counts, key=outlier_counts.get)
                insights.append(f" Potential outliers detected: {max_outliers} in {max_outlier_col}")
        
        # Print all insights
        print("\n KEY FINDINGS:")
        for i, insight in enumerate(insights, 1):
            print(f"{i:2d}. {insight}")
        
        # Save insights to file
        if SAVE_PLOTS:
            with open('analysis_insights.txt', 'w') as f:
                f.write("Data Analysis Insights\n")
                f.write("=" * 50 + "\n")
                for i, insight in enumerate(insights, 1):
                    f.write(f"{i:2d}. {insight}\n")
            print(f"\n Insights saved to 'analysis_insights.txt'")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print(" STARTING COMPLETE DATA ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load and explore
        if not self.load_and_explore_data():
            return False
        
        # Step 2: Clean data
        if not self.clean_data():
            return False
        
        # Step 3: Basic analysis
        self.perform_basic_analysis()
        
        # Step 4: Visualizations
        self.create_visualizations()
        
        # Step 5: Generate insights
        self.generate_insights()
        
        # Summary
        print("\n" + "=" * 60)
        print(" ANALYSIS COMPLETE!")
        print(" Summary of deliverables:")
        print("   • Data loading and exploration ")
        print("   • Data cleaning ")
        print("   • Basic statistical analysis ") 
        print("   • Four types of visualizations ")
        print("   • Key insights and findings ")
        
        if SAVE_PLOTS:
            print("\n Files created:")
            print("   • data_visualizations.png")
            print("   • basic_statistics.csv")
            print("   • correlation_matrix.csv")
            print("   • analysis_insights.txt")
        
        print("=" * 60)
        return True

def main():
    """Main function to run the analysis"""
    print("🎓 PANDAS & MATPLOTLIB DATA ANALYSIS ASSIGNMENT")
    print("=" * 60)
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILENAME):
        print(f" CSV file '{CSV_FILENAME}' not found!")
        print("\n To use your dataset:")
        print(f"1. Place your CSV file in the current directory")
        print(f"2. Update CSV_FILENAME variable at the top of this script")
        print(f"3. Run the script again")
        return
    
    # Create analyzer and run analysis
    analyzer = DataAnalyzer(CSV_FILENAME)
    success = analyzer.run_complete_analysis()
    
    if not success:
        print("\n Analysis failed. Please check your CSV file and try again.")
        print("\n Common issues:")
        print("• File path is incorrect")
        print("• File is corrupted or empty")
        print("• File encoding issues (try UTF-8)")
        print("• Insufficient permissions")

if __name__ == "__main__":
    main()
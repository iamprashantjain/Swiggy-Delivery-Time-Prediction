import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def univariate_analysis_numerical(df, numerical_cols=None, figsize=(15, 10)):
    """
    Perform univariate analysis on numerical columns
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe
    numerical_cols : list, optional
        List of numerical column names. If None, automatically detects numerical columns
    figsize : tuple
        Figure size for plots
    
    Returns:
    --------
    summary_df : pandas DataFrame
        Summary statistics including skewness and outlier information
    """
    
    # Auto-detect numerical columns if not provided
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("="*80)
    print("UNIVARIATE ANALYSIS FOR NUMERICAL COLUMNS")
    print("="*80)
    
    # Storage for summary statistics
    summary_list = []
    
    for col in numerical_cols:
        print(f"\n{'='*50}")
        print(f"ANALYZING: {col}")
        print('='*50)
        
        # 1. Basic statistics using describe()
        print("\n1. DESCRIPTIVE STATISTICS:")
        desc_stats = df[col].describe()
        print(desc_stats)
        
        # Calculate additional statistics
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        # 2. Check for skewness
        skewness = df[col].skew()
        print(f"\n2. SKEWNESS: {skewness:.3f}")
        
        if skewness > 1:
            print("   → Highly right-skewed (positive skew)")
        elif skewness > 0.5:
            print("   → Moderately right-skewed")
        elif skewness < -1:
            print("   → Highly left-skewed (negative skew)")
        elif skewness < -0.5:
            print("   → Moderately left-skewed")
        else:
            print("   → Approximately symmetric")
        
        # 3. Outlier detection using IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        print(f"\n3. OUTLIER DETECTION (IQR method):")
        print(f"   - Lower bound: {lower_bound:.2f}")
        print(f"   - Upper bound: {upper_bound:.2f}")
        print(f"   - Number of outliers: {outlier_count} ({outlier_pct:.2f}%)")
        
        # Identify outlier values
        if outlier_count > 0:
            outlier_values = outliers[col].tolist()
            print(f"   - Outlier values: {sorted(outlier_values)[:10]}")  # First 10 outliers
            
            # Check if outliers are due to data entry errors (e.g., negative ages)
            if col.lower().find('age') != -1 and (df[col] < 0).any():
                print("   → WARNING: Negative age values detected - likely data entry error")
            elif col.lower().find('rating') != -1 and (df[col] > 5).any():
                print("   → WARNING: Ratings > 5 detected - likely data entry error")
            elif col.lower().find('time') != -1 and (df[col] < 0).any():
                print("   → WARNING: Negative time values detected - likely data entry error")
        
        # 4. Visualization
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Univariate Analysis: {col}', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0, 0].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(df[col].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df[col].mean():.2f}')
        axes[0, 0].axvline(df[col].median(), color='green', linestyle='dashed', linewidth=2, label=f'Median: {df[col].median():.2f}')
        axes[0, 0].set_xlabel(col)
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Histogram')
        axes[0, 0].legend()
        
        # Box plot
        axes[0, 1].boxplot(df[col].dropna(), vert=True)
        axes[0, 1].set_ylabel(col)
        axes[0, 1].set_title('Box Plot (Outliers Visualized)')
        
        # Density plot
        df[col].dropna().plot(kind='density', ax=axes[0, 2])
        axes[0, 2].set_xlabel(col)
        axes[0, 2].set_title('Density Plot')
        axes[0, 2].axvline(df[col].mean(), color='red', linestyle='dashed', linewidth=2)
        axes[0, 2].axvline(df[col].median(), color='green', linestyle='dashed', linewidth=2)
        
        # Q-Q plot for normality check
        stats.probplot(df[col].dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)')
        
        # Violin plot
        axes[1, 1].violinplot(df[col].dropna(), vert=True)
        axes[1, 1].set_ylabel(col)
        axes[1, 1].set_title('Violin Plot')
        
        # Distribution with KDE
        sns.histplot(data=df, x=col, kde=True, ax=axes[1, 2])
        axes[1, 2].set_title('Distribution with KDE')
        
        plt.tight_layout()
        plt.show()
        
        # 5. Recommendations for transformation
        print(f"\n4. RECOMMENDATIONS:")
        if abs(skewness) > 1:
            print(f"   → High skewness detected ({skewness:.2f}). Consider:")
            print("      - Log transformation: np.log1p(df[col])")
            print("      - Square root transformation: np.sqrt(df[col])")
            print("      - Box-Cox transformation: scipy.stats.boxcox(df[col])")
        else:
            print("   → Skewness is within acceptable range. No transformation needed.")
        
        if outlier_pct > 10:
            print(f"   → High outlier percentage ({outlier_pct:.1f}%). Consider:")
            print("      - Capping/Winsorizing outliers")
            print("      - Using median instead of mean for central tendency")
            print("      - Using robust statistical methods")
        elif outlier_pct > 0:
            print(f"   → {outlier_pct:.1f}% outliers present. Consider:")
            print("      - Keeping if they represent real extreme values")
            print("      - Removing if they are data entry errors")
            print("      - Using robust metrics (median, IQR)")
        
        # Store summary
        summary_list.append({
            'Column': col,
            'Count': desc_stats['count'],
            'Mean': desc_stats['mean'],
            'Std': desc_stats['std'],
            'Min': desc_stats['min'],
            '25%': desc_stats['25%'],
            '50% (Median)': desc_stats['50%'],
            '75%': desc_stats['75%'],
            'Max': desc_stats['max'],
            'Skewness': skewness,
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': outlier_pct,
            'Missing_Count': missing_count,
            'Missing_Percentage': missing_pct
        })
        
        print("\n" + "-"*50)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_list)
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY OF ALL NUMERICAL COLUMNS")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # 5. Conclusion summary
    print("\n" + "="*80)
    print("CONCLUSION & RECOMMENDATIONS")
    print("="*80)
    
    print("\nKey Findings:")
    for col in numerical_cols:
        skew_val = summary_df[summary_df['Column'] == col]['Skewness'].values[0]
        outlier_pct = summary_df[summary_df['Column'] == col]['Outlier_Percentage'].values[0]
        
        if abs(skew_val) > 1:
            print(f"  • {col}: Highly skewed ({skew_val:.2f})")
        if outlier_pct > 10:
            print(f"  • {col}: Has high outliers ({outlier_pct:.1f}%)")
    
    print("\nOverall Recommendations:")
    print("  1. For skewed features: Consider log or Box-Cox transformation")
    print("  2. For outliers: Use robust statistics (median, IQR)")
    print("  3. For ML models: Consider tree-based models (less sensitive to skewness/outliers)")
    print("  4. Always document any transformations applied for reproducibility")
    
    return summary_df




def univariate_analysis_categorical(df, categorical_cols=None, figsize=(15, 8)):
    """
    Perform univariate analysis on categorical columns
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe
    categorical_cols : list, optional
        List of categorical column names. If None, automatically detects object/category columns
    figsize : tuple
        Figure size for plots
    
    Returns:
    --------
    summary_df : pandas DataFrame
        Summary statistics for categorical columns
    """
    
    # Auto-detect categorical columns if not provided
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Also include binary/boolean columns
    binary_cols = [col for col in df.columns if df[col].nunique() <= 10 and df[col].dtype in ['int64', 'float64']]
    categorical_cols.extend(binary_cols)
    categorical_cols = list(set(categorical_cols))  # Remove duplicates
    
    print("="*80)
    print("UNIVARIATE ANALYSIS FOR CATEGORICAL COLUMNS")
    print("="*80)
    print(f"\nAnalyzing {len(categorical_cols)} categorical columns:")
    print(", ".join(categorical_cols))
    
    # Storage for summary statistics
    summary_list = []
    
    for col in categorical_cols:
        print(f"\n{'='*50}")
        print(f"ANALYZING: {col}")
        print('='*50)
        
        # 1. Basic descriptive statistics
        print("\n1. DESCRIPTIVE STATISTICS:")
        print(f"   - Unique values: {df[col].nunique()}")
        print(f"   - Missing values: {df[col].isna().sum()} ({df[col].isna().sum()/len(df)*100:.2f}%)")
        print(f"   - Data type: {df[col].dtype}")
        
        # 2. Frequency distribution
        print("\n2. FREQUENCY DISTRIBUTION:")
        freq_dist = df[col].value_counts()
        freq_dist_pct = df[col].value_counts(normalize=True) * 100
        
        # Create frequency dataframe
        freq_df = pd.DataFrame({
            'Category': freq_dist.index,
            'Count': freq_dist.values,
            'Percentage': freq_dist_pct.values
        })
        print(freq_df.to_string(index=False))
        
        # 3. Check for anomalies
        print("\n3. ANOMALIES/INSIGHTS:")
        
        # Check for rare categories (<1% of data)
        rare_categories = freq_df[freq_df['Percentage'] < 1]
        if len(rare_categories) > 0:
            print(f"   ⚠️ Rare categories (<1%): {len(rare_categories)} categories")
            print(f"      {rare_categories['Category'].tolist()}")
        
        # Check for dominant categories (>50% of data)
        dominant = freq_df[freq_df['Percentage'] > 50]
        if len(dominant) > 0:
            print(f"   ⭐ Dominant category (>50%): {dominant['Category'].iloc[0]} ({dominant['Percentage'].iloc[0]:.1f}%)")
        
        # Check for high cardinality
        if df[col].nunique() > 20:
            print(f"   🔥 High cardinality: {df[col].nunique()} unique values - may need feature engineering")
        
        # Check for typos or inconsistent formatting
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            # Check for case variations
            lower_vals = [str(v).lower() for v in unique_vals]
            if len(set(lower_vals)) < len(unique_vals):
                print(f"   ⚠️ Case inconsistency detected - consider standardizing to lower/upper case")
        
        # 4. Visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Categorical Analysis: {col}', fontsize=14, fontweight='bold')
        
        # Count plot (horizontal bar chart)
        top_categories = freq_dist.head(10)  # Show top 10 if many categories
        axes[0, 0].barh(range(len(top_categories)), top_categories.values, color='skyblue')
        axes[0, 0].set_yticks(range(len(top_categories)))
        axes[0, 0].set_yticklabels(top_categories.index)
        axes[0, 0].set_xlabel('Count')
        axes[0, 0].set_title('Category Distribution (Top 10)')
        
        # Pie chart (only if few categories, otherwise bar chart is better)
        if len(freq_dist) <= 10:
            colors = plt.cm.Set3(range(len(freq_dist)))
            wedges, texts, autotexts = axes[0, 1].pie(freq_dist.values, 
                                                        labels=freq_dist.index, 
                                                        autopct='%1.1f%%',
                                                        colors=colors,
                                                        textprops={'fontsize': 8})
            axes[0, 1].set_title('Category Distribution (Pie Chart)')
        else:
            # Show bar chart instead for many categories
            axes[0, 1].bar(freq_dist.index[:10], freq_dist.values[:10], color='coral')
            axes[0, 1].set_xlabel('Category')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Top 10 Categories')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Percentage bar chart
        axes[1, 0].bar(range(len(freq_dist)), freq_dist_pct.values, color='lightgreen')
        axes[1, 0].set_xlabel('Category')
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].set_title('Percentage Distribution')
        axes[1, 0].set_xticks(range(len(freq_dist)))
        axes[1, 0].set_xticklabels(freq_dist.index, rotation=45, ha='right')
        
        # Missing values visualization
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_data = pd.DataFrame({
                'Status': ['Present', 'Missing'],
                'Count': [len(df) - missing_count, missing_count]
            })
            axes[1, 1].bar(missing_data['Status'], missing_data['Count'], color=['#2ecc71', '#e74c3c'])
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title(f'Missing Values (Total: {missing_count})')
            axes[1, 1].text(0, len(df) - missing_count + 10, f"{((len(df)-missing_count)/len(df)*100):.1f}%", 
                           ha='center', fontweight='bold')
            axes[1, 1].text(1, missing_count + 10, f"{missing_count/len(df)*100:.1f}%", 
                           ha='center', fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('Missing Values: None')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 5. Recommendations based on findings
        print("\n4. RECOMMENDATIONS:")
        
        if missing_count > 0:
            print(f"   • Missing values: {missing_count} ({missing_count/len(df)*100:.1f}%)")
            if missing_count/len(df) < 5:
                print("     → Consider dropping missing rows")
            else:
                print("     → Consider imputing with mode or creating 'Unknown' category")
        
        if len(freq_dist) > 10:
            print(f"   • High cardinality ({len(freq_dist)} categories):")
            print("     → Consider grouping rare categories into 'Other'")
            print("     → Use frequency or target encoding instead of one-hot encoding")
        
        if len(rare_categories) > 0:
            print(f"   • Rare categories detected ({len(rare_categories)} categories with <1%):")
            print("     → Consider combining rare categories into 'Other' category")
        
        # Store summary
        summary_list.append({
            'Column': col,
            'Data_Type': str(df[col].dtype),
            'Unique_Values': df[col].nunique(),
            'Most_Frequent': freq_dist.index[0] if len(freq_dist) > 0 else 'N/A',
            'Most_Freq_Count': freq_dist.values[0] if len(freq_dist) > 0 else 0,
            'Most_Freq_Pct': freq_dist_pct.values[0] if len(freq_dist_pct) > 0 else 0,
            'Missing_Count': missing_count,
            'Missing_Pct': missing_count/len(df)*100,
            'Has_Rare_Categories': len(rare_categories) > 0,
            'Cardinality_Issue': df[col].nunique() > 10
        })
        
        print("\n" + "-"*50)
    
    # Summary dataframe
    summary_df = pd.DataFrame(summary_list)
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY OF CATEGORICAL COLUMNS")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Overall recommendations
    print("\n" + "="*80)
    print("CONCLUSION & RECOMMENDATIONS")
    print("="*80)
    
    high_cardinality_cols = summary_df[summary_df['Cardinality_Issue'] == True]['Column'].tolist()
    high_missing_cols = summary_df[summary_df['Missing_Pct'] > 5]['Column'].tolist()
    
    if high_cardinality_cols:
        print(f"\n• High cardinality columns: {', '.join(high_cardinality_cols)}")
        print("  → Use frequency encoding, target encoding, or feature hashing")
    
    if high_missing_cols:
        print(f"\n• High missing value columns: {', '.join(high_missing_cols)}")
        print("  → Impute or create an 'Unknown' category")
    
    print("\n• For categorical variables in modeling:")
    print("  → Tree-based models: Label encoding is sufficient")
    print("  → Linear models: Use one-hot encoding (for low cardinality)")
    print("  → High cardinality features: Consider target encoding")
    
    return summary_df



def bivariate_analysis(df, num_col, cat_col, target_col):
    """
    Clean and interpretable bivariate analysis with target column
    """
    
    print("="*80)
    print(f"🎯 BIVARIATE ANALYSIS WITH TARGET: {target_col}")
    print("="*80)
    
    # ============================================
    # PART 1: NUMERICAL vs TARGET (Numerical-Numerical)
    # ============================================
    print("\n" + "="*80)
    print("📈 NUMERICAL FEATURES vs TARGET")
    print("="*80)
    
    for col in num_col:
        if col != target_col:
            print(f"\n{'─'*80}")
            print(f"▶ Analyzing: {col} → Impact on {target_col}")
            print(f"{'─'*80}")
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Scatter plot with trend line
            sns.regplot(data=df, x=col, y=target_col, 
                       scatter_kws={'alpha':0.5, 's':30}, 
                       line_kws={'color':'red', 'linewidth':2},
                       ax=ax1)
            ax1.set_title(f'Relationship: {col} vs {target_col}', fontsize=12, fontweight='bold')
            ax1.set_xlabel(col, fontsize=10)
            ax1.set_ylabel(target_col, fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Correlation info as text
            corr = df[[col, target_col]].corr().iloc[0, 1]
            
            # Determine relationship strength
            if abs(corr) > 0.7:
                strength = "Strong"
                direction = "Positive" if corr > 0 else "Negative"
            elif abs(corr) > 0.3:
                strength = "Moderate"
                direction = "Positive" if corr > 0 else "Negative"
            elif abs(corr) > 0.1:
                strength = "Weak"
                direction = "Positive" if corr > 0 else "Negative"
            else:
                strength = "Very Weak/No"
                direction = "No clear direction"
            
            # Create text summary
            ax2.text(0.1, 0.8, f"CORRELATION ANALYSIS", 
                    fontsize=14, fontweight='bold', transform=ax2.transAxes)
            ax2.text(0.1, 0.65, f"Correlation Coefficient: {corr:.4f}", 
                    fontsize=12, transform=ax2.transAxes)
            ax2.text(0.1, 0.5, f"Relationship Strength: {strength}", 
                    fontsize=12, transform=ax2.transAxes)
            ax2.text(0.1, 0.35, f"Direction: {direction}", 
                    fontsize=12, transform=ax2.transAxes)
            
            # Add interpretation
            if abs(corr) > 0.3:
                if corr > 0:
                    interpretation = f"📈 As {col} increases,\n   {target_col} tends to increase"
                else:
                    interpretation = f"📉 As {col} increases,\n   {target_col} tends to decrease"
            else:
                interpretation = f"⚠️ No clear linear relationship\n   between {col} and {target_col}"
            
            ax2.text(0.1, 0.15, interpretation, 
                    fontsize=11, style='italic', transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Print numerical summary
            print(f"\n📊 Statistics:")
            print(f"   • {col} - Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
            print(f"   • {target_col} - Mean: {df[target_col].mean():.2f}, Std: {df[target_col].std():.2f}")
    
    # ============================================
    # PART 2: CATEGORICAL vs TARGET (Categorical-Numerical)
    # ============================================
    print("\n" + "="*80)
    print("📊 CATEGORICAL FEATURES vs TARGET")
    print("="*80)
    
    for col in cat_col:
        print(f"\n{'─'*80}")
        print(f"▶ Analyzing: {col} → Impact on {target_col}")
        print(f"{'─'*80}")
        
        # Prepare data
        grouped_stats = df.groupby(col)[target_col].agg(['mean', 'median', 'count', 'std']).round(2)
        grouped_stats = grouped_stats.sort_values('mean', ascending=False)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Bar plot (mean values with error bars)
        bars = ax1.bar(range(len(grouped_stats)), grouped_stats['mean'], 
                       color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(grouped_stats)))
        ax1.set_xticklabels(grouped_stats.index, rotation=45, ha='right')
        ax1.set_title(f'Average {target_col} by {col}', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'Mean {target_col}', fontsize=10)
        ax1.set_xlabel(col, fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Box plot
        df.boxplot(column=target_col, by=col, ax=ax2, rot=45)
        ax2.set_title(f'Distribution of {target_col} by {col}', fontsize=12, fontweight='bold')
        ax2.set_xlabel(col, fontsize=10)
        ax2.set_ylabel(target_col, fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('')  # Remove automatic title
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n📊 Summary Statistics (sorted by mean {target_col}):")
        print(grouped_stats)
        
        # Find best and worst categories
        best_category = grouped_stats.index[0]
        worst_category = grouped_stats.index[-1]
        print(f"\n💡 Key Insight:")
        print(f"   • Best category: '{best_category}' (Avg {target_col}: {grouped_stats.loc[best_category, 'mean']:.2f})")
        print(f"   • Worst category: '{worst_category}' (Avg {target_col}: {grouped_stats.loc[worst_category, 'mean']:.2f})")
        print(f"   • Difference: {grouped_stats.loc[best_category, 'mean'] - grouped_stats.loc[worst_category, 'mean']:.2f}")
        
        # ANOVA test
        from scipy.stats import f_oneway
        groups = [df[df[col] == cat][target_col].dropna().values for cat in df[col].unique()]
        if len(groups) > 1:
            f_stat, p_val = f_oneway(*groups)
            print(f"\n📈 Statistical Significance (ANOVA):")
            print(f"   • F-statistic: {f_stat:.4f}")
            print(f"   • p-value: {p_val:.4f}")
            if p_val < 0.05:
                print(f"   • ✅ {col} has SIGNIFICANT impact on {target_col} (p < 0.05)")
            else:
                print(f"   • ❌ {col} does NOT have significant impact on {target_col}")



def quick_multivariate_eda(df, num_cols, cat_cols, target_col):
    """Quick and simple multivariate analysis"""
    
    print(f"\n{'='*50}")
    print(f"Multivariate EDA - Target: {target_col}")
    print(f"{'='*50}")
    
    # Numerical features
    if num_cols:
        print("\n📈 Numerical Feature Correlations:")
        corr_data = {}
        for col in num_cols:
            if col in df.columns:
                corr = df[col].corr(df[target_col])
                corr_data[col] = corr
                print(f"  {col:20s}: {corr:+.3f}")
        
        # Plot
        fig, axes = plt.subplots(1, min(3, len(num_cols)), figsize=(15, 4))
        if len(num_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(list(corr_data.keys())[:3]):
            axes[i].scatter(df[col], df[target_col], alpha=0.5)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(target_col)
            axes[i].set_title(f'Corr: {corr_data[col]:.3f}')
        
        plt.tight_layout()
        plt.show()
    
    # Categorical features
    if cat_cols:
        print("\n📊 Categorical Feature Impact:")
        for col in cat_cols[:3]:
            if col in df.columns:
                print(f"\n  → {col}:")
                means = df.groupby(col)[target_col].mean().sort_values(ascending=False)
                for cat, val in means.head(3).items():
                    print(f"      {cat}: {val:.2f}")
    
    print("\n✅ Analysis complete!")

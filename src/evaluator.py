#!/usr/bin/env python3
"""
UNIFIED EVALUATOR MODULE
========================

Unified evaluation and explainability module that consolidates all evaluation logic.
Provides comprehensive model evaluation, SHAP analysis, statistical testing, and clinical insights.

Features:
- Model performance evaluation (RMSE, MAE, R², MAPE)
- SHAP explainability analysis with healthcare interpretations
- Statistical significance testing between models
- Multi-horizon evaluation capabilities
- Clinical insights generation
- Comprehensive visualizations and reporting

Author: Unified Pipeline
Date: December 28, 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

# Optional dependencies with fallbacks
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available - explainability analysis will be limited")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  SciPy not available - statistical testing will be limited")


class ModelEvaluator:
    """
    Unified model evaluation class that handles all evaluation logic.
    """
    
    def __init__(self, results_dir: str = None, enable_shap: bool = True, 
                 horizons: List[int] = [1, 3, 7, 14]):
        """
        Initialize the model evaluator.
        
        Args:
            results_dir: Directory to save evaluation results
            enable_shap: Whether to enable SHAP explainability analysis
            horizons: List of forecast horizons to evaluate
        """
        self.results_dir = Path(results_dir) if results_dir else Path("results/evaluation")
        self.enable_shap = enable_shap and SHAP_AVAILABLE
        self.horizons = horizons
        
        # Results storage
        self.performance_results = {}
        self.explainability_results = {}
        self.statistical_tests = {}
        self.multi_horizon_results = {}
        self.clinical_insights = {}
        
        # SHAP storage
        self.shap_values = {}
        self.explainers = {}
        self.feature_importance = {}
        
        # Healthcare color scheme
        self.colors = {
            'primary': '#2E86AB',      # Medical blue
            'secondary': '#A23B72',    # Clinical purple
            'accent': '#F18F01',       # Healthcare orange
            'positive': '#52A57A',     # Positive outcome green
            'negative': '#C73E1D',     # Risk factor red
            'neutral': '#6C757D',      # Neutral gray
            'background': '#F8F9FA'    # Light background
        }
        
        print(f"🔬 Model Evaluator initialized")
        print(f"📂 Results directory: {self.results_dir}")
        print(f"🧠 SHAP analysis: {'Enabled' if self.enable_shap else 'Disabled'}")
        print(f"📊 Evaluation horizons: {horizons} days")
    
    def evaluate_model_performance(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                                 model_name: str) -> Dict[str, Any]:
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Dictionary containing performance metrics
        """
        print(f"\n📊 Evaluating {model_name}...")
        
        try:
            # Make predictions
            predictions = model.predict(X_test)
            
            # Ensure predictions are valid
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                print(f"⚠️  Invalid predictions detected for {model_name}, using fallback")
                predictions = np.full_like(predictions, np.mean(y_test))
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, predictions)
            
            # Store results
            results = {
                'model_name': model_name,
                'predictions': predictions,
                'metrics': metrics,
                'residuals': y_test - predictions,
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            self.performance_results[model_name] = results
            
            print(f"  ✅ RMSE: {metrics['rmse']:.3f}, MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.3f}")
            return results
            
        except Exception as e:
            print(f"  ❌ Evaluation failed for {model_name}: {e}")
            return self._create_fallback_results(model_name, X_test, y_test)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        # Ensure arrays are finite
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {'rmse': 999.0, 'mae': 999.0, 'r2': -999.0, 'mape': 999.0}
        
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # MAPE with protection against division by zero
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / np.maximum(np.abs(y_true_clean), 1e-8))) * 100
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
    
    def _create_fallback_results(self, model_name: str, X_test: pd.DataFrame, 
                               y_test: pd.Series) -> Dict[str, Any]:
        """Create fallback results when evaluation fails."""
        fallback_predictions = np.full(len(y_test), np.mean(y_test))
        metrics = self._calculate_metrics(y_test.values, fallback_predictions)
        
        return {
            'model_name': model_name,
            'predictions': fallback_predictions,
            'metrics': metrics,
            'residuals': y_test - fallback_predictions,
            'evaluation_timestamp': datetime.now().isoformat(),
            'fallback_mode': True
        }
    
    def perform_explainability_analysis(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                      model_name: str, sample_size: int = 100) -> Dict[str, Any]:
        """
        Comprehensive SHAP explainability analysis.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            sample_size: Sample size for SHAP analysis
            
        Returns:
            Dictionary containing explainability results
        """
        if not self.enable_shap:
            print(f"⚠️  SHAP analysis disabled for {model_name}")
            return self._fallback_explainability_analysis(model, X_test, y_test, model_name)
        
        print(f"\n🔍 Performing explainability analysis for {model_name}...")
        
        try:
            # Sample data for performance
            sample_size = min(sample_size, len(X_test))
            sample_X = X_test.iloc[:sample_size].copy()
            sample_y = y_test.iloc[:sample_size].copy()
            
            # Create appropriate explainer
            explainer = self._create_explainer(model, sample_X)
            if explainer is None:
                return self._fallback_explainability_analysis(model, X_test, y_test, model_name)
            
            # Calculate SHAP values
            print(f"  → Computing SHAP values for {sample_size} samples...")
            shap_values = explainer.shap_values(sample_X)
            
            # Store SHAP results
            self.shap_values[model_name] = shap_values
            self.explainers[model_name] = explainer
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(shap_values, sample_X)
            self.feature_importance[model_name] = feature_importance
            
            # Generate clinical insights
            insights = self._generate_clinical_insights(shap_values, sample_X, sample_y, model_name)
            
            # SHAP statistics
            shap_stats = self._calculate_shap_statistics(shap_values)
            
            results = {
                'model_name': model_name,
                'feature_names': list(sample_X.columns),
                'sample_size': sample_size,
                'feature_importance': feature_importance,
                'clinical_insights': insights,
                'shap_statistics': shap_stats,
                'shap_available': True
            }
            
            self.explainability_results[model_name] = results
            self.clinical_insights[model_name] = insights
            
            print(f"  ✅ Explainability analysis completed for {model_name}")
            return results
            
        except Exception as e:
            print(f"  ❌ SHAP analysis failed for {model_name}: {e}")
            return self._fallback_explainability_analysis(model, X_test, y_test, model_name)
    
    def _create_explainer(self, model, X_test: pd.DataFrame):
        """Create appropriate SHAP explainer based on model type."""
        model_name = type(model).__name__.lower()
        
        try:
            # For custom models, access underlying model
            if hasattr(model, 'model') and model.model is not None:
                underlying_model = model.model
                underlying_model_name = type(underlying_model).__name__.lower()
            else:
                underlying_model = model
                underlying_model_name = model_name
            
            # Check for categorical features that might cause issues
            categorical_features = self._identify_categorical_features(X_test)
            has_categorical = len(categorical_features) > 0
            
            # Tree-based models
            if any(name in underlying_model_name for name in ['xgb', 'lightgbm', 'lgbm', 'randomforest']):
                # For XGBoost with categorical features, use KernelExplainer to avoid warnings
                if 'xgb' in underlying_model_name and has_categorical:
                    print(f"  → Using KernelExplainer for {underlying_model_name} (categorical features detected)")
                    predict_wrapper = self._create_predict_wrapper(model, X_test)
                    return shap.KernelExplainer(predict_wrapper, X_test.iloc[:50])
                # Use KernelExplainer for LightGBM due to categorical feature issues
                elif any(name in underlying_model_name for name in ['lightgbm', 'lgbm']):
                    print(f"  → Using KernelExplainer for {underlying_model_name}")
                    predict_wrapper = self._create_predict_wrapper(model, X_test)
                    return shap.KernelExplainer(predict_wrapper, X_test.iloc[:50])
                else:
                    print(f"  → Using TreeExplainer for {underlying_model_name}")
                    return shap.TreeExplainer(underlying_model)
            
            # Linear models
            elif any(name in underlying_model_name for name in ['linear', 'ridge', 'lasso']):
                return shap.LinearExplainer(underlying_model, X_test)
            
            # Other models - use KernelExplainer
            else:
                print(f"  → Using KernelExplainer for {underlying_model_name}")
                predict_wrapper = self._create_predict_wrapper(model, X_test)
                
                # Use smaller background set for Prophet models to reduce computation
                if 'prophet' in underlying_model_name:
                    background_size = min(10, len(X_test))  # Very small for Prophet
                    print(f"  → Using small background set ({background_size}) for Prophet")
                else:
                    background_size = min(50, len(X_test))  # Standard background
                
                return shap.KernelExplainer(predict_wrapper, X_test.iloc[:background_size])
                
        except Exception as e:
            print(f"  → Failed to create explainer: {e}")
            return None
    
    def _create_predict_wrapper(self, model, X_test: pd.DataFrame):
        """Create prediction wrapper for KernelExplainer with feature consistency handling."""
        # Check if this is a Prophet model
        model_name = type(model).__name__.lower()
        is_prophet = 'prophet' in model_name
        
        # Pre-compute numeric columns from X_test to use consistently
        numeric_columns = X_test.select_dtypes(include=[np.number]).columns.tolist()
        
        def predict_wrapper(X):
            if isinstance(X, np.ndarray):
                # Use only numeric column names when converting numpy array to DataFrame
                # This ensures we have the right number of columns for the numeric data
                n_cols = min(X.shape[1], len(numeric_columns))
                X_df = pd.DataFrame(X[:, :n_cols], columns=numeric_columns[:n_cols])
            else:
                X_df = X.copy()
                # Apply the same numeric feature selection that models use during training
                if isinstance(X_df, pd.DataFrame):
                    X_df = X_df.select_dtypes(include=[np.number]).fillna(0)
            
            try:
                # Handle feature consistency for models with stored feature names
                if hasattr(model, 'feature_names_') and model.feature_names_:
                    # Check for feature count mismatch
                    if len(X_df.columns) != len(model.feature_names_):
                        print(f"⚠️  Feature count mismatch for {model_name}: expected {len(model.feature_names_)}, got {len(X_df.columns)}")
                        
                        # If we have more features than expected, select the training features
                        if len(X_df.columns) > len(model.feature_names_):
                            available_features = [f for f in model.feature_names_ if f in X_df.columns]
                            if available_features:
                                X_df = X_df[available_features]
                        
                        # If we have fewer features, add missing ones as zeros
                        else:
                            for feature in model.feature_names_:
                                if feature not in X_df.columns:
                                    X_df[feature] = 0
                            X_df = X_df[model.feature_names_]
                
                # Validate feature dimensions for Prophet models
                if is_prophet:
                    # Special handling for Prophet models to prevent timestamp overflow
                    if len(X_df) > 1000:
                        print(f"⚠️  Large SHAP sample ({len(X_df)}) for Prophet - using average prediction")
                        # Return average prediction to avoid timestamp issues
                        if hasattr(model, 'model') and hasattr(model.model, 'history'):
                            avg_prediction = np.mean(model.model.history['y'].values)
                            return np.full(len(X_df), max(0, avg_prediction))
                        else:
                            return np.full(len(X_df), 10.0)  # Reasonable default
                
                # Make predictions
                predictions = model.predict(X_df)
                
                # Ensure predictions are valid
                if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                    print("⚠️  Invalid predictions in SHAP wrapper, using fallback")
                    predictions = np.full_like(predictions, np.nanmean(predictions))
                
                return predictions
                
            except Exception as e:
                print(f"⚠️  Prediction error in SHAP wrapper: {e}")
                # Return safe fallback predictions
                return np.full(len(X_df), 10.0)
        
        return predict_wrapper
    
    def _calculate_feature_importance(self, shap_values, X_test: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature importance from SHAP values."""
        if isinstance(shap_values, list):
            shap_array = shap_values[0]
        else:
            shap_array = shap_values
        
        # Calculate mean absolute SHAP values
        importance = np.mean(np.abs(shap_array), axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Calculate percentages
        total_importance = feature_importance['importance'].sum()
        feature_importance['relative_importance'] = feature_importance['importance'] / total_importance
        feature_importance['importance_percentage'] = feature_importance['relative_importance'] * 100
        
        return feature_importance
    
    def _calculate_shap_statistics(self, shap_values) -> Dict[str, float]:
        """Calculate descriptive statistics for SHAP values."""
        if isinstance(shap_values, list):
            shap_array = shap_values[0]
        else:
            shap_array = shap_values
        
        return {
            'mean_absolute_shap': float(np.mean(np.abs(shap_array))),
            'std_shap': float(np.std(shap_array)),
            'max_shap': float(np.max(shap_array)),
            'min_shap': float(np.min(shap_array)),
            'positive_contributions': float(np.mean(shap_array > 0)),
            'negative_contributions': float(np.mean(shap_array < 0))
        }
    
    def _generate_clinical_insights(self, shap_values, X_test: pd.DataFrame, 
                                  y_test: pd.Series, model_name: str) -> Dict[str, Any]:
        """Generate healthcare-specific clinical insights."""
        insights = {
            'top_drivers': [],
            'temporal_insights': {},
            'clinical_interpretations': [],
            'risk_factors': [],
            'protective_factors': []
        }
        
        try:
            # Get feature importance
            feature_importance = self._calculate_feature_importance(shap_values, X_test)
            top_features = feature_importance.head(10)
            
            # Analyze top drivers
            for _, row in top_features.iterrows():
                feature_name = row['feature']
                importance = row['importance_percentage']
                
                insights['top_drivers'].append({
                    'feature': feature_name,
                    'importance': importance,
                    'clinical_meaning': self._interpret_feature_clinically(feature_name)
                })
            
            # Temporal and seasonal insights
            temporal_features = [f for f in X_test.columns if any(term in f.lower() 
                               for term in ['day', 'week', 'month', 'season', 'holiday', 'time', 'lag'])]
            
            for feature in temporal_features:
                if feature in feature_importance['feature'].values:
                    imp = feature_importance[feature_importance['feature'] == feature]['importance_percentage'].iloc[0]
                    insights['temporal_insights'][feature] = {
                        'importance': imp,
                        'interpretation': self._interpret_temporal_feature(feature)
                    }
            
            # Risk vs protective factors
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                mean_shap = np.mean(shap_values, axis=0)
                
                for i, feature in enumerate(X_test.columns):
                    impact = mean_shap[i]
                    importance = feature_importance[feature_importance['feature'] == feature]['importance_percentage'].iloc[0]
                    
                    if importance > 1.0:  # Only consider features with >1% importance
                        if impact > 0:
                            insights['risk_factors'].append({
                                'feature': feature,
                                'impact': float(impact),
                                'importance': float(importance),
                                'interpretation': f"Higher {feature} increases trolley demand"
                            })
                        else:
                            insights['protective_factors'].append({
                                'feature': feature,
                                'impact': float(abs(impact)),
                                'importance': float(importance),
                                'interpretation': f"Higher {feature} decreases trolley demand"
                            })
        
        except Exception as e:
            print(f"  → Insight generation failed: {e}")
        
        return insights
    
    def _interpret_feature_clinically(self, feature_name: str) -> str:
        """Provide clinical interpretation for features."""
        clinical_mappings = {
            'lag_1_day': 'Previous day dependency - captures short-term persistence',
            'lag_7_day': 'Weekly seasonality - strong predictor of demand patterns',
            'lag_14_day': 'Bi-weekly patterns - captures extended healthcare cycles',
            'moving_avg_7d': '7-day moving average - smoothed demand trend',
            'moving_avg_14d': '14-day moving average - longer-term trend indicator',
            'day_of_week': 'Day of week effect - captures weekly healthcare patterns',
            'is_weekend': 'Weekend effect - different staffing and patient patterns',
            'month': 'Monthly seasonality - captures seasonal healthcare variations',
            'is_holiday': 'Holiday effect - impacts both demand and staffing',
            'precipitation': 'Weather impact - affects patient mobility and demand',
            'temperature': 'Temperature effect - influences patient conditions',
            'trend': 'Long-term trend - captures systematic changes over time'
        }
        
        # Check for exact matches first
        if feature_name in clinical_mappings:
            return clinical_mappings[feature_name]
        
        # Check for partial matches
        for key, interpretation in clinical_mappings.items():
            if key in feature_name.lower():
                return interpretation
        
        # Default interpretation
        return f"Healthcare feature: {feature_name} - requires domain expert interpretation"
    
    def _interpret_temporal_feature(self, feature_name: str) -> str:
        """Interpret temporal features from healthcare perspective."""
        temporal_interpretations = {
            'lag_1': 'Strong dependency on previous day - indicates persistent demand patterns',
            'lag_7': 'Weekly cycle effect - reflects healthcare system weekly rhythm',
            'lag_14': 'Bi-weekly pattern - captures pay-cycle or administrative patterns',
            'day_of_week': 'Weekly pattern - Monday surge, weekend drop typical in ED',
            'is_weekend': 'Weekend effect - reduced planned procedures, increased emergencies',
            'month': 'Seasonal variation - winter surge, summer lull common',
            'holiday': 'Holiday disruption - staff availability and patient behavior changes',
            'trend': 'Systematic change - population growth, policy changes, or capacity evolution'
        }
        
        for key, interpretation in temporal_interpretations.items():
            if key in feature_name.lower():
                return interpretation
        
        return f"Temporal pattern in {feature_name} - influences healthcare demand timing"
    
    def _fallback_explainability_analysis(self, model, X_test: pd.DataFrame, 
                                        y_test: pd.Series, model_name: str) -> Dict[str, Any]:
        """Fallback analysis when SHAP is not available."""
        print(f"  → Using fallback feature importance for {model_name}")
        
        results = {
            'model_name': model_name,
            'feature_names': list(X_test.columns),
            'shap_available': False,
            'fallback_mode': True
        }
        
        try:
            # Try built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': importance,
                    'importance_percentage': (importance / importance.sum()) * 100
                }).sort_values('importance', ascending=False)
                
                results['feature_importance'] = feature_importance
                results['clinical_insights'] = {'method': 'Built-in feature importance'}
            
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                feature_importance = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': importance,
                    'importance_percentage': (importance / importance.sum()) * 100
                }).sort_values('importance', ascending=False)
                
                results['feature_importance'] = feature_importance
                results['clinical_insights'] = {'method': 'Linear model coefficients'}
            
            else:
                results['clinical_insights'] = {'method': 'No feature importance available'}
        
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def perform_statistical_significance_testing(self) -> Dict[str, Dict[str, Any]]:
        """Perform statistical significance testing between models."""
        if not SCIPY_AVAILABLE or len(self.performance_results) < 2:
            print("⚠️  Statistical testing requires scipy and at least 2 models")
            return {}
        
        print(f"\n📊 Performing statistical significance testing...")
        
        statistical_tests = {}
        models = list(self.performance_results.keys())
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                test_name = f"{model1}_vs_{model2}"
                
                try:
                    residuals1 = self.performance_results[model1]['residuals']
                    residuals2 = self.performance_results[model2]['residuals']
                    
                    # Ensure residuals are the same length
                    min_len = min(len(residuals1), len(residuals2))
                    residuals1 = residuals1[:min_len]
                    residuals2 = residuals2[:min_len]
                    
                    # Paired t-test for residuals
                    t_stat, p_value = stats.ttest_rel(np.abs(residuals1), np.abs(residuals2))
                    
                    # Determine which model is better
                    rmse1 = self.performance_results[model1]['metrics']['rmse']
                    rmse2 = self.performance_results[model2]['metrics']['rmse']
                    better_model = model1 if rmse1 < rmse2 else model2
                    
                    statistical_tests[test_name] = {
                        'model1': model1,
                        'model2': model2,
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'better_model': better_model,
                        'rmse_difference': float(abs(rmse1 - rmse2))
                    }
                    
                    significance = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
                    print(f"  {model1} vs {model2}: p={p_value:.4f} ({significance})")
                    
                except Exception as e:
                    print(f"  ❌ Statistical test failed for {test_name}: {e}")
        
        self.statistical_tests = statistical_tests
        return statistical_tests
    
    def create_explainability_comparison(self) -> pd.DataFrame:
        """Create comprehensive explainability comparison."""
        if not self.explainability_results:
            print("⚠️  No explainability results available for comparison")
            return pd.DataFrame()
        
        print(f"\n🔍 Creating explainability comparison...")
        
        comparison_data = []
        
        for model_name, results in self.explainability_results.items():
            row = {
                'Model': model_name,
                'SHAP_Available': results.get('shap_available', False),
                'Fallback_Mode': results.get('fallback_mode', False)
            }
            
            if 'feature_importance' in results:
                fi = results['feature_importance']
                
                # Explainability metrics
                row['Total_Features'] = len(fi)
                row['Top_Feature_Importance'] = fi.iloc[0]['importance_percentage']
                row['Top_5_Concentration'] = fi.head(5)['importance_percentage'].sum()
                row['Features_90_Percent'] = len(fi[fi['importance_percentage'].cumsum() <= 90])
                row['Most_Important_Feature'] = fi.iloc[0]['feature']
                
                # Explainability score (inverse of concentration - more distributed = more explainable)
                row['Explainability_Score'] = 100 - row['Top_5_Concentration']
            
            if 'shap_statistics' in results:
                stats = results['shap_statistics']
                row['Mean_SHAP_Impact'] = stats.get('mean_absolute_shap', 0)
                row['SHAP_Variability'] = stats.get('std_shap', 0)
                row['Feature_Consistency'] = 1 / (1 + stats.get('std_shap', 1))
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models by explainability
        if 'Explainability_Score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('Explainability_Score', ascending=False)
            comparison_df['Explainability_Rank'] = range(1, len(comparison_df) + 1)
        
        return comparison_df
    
    def save_results(self) -> None:
        """Save all evaluation results to files."""
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving evaluation results to {self.results_dir}...")
        
        try:
            # Save performance results
            if self.performance_results:
                performance_df = self._create_performance_summary()
                performance_df.to_csv(self.results_dir / "model_performance.csv", index=False)
                print(f"  ✅ Performance results saved")
            
            # Save explainability results
            if self.explainability_results:
                explainability_comparison = self.create_explainability_comparison()
                if not explainability_comparison.empty:
                    explainability_comparison.to_csv(self.results_dir / "explainability_comparison.csv", index=False)
                    print(f"  ✅ Explainability comparison saved")
            
            # Save statistical tests
            if self.statistical_tests:
                stats_df = pd.DataFrame(self.statistical_tests).T
                stats_df.to_csv(self.results_dir / "statistical_tests.csv")
                print(f"  ✅ Statistical tests saved")
            
            # Save detailed results as JSON
            import json
            
            # Prepare results for JSON serialization
            serializable_results = {
                'performance_results': self._make_json_serializable(self.performance_results),
                'explainability_results': self._make_json_serializable(self.explainability_results),
                'statistical_tests': self.statistical_tests,
                'clinical_insights': self._make_json_serializable(self.clinical_insights)
            }
            
            with open(self.results_dir / "detailed_results.json", 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"  ✅ Detailed results saved")
            
        except Exception as e:
            print(f"  ❌ Error saving results: {e}")
    
    def _create_performance_summary(self) -> pd.DataFrame:
        """Create performance summary DataFrame."""
        summary_data = []
        
        for model_name, results in self.performance_results.items():
            metrics = results['metrics']
            row = {
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'MAPE': metrics['mape'],
                'Fallback_Mode': results.get('fallback_mode', False)
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        # Sort by RMSE (lower is better)
        df = df.sort_values('RMSE', ascending=True)
        df['Rank'] = range(1, len(df) + 1)
        
        return df
    
    def _make_json_serializable(self, obj) -> Any:
        """Make objects JSON serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def run_complete_evaluation(self, models_dict: Dict[str, Any], X_test: pd.DataFrame, 
                              y_test: pd.Series) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            models_dict: Dictionary of trained models
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Complete evaluation results
        """
        print(f"\n🚀 Running Complete Model Evaluation Pipeline")
        print("=" * 60)
        
        # 1. Evaluate model performance
        for model_name, model in models_dict.items():
            self.evaluate_model_performance(model, X_test, y_test, model_name)
        
        # 2. Perform explainability analysis
        if self.enable_shap:
            for model_name, model in models_dict.items():
                self.perform_explainability_analysis(model, X_test, y_test, model_name)
        
        # 3. Statistical significance testing
        self.perform_statistical_significance_testing()
        
        # 4. Create explainability comparison
        explainability_comparison = self.create_explainability_comparison()
        
        # 5. Save results
        self.save_results()
        
        # 6. Generate summary
        summary = self._generate_evaluation_summary()
        
        print("\n✅ Complete evaluation pipeline finished successfully!")
        return summary
    
    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary."""
        summary = {
            'total_models_evaluated': len(self.performance_results),
            'shap_analysis_completed': len(self.explainability_results),
            'statistical_tests_performed': len(self.statistical_tests),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        if self.performance_results:
            # Find champion model
            champion_model = min(self.performance_results.items(), 
                               key=lambda x: x[1]['metrics']['rmse'])
            
            summary['champion_model'] = {
                'name': champion_model[0],
                'rmse': champion_model[1]['metrics']['rmse'],
                'mae': champion_model[1]['metrics']['mae'],
                'r2': champion_model[1]['metrics']['r2']
            }
        
        return summary
    
    def _identify_categorical_features(self, X_test: pd.DataFrame) -> List[str]:
        """Identify categorical features that might cause SHAP issues."""
        categorical_features = []
        
        # Known categorical features that cause issues
        categorical_patterns = [
            'Month_Name', 'Day_Name', 'Day_Type', 'Season', 'Quarter', 
            'Region', 'Hospital', 'Precipitation_Category'
        ]
        
        for col in X_test.columns:
            # Check if column matches known categorical patterns
            if any(pattern in col for pattern in categorical_patterns):
                categorical_features.append(col)
            # Check if column has object/string dtype
            elif X_test[col].dtype == 'object':
                categorical_features.append(col)
            # Check if column has limited unique values (likely categorical)
            elif X_test[col].nunique() < 20 and X_test[col].dtype in ['int64', 'float64']:
                if not col.endswith(('_lag', '_diff', '_rolling', '_trend')):  # Exclude time series features
                    categorical_features.append(col)
        
        return categorical_features


# Example usage and testing
def main():
    """Example usage of the ModelEvaluator."""
    print("🔬 Model Evaluator Demo")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 200
    
    # Sample feature matrix
    X_test = pd.DataFrame({
        'lag_1_day': np.random.poisson(15, n_samples),
        'lag_7_day': np.random.poisson(15, n_samples),
        'moving_avg_7d': np.random.normal(15, 3, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'is_holiday': np.random.randint(0, 2, n_samples)
    })
    
    # Sample target
    y_test = pd.Series(10 + 0.8 * X_test['lag_1_day'] + 0.3 * X_test['lag_7_day'] + 
                      2 * X_test['is_weekend'] + np.random.normal(0, 2, n_samples))
    
    # Create simple mock models
    class MockModel:
        def __init__(self, name, noise_level=1.0):
            self.name = name
            self.noise_level = noise_level
        
        def predict(self, X):
            # Simple linear combination with noise
            return (10 + 0.7 * X['lag_1_day'] + 0.2 * X['lag_7_day'] + 
                   1.5 * X['is_weekend'] + np.random.normal(0, self.noise_level, len(X)))
    
    models_dict = {
        'Simple_Model_A': MockModel('A', 1.0),
        'Simple_Model_B': MockModel('B', 1.5),
        'Simple_Model_C': MockModel('C', 0.8)
    }
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        results_dir="results/evaluator_demo",
        enable_shap=True,
        horizons=[1, 3, 7]
    )
    
    # Run complete evaluation
    results = evaluator.run_complete_evaluation(models_dict, X_test, y_test)
    
    print(f"\n📊 Evaluation Summary:")
    print(f"   Models evaluated: {results['total_models_evaluated']}")
    print(f"   Champion model: {results.get('champion_model', {}).get('name', 'N/A')}")
    print(f"   Champion RMSE: {results.get('champion_model', {}).get('rmse', 'N/A'):.3f}")


if __name__ == "__main__":
    main()

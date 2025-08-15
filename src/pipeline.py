#!/usr/bin/env python3
"""
UNIFIED PIPELINE
================

Main pipeline script that orchestrates all components of the Emergency Department 
trolley demand forecasting system in a clean, streamlined workflow.

This unified pipeline replaces the complex multi-script system with a single, 
well-organized pipeline that handles:
- Data processing and feature engineering
- Model training and evaluation
- Regional modeling capabilities
- Comprehensive evaluation and explainability analysis
- Professional visualization generation

Components:
- DataProcessor: Unified data processing and feature engineering
- ModelTrainer: Unified model training with regional capabilities
- ModelEvaluator: Unified evaluation and explainability analysis
- Visualizer: Unified visualization generation

"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import argparse

# Add src directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import unified modules
from data_processor import DataProcessor
from models import ModelTrainer, RegionalModelTrainer
from evaluator import ModelEvaluator
from visualizer import Visualizer

warnings.filterwarnings('ignore')


class UnifiedPipeline:
    """
    Unified pipeline that orchestrates all components.
    """
    
    def __init__(self, data_path: str = None, results_dir: str = None, 
                 config: dict = None):
        """
        Initialize the unified pipeline.
        
        Args:
            data_path: Path to the master data file
            results_dir: Directory for results
            config: Configuration dictionary
        """
        # Set up paths
        self.data_path = data_path or "/Users/karthik/dissertation/data/master_trolley_data.csv"
        self.results_dir = Path(results_dir) if results_dir else Path("results")
        
        # Default configuration
        self.config = config or {
            'test_size': 0.2,
            'enable_regional': True,
            'enable_shap': True,
            'horizons': [1, 3, 7, 14],
            'regional_sample_size': 500,  # For demonstration
            'models_to_train': ['xgboost', 'optimized_xgboost', 'lightgbm', 'sarima', 'prophet', 'ensemble', 'baseline_14']
        }
        
        # Initialize components
        self.data_processor = DataProcessor(data_path=str(self.data_path), results_dir=str(self.results_dir / "data"))
        self.model_trainer = ModelTrainer(results_dir=str(self.results_dir / "models"))
        self.regional_trainer = RegionalModelTrainer(results_dir=str(self.results_dir / "models"))
        self.evaluator = ModelEvaluator(results_dir=str(self.results_dir / "evaluation"))
        self.visualizer = Visualizer(results_dir=str(self.results_dir / "visualizations"))
        
        # Results storage
        self.processed_data = None
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained_models = {}
        self.regional_models = {}
        self.evaluation_results = {}
        
        print(f"🚀 Unified Pipeline initialized")
        print(f"📂 Data path: {self.data_path}")
        print(f"📂 Results directory: {self.results_dir}")
        print(f"⚙️  Configuration: {self.config}")
    
    def run_complete_pipeline(self) -> dict:
        """
        Run the complete pipeline from data processing to visualization.
        
        Returns:
            Dictionary containing pipeline results
        """
        print(f"\n🚀 STARTING UNIFIED PIPELINE")
        print("=" * 80)
        
        try:
            # 1. Data Processing
            self._run_data_processing()
            
            # 2. Model Training
            self._run_model_training()
            
            # 3. Regional Training (if enabled)
            if self.config.get('enable_regional', True):
                self._run_regional_training()
            
            # 4. Model Evaluation
            self._run_evaluation()
            
            # 5. Visualization
            self._run_visualization()
            
            # 6. Generate Summary
            summary = self._generate_pipeline_summary()
            
            print(f"\n✅ UNIFIED PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            return summary
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _run_data_processing(self):
        """Run data processing stage."""
        print(f"\n📊 STAGE 1: DATA PROCESSING")
        print("-" * 40)
        
        # Load and process data
        print("Loading data...")
        if not self.data_processor.load_data():
            raise ValueError("Failed to load data")
        
        print("Engineering features...")
        if not self.data_processor.engineer_features():
            raise ValueError("Failed to engineer features")
        
        # Get processed data
        self.processed_data = self.data_processor.data
        
        print("Preparing temporal train/test split...")
        # Use proper temporal splitting for time series data (NO RANDOM SHUFFLING)
        # This prevents data leakage by ensuring training data comes before test data
        split_idx = int(len(self.processed_data) * (1 - self.config['test_size']))
        
        # Temporal split - training data from beginning, test data from end
        self.train_data = self.processed_data.iloc[:split_idx].copy()
        self.test_data = self.processed_data.iloc[split_idx:].copy()
        
        print(f"  ✓ Temporal split: {len(self.train_data)} train, {len(self.test_data)} test samples")
        print(f"  ✓ Training period: {self.train_data.index[0]} to {self.train_data.index[-1]}")
        print(f"  ✓ Testing period: {self.test_data.index[0]} to {self.test_data.index[-1]}")
        print("  ✓ NO DATA LEAKAGE: Future data cannot influence past predictions")
        
        # Extract features and targets
        feature_columns = [col for col in self.processed_data.columns 
                          if col not in ['Total_Trolleys', 'Date']]
        
        self.X_train = self.train_data[feature_columns]
        self.y_train = self.train_data['Total_Trolleys']
        self.X_test = self.test_data[feature_columns]
        self.y_test = self.test_data['Total_Trolleys']
        
        print(f"✅ Data processing completed")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Test samples: {len(self.X_test)}")
        print(f"   Features: {len(feature_columns)}")
        
        # Save processed data
        # Save processed data (DataProcessor uses its internal data)
        self.data_processor.save_processed_data()
    
    def _run_model_training(self):
        """Run model training stage."""
        print(f"\n🤖 STAGE 2: MODEL TRAINING")
        print("-" * 40)
        
        # Use the correct ModelTrainer interface
        results_df = self.model_trainer.train_models(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        # Update tracked models from ModelTrainer
        self.trained_models = self.model_trainer.models.copy()
        
        if results_df is not None:
            print(f"✅ Model training completed")
            print(f"   Models trained: {len(self.trained_models)}")
            print(f"   Champion: {self.model_trainer.champion_model['name'] if self.model_trainer.champion_model else 'None'}")
        else:
            print(f"❌ No models trained successfully")
        
        # Save trained models (ModelTrainer handles this internally)
        self.model_trainer.save_models()
    
    def _run_regional_training(self):
        """Run regional training stage."""
        print(f"\n🏥 STAGE 3: REGIONAL TRAINING")
        print("-" * 40)
        
        # Prepare regional data
        print("Preparing regional data...")
        regional_data = self.data_processor.prepare_regional_data()
        
        if regional_data:
            print(f"Training regional models for {len(regional_data)} regions...")
            
            try:
                results_df = self.regional_trainer.train_regional_models(regional_data)
                
                # Update tracked regional models
                self.regional_models = self.regional_trainer.regional_models.copy()
                
                if results_df is not None:
                    print(f"✅ Regional training completed")
                    print(f"   Regional models trained: {len(self.regional_models)}")
                    
                    # Generate regional forecasts
                    forecasts = self.regional_trainer.generate_regional_forecasts(
                        regional_data, forecast_days=14
                    )
                    
                    if forecasts:
                        print(f"   Regional forecasts generated: {len(forecasts)}")
                else:
                    print("❌ Regional training failed")
                    
            except Exception as e:
                print(f"  ❌ Regional training failed: {e}")
        else:
            print("⚠️  No regional data available for training")
    
    def _run_evaluation(self):
        """Run evaluation stage."""
        print(f"\n📊 STAGE 4: MODEL EVALUATION")
        print("-" * 40)
        
        # Use models from ModelTrainer
        trained_models = self.model_trainer.models
        
        if not trained_models:
            print("⚠️  No trained models available for evaluation")
            return
        
        # Run comprehensive evaluation
        self.evaluation_results = self.evaluator.run_complete_evaluation(
            trained_models, self.X_test, self.y_test
        )
        
        print(f"✅ Model evaluation completed")
        print(f"   Models evaluated: {self.evaluation_results.get('total_models_evaluated', 0)}")
        
        # Display champion model
        champion = self.evaluation_results.get('champion_model')
        if champion:
            print(f"   🏆 Champion model: {champion['name']}")
            print(f"   🏆 Champion RMSE: {champion['rmse']:.3f}")
    
    def _run_visualization(self):
        """Run visualization stage."""
        print(f"\n🎨 STAGE 5: VISUALIZATION")
        print("-" * 40)
        
        # Prepare data for visualization
        viz_data = self.processed_data.copy()
        if 'Date' in viz_data.columns:
            viz_data = viz_data.set_index('Date')
        
        # Clean up categorical columns that can cause matplotlib errors
        categorical_cols_to_remove = ['Month_Name', 'Day_Name', 'Day_Type', 'Season', 'Quarter', 'Region', 'Hospital']
        for col in categorical_cols_to_remove:
            if col in viz_data.columns and viz_data[col].dtype == 'object':
                print(f"  Removing categorical column from visualization: {col}")
                viz_data = viz_data.drop(columns=[col])
        
        # Ensure only numeric data is used for visualization
        numeric_cols = viz_data.select_dtypes(include=[np.number]).columns
        viz_data = viz_data[numeric_cols]
        print(f"  Using {len(numeric_cols)} numeric columns for visualization")
        
        # Create performance summary
        performance_df = None
        if self.evaluator.performance_results:
            performance_data = []
            for model_name, results in self.evaluator.performance_results.items():
                metrics = results['metrics']
                performance_data.append({
                    'Model': model_name,
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'R²': metrics['r2'],
                    'MAPE': metrics['mape']
                })
            performance_df = pd.DataFrame(performance_data)
        
        # Generate sample forecasts for visualization
        forecasts_dict = None
        if self.trained_models:
            forecasts_dict = {}
            forecast_horizon = 14
            
            for model_name, model in self.trained_models.items():
                try:
                    # Generate simple forecast
                    last_features = self.X_test.tail(1)
                    forecast_values = []
                    
                    for i in range(forecast_horizon):
                        pred = model.predict(last_features)[0]
                        forecast_values.append(pred)
                        # Simple feature update (this is simplified)
                        last_features = last_features.copy()
                    
                    forecast_dates = pd.date_range(
                        start=viz_data.index[-1] + pd.Timedelta(days=1),
                        periods=forecast_horizon, freq='D'
                    )
                    
                    forecasts_dict[model_name] = pd.Series(forecast_values, index=forecast_dates)
                    
                except Exception as e:
                    print(f"  ⚠️ Forecast generation failed for {model_name}: {e}")
        
        # Generate all visualizations
        self.visualizer.save_all_plots(viz_data, performance_df, forecasts_dict)
        
        print(f"✅ Visualization completed")
        print(f"   Charts saved to: {self.visualizer.charts_dir}")
        print(f"   Dashboards saved to: {self.visualizer.dashboards_dir}")
    
    def _generate_pipeline_summary(self) -> dict:
        """Generate pipeline execution summary."""
        summary = {
            'pipeline_completed': True,
            'execution_timestamp': datetime.now().isoformat(),
            'data_processing': {
                'training_samples': len(self.X_train) if self.X_train is not None else 0,
                'test_samples': len(self.X_test) if self.X_test is not None else 0,
                'features_engineered': len(self.X_train.columns) if self.X_train is not None else 0
            },
            'model_training': {
                'models_trained': len(self.trained_models),
                'regional_models': len(self.regional_models),
                'model_types': list(self.trained_models.keys())
            },
            'evaluation': self.evaluation_results,
            'results_directory': str(self.results_dir)
        }
        
        # Save summary
        import json
        summary_file = self.results_dir / "pipeline_summary.json"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📋 Pipeline Summary:")
        print(f"   Training samples: {summary['data_processing']['training_samples']}")
        print(f"   Test samples: {summary['data_processing']['test_samples']}")
        print(f"   Features: {summary['data_processing']['features_engineered']}")
        print(f"   Models trained: {summary['model_training']['models_trained']}")
        print(f"   Regional models: {summary['model_training']['regional_models']}")
        print(f"   Summary saved: {summary_file}")
        
        return summary
    
    def run_quick_demo(self):
        """Run a quick demonstration with sample data."""
        print(f"\n🚀 RUNNING QUICK DEMO")
        print("=" * 50)
        
        # Create sample data for demo
        print("Creating sample data for demonstration...")
        
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        
        # Create realistic trolley demand data
        trend = np.linspace(10, 15, n_samples)
        seasonal_weekly = 3 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
        seasonal_yearly = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
        noise = np.random.normal(0, 2, n_samples)
        
        trolleys = np.maximum(0, trend + seasonal_weekly + seasonal_yearly + noise)
        
        # Create sample dataset
        sample_data = pd.DataFrame({
            'Date': dates,
            'Total_Trolleys': trolleys,
            'ED_Trolleys': trolleys * 0.6,
            'Ward_Trolleys': trolleys * 0.4,
            'Precipitation': np.random.exponential(2, n_samples),
            'Temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25) + np.random.normal(0, 2, n_samples),
            'Is_Public_Holiday': np.random.binomial(1, 0.05, n_samples),
            'Region': np.random.choice(['HSE_East', 'HSE_West', 'HSE_South'], n_samples)
        })
        
        # Save sample data
        sample_data_path = self.results_dir / "sample_data.csv"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        sample_data.to_csv(sample_data_path, index=False)
        
        print(f"✅ Sample data created: {sample_data_path}")
        print(f"   Samples: {len(sample_data)}")
        print(f"   Date range: {dates[0]} to {dates[-1]}")
        
        # Update config for demo
        demo_config = self.config.copy()
        demo_config.update({
            'test_size': 0.2,
            'enable_regional': True,
            'regional_sample_size': 200,
            'models_to_train': ['xgboost', 'optimized_xgboost', 'lightgbm', 'sarima', 'prophet', 'ensemble', 'baseline_14']
        })
        
        # Update data path
        self.data_path = str(sample_data_path)
        self.config = demo_config
        
        # Run pipeline with sample data
        return self.run_complete_pipeline()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Unified Emergency Department Trolley Demand Forecasting Pipeline')
    
    parser.add_argument('--data', type=str, 
                       default="/Users/karthik/dissertaion-backup/data/master_trolley_data.csv",
                       help='Path to the master data file')
    
    parser.add_argument('--results', type=str, default="results",
                       help='Results directory')
    
    parser.add_argument('--demo', action='store_true',
                       help='Run quick demo with sample data')
    
    parser.add_argument('--no-regional', action='store_true',
                       help='Disable regional modeling')
    
    parser.add_argument('--no-shap', action='store_true',
                       help='Disable SHAP explainability analysis')
    
    parser.add_argument('--models', nargs='+', 
                       default=['xgboost', 'optimized_xgboost', 'lightgbm', 'sarima', 'prophet', 'ensemble', 'baseline_14'],
                       help='Models to train')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    # Configure based on arguments
    config = {
        'test_size': args.test_size,
        'enable_regional': not args.no_regional,
        'enable_shap': not args.no_shap,
        'horizons': [1, 3, 7, 14],
        'models_to_train': args.models,
        'regional_sample_size': 500
    }
    
    # Initialize pipeline
    pipeline = UnifiedPipeline(
        data_path=args.data,
        results_dir=args.results,
        config=config
    )
    
    try:
        if args.demo:
            # Run demo
            summary = pipeline.run_quick_demo()
        else:
            # Run full pipeline
            summary = pipeline.run_complete_pipeline()
        
        print(f"\n🎉 Pipeline execution summary:")
        print(f"   Models trained: {summary['model_training']['models_trained']}")
        print(f"   Evaluation completed: {'champion_model' in summary['evaluation']}")
        print(f"   Results location: {summary['results_directory']}")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

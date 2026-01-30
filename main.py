import argparse
import sys
import os

# This ensures Python can see the 'src' folder even if run from different locations
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

#Now we import your functions from the src folder
try:
    from src.train import train_model
    from src.predict import make_prediction
    from src.evaluate import evaluate_features
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure your files are names correctly in the 'src' folder")
    
def main():
    #1. Set up the command-line interface
    parser = argparse.ArgumentParser(description="JobAI Salary Prediction System")
    
    #2. Add the 'mode' argument
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'predict', 'evaluate'],
        help="Choose 'train' to build the model, 'predict' for a salary estimate, or 'evaluate' for charts."
        )
    
    args = parser.parse_args()
    
    #3.Route the command to the correct Script
    if args.mode == 'train':
        print("--- Starting Training Pipeline ---")
        train_model()
        
    elif args.mode == 'predict':
        print("--- Running Salary Prediction ---")
        make_prediction()
        
    elif args.mode == 'evaluate':
        print("--- Generating Feature Importance Chart ---")
        evaluate_features()
        
if __name__ == "__main__":
    main()
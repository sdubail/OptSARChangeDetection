""" 
The script analyzes patch data from a JSON file, calculates the ratio of positive patches,
and generates a histogram to visualize the distribution of positive patch ratios across images.

Also, it provides command-line arguments to specify the input data file and output figure path.
It includes functions to load data, calculate overall and image-specific statistics, and plot the distribution.
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import argparse
from pathlib import Path


def load_data(file_path):
    """
    Load data from a JSON file.
   
    Args:
        file_path (str or Path): Path to the JSON file
       
    Returns:
        pd.DataFrame: DataFrame containing the loaded data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def calculate_overall_stats(df):
    """
    Calculate the overall ratio of positive patches.
   
    Args:
        df (pd.DataFrame): DataFrame with patch data
       
    Returns:
        float: Ratio of positive patches
    """
    positive_ratio = df['is_positive'].mean()
    print(f"Overall ratio of positive patches: {positive_ratio:.4f}")
    return positive_ratio


def calculate_image_stats(df):
    """
    Calculate statistics per image.
   
    Args:
        df (pd.DataFrame): DataFrame with patch data
       
    Returns:
        pd.DataFrame: DataFrame with image statistics
    """
    image_stats = df.groupby('image_id').agg(
        total_patches=('is_positive', 'count'),
        positive_patches=('is_positive', 'sum')
    )
    print(f"Total number of images: {len(image_stats)}")
   
    # Calculate ratio of positive patches per image
    image_stats['positive_ratio'] = image_stats['positive_patches'] / image_stats['total_patches']
    return image_stats


def plot_distribution(image_stats, positive_ratio, file_path):
    """
    Create a histogram of positive patch ratios across images.
   
    Args:
        image_stats (pd.DataFrame): DataFrame with image statistics
        positive_ratio (float): Overall ratio of positive patches
        file_path (str or Path): Path to save the figure
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(image_stats['positive_ratio'], bins=100, edgecolor='black')
   
    # Calculate statistical values
    median_val = image_stats['positive_ratio'].median()
    q1_val = image_stats['positive_ratio'].quantile(0.25)
    q3_val = image_stats['positive_ratio'].quantile(0.75)
   
    # Add mean line with value
    plt.axvline(positive_ratio, color='red', linestyle='dashed', linewidth=1,
                label=f'Overall Mean: {positive_ratio:.3f}')
   
    # Add median line with value
    plt.axvline(median_val, color='red', linestyle='-.', linewidth=1,
                label=f'Median: {median_val:.3f}')
   
    # Add quartiles lines with values
    plt.axvline(q1_val, color='black', linestyle='dashed', linewidth=1,
                label=f'Q1: {q1_val:.3f}')
    plt.axvline(q3_val, color='black', linestyle='-.', linewidth=1,
                label=f'Q3: {q3_val:.3f}')
   
    plt.xlabel('Ratio of Positive Patches')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Positive Patch Ratios Across Images')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.savefig(file_path)
    print(f"Figure saved to: {file_path}")


def get_default_paths():
    """
    Get default paths for data and output files relative to the script location.
    
    Returns:
        tuple: (data_path, output_path)
    """
    # Get the directory where the script is located
    script_dir = Path(__file__).resolve().parent
    
    # Define paths relative to the script directory
    data_path = script_dir / "processed_patches" / "train_metadata.json"
    output_path = script_dir / "processed_patches" / "positive_ratio_distribution.png"
    
    return str(data_path), str(output_path)


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    default_data_path, default_output_path = get_default_paths()
    
    parser = argparse.ArgumentParser(description="Analyze patch data and create distribution plot")
    parser.add_argument("--data", type=str, default=default_data_path,
                        help=f"Path to the JSON data file (default: {default_data_path})")
    parser.add_argument("--output", type=str, default=default_output_path,
                        help=f"Path to save the output figure (default: {default_output_path})")
    
    return parser.parse_args()


def main():
    """Main function to analyze patch data."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Display paths being used
    print(f"Using data file: {args.data}")
    print(f"Output will be saved to: {args.output}")
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        print("Please provide the correct path using the --data argument")
        return
    
    # Load data
    df = load_data(args.data)
    
    # Calculate overall stats
    positive_ratio = calculate_overall_stats(df)
    
    # Calculate image stats
    image_stats = calculate_image_stats(df)
    
    # Plot distribution
    plot_distribution(image_stats, positive_ratio, args.output)


if __name__ == "__main__":
    main()
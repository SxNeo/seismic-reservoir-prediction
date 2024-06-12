import matplotlib.colors as mcolors
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

way = 'ISD'  # CWT

def load_predictions(file_path):
    """
    Load predictions from an Excel file, ignoring the first row.
    """
    try:
        # Use skiprows=1 to ignore the first row
        return pd.read_excel(file_path, header=None, skiprows=1)
    except Exception as e:
        print(f"Error loading the predictions file: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

def plot_classification_profile(predictions_df, output_path, algorithm, sample_num, n_classes):
    colors = {0: 'red', 1: 'yellow', 2: 'gray'}
    cmap = ListedColormap([colors.get(i, 'black') for i in range(n_classes)])  # Default color is black

    rgb_image = np.zeros((*predictions_df.shape, 3))
    for value, color in colors.items():
        mask = predictions_df == value
        rgb_image[mask.values] = mcolors.to_rgb(color)

    fig, ax = plt.subplots()
    ax.imshow(rgb_image, interpolation='nearest')
    ax.axis('off')
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    ax.set_position([0, 0, 1, 1])

    output_file = os.path.join(output_path, f'{way}_{algorithm}_predictions_{n_classes}.png')  # _CNN
    plt.savefig(output_file, format='png', bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

def plot_proba_profile(output_path, algorithm, sample_num, n_classes):
    """
    Plot lithology profile based on probability predictions.
    """
    # Create a gradient color map
    gray_Green_yellow_cmap = LinearSegmentedColormap.from_list(
        "GrayGreenYellow",  # Name
        ['gray', 'green', 'yellow', 'red']  # Color range: from gray (0.5, 0.5, 0.5) to green (0, 1, 0) to yellow (1, 1, 0)
    )
    cmap = gray_Green_yellow_cmap

    # Plot images for each class
    for class_index in range(n_classes):
        # Load sorted probability data for each class
        sorted_proba_df = pd.read_excel(
            rf"{output_path}\{way}_{algorithm}_sorted_proba_class_{n_classes}_{class_index}.xlsx")
        sorted_class_proba = sorted_proba_df.values

        # Get colors based on probability values
        class_image = cmap(sorted_class_proba)

        # Create the image and adjust layout
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(class_image, aspect='auto')
        ax.axis('off')
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
        ax.set_position([0, 0, 1, 1])

        # Show and save the image
        plt.tight_layout()
        plt.savefig(
            rf'{output_path}\{way}_{algorithm}_sorted_proba_predictions_image_class_{n_classes}_{class_index}.png',
            format='png',
            bbox_inches='tight', pad_inches=0, dpi=600)

def main():

    algorithm = 'SAMME.R'  # 'SAMME' or 'SAMME.R'
    sample_num = 119
    n_classes = 3
    output_path = rf"C:\01 CodeofPython\AdaBoost_CNN\AdaBoost_CNN-master\code\{way}"

    if algorithm == 'SAMME':
        file_path = f"{output_path}/{way}_{algorithm}_predictions_{n_classes}.xlsx"

        predictions_df = load_predictions(file_path)
        plot_classification_profile(predictions_df, output_path, algorithm, sample_num, n_classes)
    else:
        plot_proba_profile(output_path, algorithm, sample_num, n_classes)

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from itertools import cycle
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

def read_results_from_txt(file_path):
    results = {}
    current_method = None
    current_sample_length = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('Method:'):
                current_method = line.split(':')[1].strip()
                results[current_method] = {}
            elif line.startswith('Sample_Length_'):
                current_sample_length = int(line.split('_')[-1].strip()[:-1])
                results[current_method][current_sample_length] = {}
            elif line.startswith('CNN_training') or line.startswith('CNN_testing') \
                or line.startswith('Ada_CNN_training') or line.startswith('Ada_CNN_testing'):
                metric_name, values_str = line.split(':')
                metric_name = metric_name.strip()
                values = [float(value.strip()) for value in values_str.split(',')]
                results[current_method][current_sample_length][metric_name] = values

    return results

def plot_f1_scores_by_sample_length(all_results, sample_lengths, output_path):
    evaluation_types = ['training', 'testing']
    titles = ['Training Data', 'Testing Data']
    colors = cycle(['blue', 'red'])
    line_styles = cycle(['-', '--', '-.', ':'])

    for i, evaluation in enumerate(evaluation_types):
        plt.figure(figsize=(6, 4))
        handles = []
        labels = []

        for way, results in all_results.items():
            color = next(colors)
            line_style = '-'  # next(line_styles)
            scores_cnn = [np.mean(results[length]['CNN_' + evaluation]) for length in sample_lengths]
            scores_ada_cnn = [np.mean(results[length]['Ada_CNN_' + evaluation]) for length in sample_lengths]
            line1, = plt.plot(sample_lengths, scores_cnn, marker='o', color=color, linestyle=line_style, label=f'{way} CNN')
            line2, = plt.plot(sample_lengths, scores_ada_cnn, marker='^', color=color, linestyle=line_style, label=f'{way} Ada_CNN')
            handles.extend([line1, line2])
            labels.extend([f'{way}-CNN', f'{way}-AdaBoost-CNN'])

        plt.xlabel('Sample Length')
        plt.ylabel(f'F1 of {evaluation} data')
        plt.xticks(sample_lengths)
        plt.grid(True)

        plt.savefig(os.path.join(output_path, f'f1_scores_{evaluation}_combined.png'), bbox_inches='tight', pad_inches=0, dpi=600)
        plt.close()

        fig_leg = plt.figure(figsize=(3, 2))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.legend(handles, labels, loc='center')
        ax_leg.axis('off')
        fig_leg.savefig(os.path.join(output_path, f'legend_{evaluation}.png'), bbox_inches='tight', pad_inches=0, dpi=600)
        plt.close(fig_leg)

file_path = r"C:\01 CodeofPython\AdaBoost_CNN\AdaBoost_CNN-master\code\sample length\all_results.txt"
results = read_results_from_txt(file_path)

sample_lengths = [5,10,15,20,25]

output_path = r"C:\01 CodeofPython\AdaBoost_CNN\AdaBoost_CNN-master\code\sample length"
plot_f1_scores_by_sample_length(results, sample_lengths, output_path)

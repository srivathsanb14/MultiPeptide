import os
import matplotlib.pyplot as plt
import yaml

def parse_pdb(pdb_file):
    num_atoms = 0
    num_alpha_carbons = 0
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                num_atoms += 1
                if line[13:15].strip() == "CA":
                    num_alpha_carbons += 1
    return num_atoms, num_alpha_carbons

def get_counts_from_pdb_folder(folder_path):
    num_atoms_list = []
    num_alpha_carbons_list = []
    for pdb_file in os.listdir(folder_path):
        if pdb_file.endswith(".pdb"):
            pdb_file_path = os.path.join(folder_path, pdb_file)
            num_atoms, num_alpha_carbons = parse_pdb(pdb_file_path)
            num_atoms_list.append(num_atoms)
            num_alpha_carbons_list.append(num_alpha_carbons)
    return num_atoms_list, num_alpha_carbons_list

def plot_histogram(data, xlabel, ylabel, title, path, bins=30):

    min_value = min(data)
    max_value = max(data)
    
    width = 1.5
    fig, ax = plt.subplots(figsize = [5,3], dpi = 600)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(width)
    tick_width = 0.5
    plt.hist(data, bins=bins, edgecolor='black')
    plt.axvline(min_value, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(max_value, color='red', linestyle='dashed', linewidth=1)
    plt.text(min_value, plt.ylim()[1] * 0.9, f'Min: {min_value}', color='black', fontsize=10, ha='right', rotation=90, verticalalignment='top')
    plt.text(max_value, plt.ylim()[1] * 0.9, f'Max: {max_value}', color='black', fontsize=10, ha='right', rotation=90, verticalalignment='top')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(path, title + ".png"))
    plt.close(fig)


if __name__ == "__main__":
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)

    task = config['task']
    pdb_folder_path = f'./data/{task}/pdb/'

    image_path = os.path.join('plots/sequence_analysis/', task)
    os.makedirs(image_path, exist_ok=True)

    num_atoms_list, num_alpha_carbons_list = get_counts_from_pdb_folder(pdb_folder_path)

    plot_histogram(num_atoms_list, 'Number of atoms', 'Frequency', f'Number of atoms - {task}', image_path)
    plot_histogram(num_alpha_carbons_list, 'Sequence length', 'Frequency', f'Sequence length - {task}', image_path)

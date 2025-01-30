# MultiPeptide

Welcome to the MultiPeptide repository! This repository corresponds to the MultiPeptide project. If you'd like to learn more about the research behind this project, please check out our paper:  
[Multi-Peptide: Multimodality Leveraged Language-Graph Learning of Peptide Properties](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01443)

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/srivathsanb14/MultiPeptide.git
   cd MultiPeptide

2. **Install Required Packages:**

Ensure you have Python and pip installed. Then, install the necessary packages:

  `pip install -r requirements.txt`

3. **Download Datasets and Checkpoints:** 

Datasets: Download the datasets from the specified [Hugging Face link](https://huggingface.co/srivathsanb14/MultiPeptide/).
Save the datasets under the existing data folder in this repository.

Models/Checkpoints: Download the models/checkpoints from the specified [Hugging Face link](https://huggingface.co/srivathsanb14/MultiPeptide/). For the checkpoints - look under the 'updated_inference' folder under the checkpoints directory.
Save the models/checkpoints under a new checkpoints folder in this repository.

4. **Run Training or Inference:**

To train the CLIP process, run:
  `python main.py`

To perform inference using the trained checkpoints, run:
  `python inference.py`
 
## Citation

If you use *Multi-Peptide* in your work, please cite the following:

---

@article{doi:10.1021/acs.jcim.4c01443,
author = {Badrinarayanan, Srivathsan and Guntuboina, Chakradhar and Mollaei, Parisa and Barati Farimani, Amir},
title = {Multi-Peptide: Multimodality Leveraged Language-Graph Learning of Peptide Properties},
journal = {Journal of Chemical Information and Modeling},
volume = {65},
number = {1},
pages = {83-91},
year = {2025},
doi = {10.1021/acs.jcim.4c01443},
    note ={PMID: 39700492},
URL = { 
        https://doi.org/10.1021/acs.jcim.4c01443
},
eprint = { 
        https://doi.org/10.1021/acs.jcim.4c01443
}
}
---

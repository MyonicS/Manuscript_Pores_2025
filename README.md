# Description
This repository hosts the companion jupyter notebook for the publication *The Effect of Impurities in Post-Consumer Plastic Waste on the Cracking of Polyolefins with Zeolite-based Catalysts* (https://doi.org/10.26434/chemrxiv-2025-hw10r)
It provides an executable version of the manuscript generating all figures and analyses from raw experimental data.
The raw data is hosted on the Open Science Foundation repository under DOI: https://doi.org/10.17605/OSF.IO/5WTZY

# Getting started
## In Google Colab
The easiest way to run the notebook is in Google Colab.
Click the link below and run the notebook.
<a href="https://colab.research.google.com/github/MyonicS/Manuscript_Pores_2025/blob/main/Manuscript_Notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
At release the notebook is running python 3.12. 
Cells at the top of the notebook clone the repository, install packages and download the experimental data to the Colab runtime. Note that these steps can take a couple minutes.  

## Running the notebook locally
The notebook is optimized to run in Google Colab. When running the notebook locally delete or comment out all code cells before importing of modules. 
1. Clone the repository:
   ```sh
   git clone https://github.com/MyonicS/Manuscript_Pores_2025
   ```
2. (optional but recommended: Create a new python 3.12 environment)
3. When you have UV installed, you can install the remaining requirements by navigating to the repository and running:
   ```sh
   uv pip sync pyproject.toml
   ```
4. Download the experimental data from the OSF repository.
   You can do this manually by downloading the ZIP file from [here](https://doi.org/10.17605/OSF.IO/ZXCTH) and unzipping the folder into the repository, or by using the [datahugger](https://github.com/J535D165/datahugger) library
5. Run the notebook in your IDE of choice. We recommend VS code as it allows for interactive plots in-line, which can be enabled using 
   ```python
   %matplotlib widget
   ```
	at the start of a cell.
	
# Citing

When utilizing code or data from this study in an academic publication please cite the following manuscript:
> Rejman, S., Meijer, T., van de Minkelis, J.H., Seyed Khabbaz, H., Gooneie, A., Duijndam, A.J.A., Vollmer, I., Weckhuysen, B.M. On the Influence of Catalyst Pore Structure in the Catalytic Cracking of Polypropylene. _ChemRxiv_ (2025). https://doi.org/10.26434/chemrxiv-2025-hw10r 
Alternatively, the data itself can be cited as:
> Rejman, S. et al. Experimental data supporting: ‘On the Influence of Catalyst Pore Structure in the Catalytic Cracking of Polypropylene’. OSF http://dx.doi.org/10.17605/OSF.IO/5WTZY (2025)


# Bugs and Comments
Feels free to submit bug reports, comments or questions as issues.
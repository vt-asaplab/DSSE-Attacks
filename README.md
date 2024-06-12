# Update Frequency Attacks for Forward/Backward-Private Searchable Encryption

This repository contains the code used to simulate the attacks presented in our CODASPY '24 Paper, "Exploiting Update Leakage in Searchable Encryption"

# Running the Experiments

First, download the code into the environment that you would like to run the attacks. We use Linux-style paths which may break on Windows.

Next, install NumPy and SciPy into your virtual environment.

Run the experiments using the command below. This will take a long time - we recommend running the experiments overnight.

  python3 automate_experiments.py

The experiments run in the order that they are listed in the `exp_settings.csv` file.
As experiments are completed, you will see intermediate result files appear in the `experiment_results/` directory.
If for any reason the experiments stop (crash, want to stop and continue later, etc.), place a `#` in front of all lines for experiments
in `exp_settings.csv` that have already been ran (or simply delete them).

# Preparing the Results

Before the results can be graphed, we have an intermediate processing step to combine the results for each type of experiment into one file.

Run the following command:

  python3 output_to_graph_format.py

You should see 10 files appear in the `results_for_latex` directory.

# Graphing the Results

Open `main.tex` in your favorite LaTeX editor, such as OverLeaf. Drop the `results_for_latex/` folder into the environment (i.e., do not put the contents of `results_for_latex/` into the same level as `main.tex`, the working directory should not be flat).

Compile `main.tex` and view the results of the graphs. We use constant values in 4 places in the figures; please Ctrl-F for "Constant" and verify that each of the values match the results from your `eta_mu.txt`. If not, replace those constants with the values you have.

# Notes

We have wrapped the logic used to generate our paper's experimental results to automate our previously-tedious workflow; the result is the code in this repo. The intent and make it user-friendly for readers to verify our results. We have done some testing to make sure that the code in this repository works, but we expect this process to have introduced some bugs. If you run into bugs or crashes, please don't spend your time trying to fix them - report them to jacobshalt(at)vt(dot)edu and we will do our best to fix them promptly.

This code may be used for academic research only. All other uses are prohibited.

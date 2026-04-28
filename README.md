# Genetic Algorithm Optimizer of Protein Stability
The genetic algorithm optimizer is a software tool designed to identify protein-stabilizing mutations. It was developed as part of a bachelor thesis focused on designing stabilizing mutations for the IsPETase enzyme. However, it can be reused for other proteins by adjusting the parameters in the parametrization file. 
## Files in the repository
The *Documentation.md* file contains description of the implementation.

The *genetic_algorithm_optimizer* directory contains the following directories and files.
- Directory *foldx* contains the FoldX stability predictor software used in the program to evaluate protein stability and guide the search.
- Directory *protein_data* contains the protein sequence in FASTA format (downloaded from the UniProt database) and the structure in PDB format (downloaded from the AlphaFold database). Currently, it contains data for IsPETase. When using the program with another protein, its sequence and structure will be downloaded into this directory.
- File *Parallel_ga_optimizer.py* contains the full implementation of the algorithm in Python.
- File *Results_over_generation.txt* contains a summary of input parameters and results, including the five best individuals from each generation with their fitness values, and the ten best individuals from the final generation.
- File *Parametrization_file.txt* contains input parameters for the program, such as protein code, restricted sites, allowed amino acids, number of individuals, and number of generations.
- File *Results.txt* contains results from multiple runs (added manually) for the purpose of analyzing consistency. These results are specific to the IsPETase enzyme and are not required for running the program on other proteins.
- File *A0A0K8P6T7.pdb* is the wild-type structure of IsPETase used internally during computation. This file is not required for running the program on other proteins.
- File *fitness_plot.png* shows the evolution of average and best fitness over generations. It is specific to the IsPETase experiment.
- File *best_individual_structure.pdb* contains the structure of the best mutant found during the run. This file is also specific to the IsPETase experiment.

Files *Results_over_generation.txt*, *fitness_plot.png* and *best_individual_structure.pdb* are currently specific to the IsPETase experiment. However, when running this program on another protein, these files will be overwritten with the new resuls.

## Using of Genetic Algorithm Optimizer
To use the program for another protein, clone the repository (preferably on a Linux-based system due to FoldX) and modify the parametrization file.

### Parameters Setting

protein_code = \<UniProt ID>

restricted_sites = \<positions separated by commas>

amino_acids = \<one-letter amino acid codes>

number_of_individuals = <e.g. 100>

number_of_generations = <e.g. 100>

### Example Parameters Setting for IsPETase

protein_code = A0A0K8P6T7

restricted_sites = 87,160,161,185,203,206,237,239,273,289

amino_acids = ACDEFGHIKLMNPQRSTVWY

number_of_individuals = 100

number_of_generations = 100

### Results Analysis
Results from the algorithm run can be further analysed to validate the biological importance of identified mutations. Mutated amino acids at given positions can be examined using available literature and scientific articles, as well as protein databases that contain results of experimental studies. For visualization, tools such as PyMOL can be used, in which the structure outputted by this program (internally by FoldX) can be loaded.

### Warnings
There are several things that should be kept in mind when using this program.
- **Sequence consistency**: it is necessary to manually verify that the sequence obtained from the UniProt database and the structure from the AlphaFold database correspond to the same protein sequence. Any mismatch may cause FoldX to crash and lead to inconsistent results.
-  **Positions of mutations**: residue positions are internally represented using 0-based indexing, following the standard Python convention. For FoldX computations, these positions are converted to standard 1-based protein numbering. Therefore, mutations correspond to *position + 1* in standard protein residue numbering. For example, mutation (213, 'P') represents a mutation to proline at Python position 213, but in the protein it actually corresponds to position 214.
- **Computation time**: the primary limitation of the implementation lies in the computational cost of the fitness evaluation. FoldX calculations are relatively expensive, and despite parallelization, execution may require several hours depending on the hardware. For example, on the computational setup used in the study of IsPETase, the evaluation of a single individual required approximately 2.5 seconds. For a population of 100 individuals over 100 generations, the total run time of one run was approximately 7-8 hours.
- **Results consistency**: due to the stochastic nature of the algorithm, different runs may produce different results. However, repeated convergence toward similar mutations across independent runs may indicate potentially meaningful stabilizing effects.
- **Biological importance**: it should be noted that all promising results obtained with this algorithm are based on computational predictions and therefore require further validation through experimental studies.

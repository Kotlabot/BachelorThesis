import warnings
import random
import requests
import subprocess
from pathlib import Path
from Bio import SeqIO
import os
import glob
from multiprocessing import Pool, cpu_count
import shutil
import matplotlib.pyplot as plt

# Method to parse the parametrization file and extract the necessary parameters,
# such as the protein code (UniProt ID), the restricted sites, the allowed amino acids,
# the number of individuals in the population, and the number of generations for the genetic algorithm.
def parse_parametization_file(file_path):
    protein_code = None
    restricted_sites = []
    amino_acids = None
    number_of_individuals = None
    number_of_generations = None

    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue 

        if line.startswith('protein_code'):
            protein_code = line.split('=')[1].strip()
            # If the protein code is missing in the parametrization file, raise an exception 
            # with error message and end the program.
            if not protein_code:
                raise ValueError("Required parameter 'protein_code' (UniProt ID) is missing in the parametrization file.")
            
        elif line.startswith('restricted_sites'):
            restricted_sites = [int(x) for x in line.split('=')[1].strip().split(',')]
            # If the restricted sites are missing in the parametrization file, 
            # raise a warning and default to an empty list (no restricted sites).
            if not restricted_sites:
                warnings.warn("Parameter 'restricted_sites' is missing. Defaulting to an empty list (no restricted sites).")
                restricted_sites = []

        elif line.startswith('amino_acids'):
            amino_acids = line.split('=')[1].strip()
            # If the allowed amino acids are missing in the parametrization file, 
            # raise a warning and default to all 20 standard amino acids.
            if not amino_acids:
                warnings.warn("Parameter 'amino_acids' is missing. Defaulting to all 20 standard amino acids.")
                amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            
            # Validate that the provided amino acid codes are 
            # valid single-letter codes for standard amino acids.
            valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
            if not set(amino_acids).issubset(valid_amino_acids):
                raise ValueError("Invalid amino acid codes provided.")
            
        elif line.startswith('number_of_individuals'):
            number_of_individuals = int(line.split('=')[1].strip())
            # If the number of individuals is missing in the parametrization file, 
            # raise a warning and default to 100 individuals in the population.
            if not number_of_individuals:
                warnings.warn("Parameter 'number_of_individuals' is missing. Defaulting to 100.")
                number_of_individuals = 100

        elif line.startswith('number_of_generations'):
            number_of_generations = int(line.split('=')[1].strip())
            # If the number of generations is missing in the parametrization file, 
            # raise a warning and default to 100 generations for the genetic algorithm.
            if not number_of_generations:
                warnings.warn("Parameter 'number_of_generations' is missing. Defaulting to 100.")
                number_of_generations = 100

    return protein_code, restricted_sites, amino_acids, number_of_individuals, number_of_generations

# Method to download the FASTA sequence and the AlphaFold PDB structure for a given UniProt ID.
def download_protein_data(uniprot_id, out_dir='protein_data'):
    # Create the output directory for protein sequence and structure, if it doesn't exist.
    Path(out_dir).mkdir(exist_ok=True)
    
    # Construct the URLs for the FASTA and PDB files based on the UniProt ID.
    fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
    
    # Define the local paths where the downloaded FASTA and PDB files will be saved.
    fasta_path = Path(out_dir) / f"{uniprot_id}.fasta"
    pdb_path = Path(out_dir) / f"{uniprot_id}.pdb"
    
    # Download the FASTA file and save it locally. 
    fasta_response = requests.get(fasta_url)
    if fasta_response.status_code == 200:
        fasta_path.write_text(fasta_response.text)
    # If the download fails, raise an exception.
    else:
        raise Exception(f"Failed FASTA for {uniprot_id}")
    
    # Download the PDB file and save it locally.
    pdb_response = requests.get(pdb_url)
    if pdb_response.status_code == 200:
        with open(pdb_path, "wb") as f:
            f.write(pdb_response.content)
    # If the download fails, raise an exception.
    else:
        raise Exception(f"Failed AlphaFold PDB for {uniprot_id}")
    
    return fasta_path, pdb_path

# Method to process the FASTA file and extract the wild-type sequence and its length.
def get_wild_type_sequence(fasta_path):
    record = SeqIO.read(fasta_path, "fasta")
    return str(record.seq), len(record.seq)

# Method to ensure that each position appears only once in an individual.
def enforce_unique_positions(individual):
    seen = {}
    # Store individuals mutations in a dictionary. If two or more mutations at
    # the same position appears, rewrite the amino acid change in the dictionary
    # to keep only one to prevent crashing of FoldX (two mutations at the same 
    # position are biologicaly imposible).
    for position, amino_acid in individual:
        seen[position] = amino_acid
    # Return individual as a list of valid mutations.
    return [(position, amino_acid) for position, amino_acid in seen.items()]

# Method to initialize a random population of mutants.
def initialize_random_population(wild_type_sequence, wild_type_length, restricted_sites, amino_acids, population_size):
    # Select allowed positions for mutation by excluding the restricted sites from the wild-type sequence 
    # (i+1 because restricted sites are 1-based).
    allowed_positions = [i for i in range(wild_type_length) if (i+1) not in restricted_sites]

    # Generate all possible unique single-point mutations (each allowed position with all non–wild-type amino acids)
    # to ensure a diverse initial population without duplicate individuals.
    all_possible_mutations = []
    for position in allowed_positions:
        for amino_acid in amino_acids:
            if amino_acid != wild_type_sequence[position]:
                all_possible_mutations.append((position, amino_acid))

    # Randomly choose from all unique mutations which will be part of the initialization population.
    selected_mutations = random.sample(all_possible_mutations, population_size)

    # Add each selected mutation to population as an individual in correct format (list mutations, now with one mutation).
    population = [[mutation] for mutation in selected_mutations]

    return population

# Parallel method to evaluate population fitness.
def evaluate_population_fitness_parallel(population, wild_type_sequence, protein_code, generation):
    # Convert population into argument tuples including all parameters required for fitness computation and file handling.
    args = [(individual, wild_type_sequence, protein_code, generation, i) for i, individual in enumerate(population)]

    # Determine number of working threads according to number of available CPU cores, while excluding one to
    # prevent freezing of the operation system.
    number_of_processors = max(1, cpu_count() - 1)

    # Create multiprocessing pool where each working thread independently evaluate one individual.
    with Pool(number_of_processors) as pool:
        # pool.map ensures that input order will match the output order of the individuals in population.
        fitness = pool.map(evaluate_individual, args)

    return fitness

# Method to evaluate fitness of an individual.
def evaluate_individual(args):
    # Each working thread receives parameters required for the foldX run.
    individual, wild_type_sequence, protein_code, generation, index = args

    # Create unique directory for current individual to prevent file collision in parallel execution.
    run_directory = f"foldx_run_g{generation}_i{index}"
    os.makedirs(run_directory, exist_ok=True)

    # Link the wild-type structure to the working directory (avoid copying the structure to every directory).
    pdb_source = f"{protein_code}.pdb"
    pdb_destination = os.path.join(run_directory, f"{protein_code}.pdb")
    if not os.path.exists(pdb_destination):
        os.symlink(os.path.abspath(pdb_source), pdb_destination)

    # Create mutation file which will be passed to foldX.
    mutation_file = os.path.join(run_directory, "individual_list.txt")
    with open(mutation_file, "w") as f:
        mutations = []
        for position, mutated_amino_acid in individual:
            wild_type_amino_acid = wild_type_sequence[position]
            # Pass the variant to foldX in correct format.
            mutations.append(f"{wild_type_amino_acid}A{position+1}{mutated_amino_acid}")
        # Add comma between the mutations (variants), and semicolon at the end of the line,
        # as this format foldX requires.
        f.write(','.join(mutations) + ";\n")

    # Create a command passed to foldX to evaluate the individual
    # given the wild-type structure and a list of mutations.
    cmd = [
        "../foldx/foldx_20270131",
        "--command=BuildModel",
        f"--pdb={protein_code}.pdb",
        "--mutant-file=individual_list.txt",
        "--noHeader=1",
        "--numberOfRuns=3"
    ]
    result = subprocess.run(cmd, cwd=run_directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # If foldX evaluation crashes for an individual, assign it the worst fitness possible.
    # Crash can be caused by wrong command or invalid mutation input.
    if result.returncode != 0:
        print("The run itself was corrupted.")
        return 1e6

    # Path to the file where energy differences are stored.
    output_file = os.path.join(run_directory, f"Dif_{protein_code}.fxout")

    # If foldX did not generate the output difference file (file or directory error),
    # assign this individual teh worst fitness possible.
    if not os.path.exists(output_file):
        print("The Dif file does not exist.")
        return 1e6

    # List of energy differences (ddGs) for one mutant structure through different runs of foldX.
    ddgs = []
    # Read from foldX output file with energy differencies for the mutant structures.
    with open(output_file) as f:
        for line in f:
            # Select only lines which start with the uniprot protein code, as in these 
            # lines are stored the energy differences.
            if line.startswith(protein_code):
                parts = line.split()
                if len(parts) >= 2:
                    # Extract and append the energy difference used as fitness.
                    ddgs.append(float(parts[1]))

    # If the difference file produced by foldX did not contain the expected ddG values and
    # the ddG list is therefore empty, than assing this individual the worst fitness possible.
    if not ddgs:
        print("No ddGs obtained.")
        return 1e6

    # Compute and return the average ddG of the three foldX runs.
    return sum(ddgs) / len(ddgs)

# Method to perform selection of individuals for the next generation using elitism, tournament selection and variation operators.
def perform_population_selection(population, fitness, restricted_sites, amino_acids, wild_type_sequence, elitism_rate=0.05):
    # Combine lists of population and fitness into a single list sorted in ascending order 
    # (minimum fitness first = lower ddG first, as the lower ddG means better stabilization).
    combined = sorted(zip(population, fitness), key=lambda x: x[1], reverse=False)
    # Calculate the number of elite individuals to keep and extract them to next generation.
    elite_count = max(1, int(len(population) * elitism_rate))
    elite_individuals = [ind for ind, _ in combined[:elite_count]]
    new_generation = elite_individuals.copy()

    # Perform the tournament selection to fill the rest of the population.
    while len(new_generation) < len(population):
        # Select 3 randomly picked individuals from the population for the tournament.
        tournament = random.sample(combined, 3)
        # With a 60% probability select the fittest parent from the tournament (fittest individual = lowest fitness value).
        if random.random() < 0.6:
            parent = min(tournament, key=lambda x: x[1])[0]
        # Otherwise select randomly one parent from the tournament (slightly increase diversity).
        else:
            parent = random.choice(tournament)[0]
        
        probability = random.random()

        # With a 5% probability, create the offspring by deleting the last individuals 
        # mutation (only if the individual has more than one mutation).
        if probability < 0.05 and len(parent) > 1:
            offspring = parent[:-1]

        # With a 25% probability, create the offspring by combining mutations from two different parents.
        elif probability < 0.3:
            second_tournament = random.sample(combined, 3)
            second_parent = min(second_tournament, key=lambda x: x[1])[0]
            offspring = crossover(parent, second_parent)
        
        # With a 40% probability, create the offspring by inheriting parents mutations and obtaining one new mutation.
        elif probability < 0.7 and len(parent) < 4:
            offspring = add_new_mutation(parent, restricted_sites, amino_acids, wild_type_sequence)
        
        # With a 30% probability, select one of individuals mutations and mutate its position or change its amino acid.
        else:
            offspring = change_mutation(parent, restricted_sites, amino_acids, wild_type_sequence)

        new_generation.append(offspring)

    return new_generation

# Method to add one new mutation to the provided individual.
def add_new_mutation(parent, restricted_sites, amino_acids, wild_type_sequence):
    offspring = parent.copy()
    # Select allowed positions for mutation by excluding the restricted sites from the wild-type sequence and all already mutated positions in the individual.
    allowed_positions = [i for i in range(len(wild_type_sequence)) if (i+1) not in restricted_sites and all(position != i for position, _ in offspring)]
    
    # In case no allowed positions remained, return individual unchanged.
    if not allowed_positions:
        return offspring

    # Randomly select position of mutation from allowed positions.
    mutated_position = random.choice(allowed_positions)

    # Select allowed amino acids for mutation from list of amino acids excluding the wild-type.
    allowed_amino_acids = [amino_acid for amino_acid in amino_acids if amino_acid != wild_type_sequence[mutated_position]]
    # Randomly select the mutated amino acid.
    mutated_amino_acid = random.choice(allowed_amino_acids)

    offspring.append((mutated_position, mutated_amino_acid))
    # Return the double-checked offspring (no position collision).
    return enforce_unique_positions(offspring)

# Method to perform crossover of two parents resulting in one combined offspring.
def crossover(first_parent, second_parent):
    offspring = []
    # Randomly select one mutation from the first parent and assign it to the offspring.
    offspring = [random.choice(first_parent)]
    # Randomly select one mutation from the second parent.
    second_mutation = random.choice(second_parent)

    # Try to perform crossover. If the two selected mutations are at the same position, keep only the mutation from the first parent.
    if second_mutation[0] != offspring[0][0]:
        offspring.append(second_mutation)
    
    # Return the double-checked offspring (no position collision).
    return enforce_unique_positions(offspring)

# Method change position or amino acid of one of the individual mutations. 
def change_mutation(individual, restricted_sites, amino_acids, wild_type_sequence):
    offspring = individual.copy()
    # Choose randomly which mutation will be modified.
    mutation_index = random.randint(0, len(offspring)-1)
    mutation_position, mutation_amino_acid = offspring[mutation_index]
    probability = random.random()

    # With a 40% chance, mutate the position of the mutation while keeping the amino acid the same.
    if probability < 0.4:
        # Define allowed positions by excluding the restricted sites and positions of another individuals mutations.
        allowed_positions = [i for i in range(len(wild_type_sequence)) if (i+1) not in restricted_sites and all(position != i for j, (position, _) in enumerate(offspring) if j != mutation_index)]
        
        # Select randomly new position of current mutation.
        if allowed_positions:
            mutation_position = random.choice(allowed_positions)
    
    # With a 60% chance, mutate the amino acid while keeping the same position of the mutation.
    else:
        # Define allowed amino acids from amino acid list by excluding the wild-type amino acid and the current amino acid.
        allowed_amino_acids = [amino_acid for amino_acid in amino_acids if amino_acid != wild_type_sequence[mutation_position] and amino_acid != mutation_amino_acid]
        mutation_amino_acid = random.choice(allowed_amino_acids)

    offspring[mutation_index] = (mutation_position, mutation_amino_acid)
    # Return the double-checked offspring (no position collision).
    return enforce_unique_positions(offspring)

# Method to print and also save the information from the parametrization file and
# also best individuals along with their fitness over generations to the result file.
def log_and_print(text, file):
    print(text)
    file.write(text + "\n")

# Method to clean directories and files produced by foldX computations.
def cleanup_foldx_directories():
    for folder in os.listdir():
        if folder.startswith("foldx_run_"):
            shutil.rmtree(folder, ignore_errors=True)

# Method to plot best fitness values and average fitness values over generations.
def plot_fitness(best_fitness, average_fitness, number_of_generations, output_file="fitness_plot.png"):
    generations = list(range(number_of_generations + 1))
    
    # Plot the results.
    plt.figure()
    plt.plot(generations, best_fitness, label="Best Fitness")
    plt.plot(generations, average_fitness, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (ddG kcal/mol)")
    plt.title("Fitness over Generations")
    plt.legend()
    plt.grid()

    # Save the plot into the working directory.
    plt.savefig(output_file)
    plt.close()

# Method to save the structure of the best individual found.
def save_best_structure(best_index, generation, protein_code):
    # Select the correct directory of the best performing individual, produced by foldX.
    run_directory = f"foldx_run_g{generation}_i{best_index}"

    # Get the structure of the best performing individual from foldX run directory.
    pdb_files = glob.glob(os.path.join(run_directory, f"{protein_code}_*.pdb"))
    best_pdb = pdb_files[0]

    # Copy this structure into working directory.
    shutil.copy(best_pdb, "best_individual_structure.pdb")

def main():
    # Clean the working space.
    cleanup_foldx_directories()
    # Set path to the file where results will be saved. 
    result_file = "Results_over_generations.txt"
    # Delete previous result file if it exists.
    if os.path.exists(result_file):
        os.remove(result_file)

    with open(result_file, "a") as f:
        # Set the path to parametrization file which includes the essential parameters.
        file_path = 'Parametrization_file.txt'
        # Extract these parameters and store them as variables.
        protein_code, restricted_sites, amino_acids, number_of_individuals, number_of_generations = parse_parametization_file(file_path)

        # Print and save to result file defined parameters for current run.
        log_and_print(f'Protein Code: {protein_code}', f)
        log_and_print(f'Restricted Sites: {restricted_sites}', f)
        log_and_print(f'Amino Acids: {amino_acids}', f)
        log_and_print(f'Number of Individuals: {number_of_individuals}', f)
        log_and_print(f'Number of Generations: {number_of_generations}', f)

        # Extract the paths to protein sequence and structure files.
        fasta_path, pdb_path = download_protein_data(protein_code)
        # Extract the actual wild-type protein sequence from fasta file.
        wild_type_sequence, wild_type_length = get_wild_type_sequence(fasta_path)

        # Ensure PDB is in working directory.
        shutil.copy(pdb_path, f"{protein_code}.pdb")

        # Initialize random population of individuals.
        population = initialize_random_population(wild_type_sequence, wild_type_length, restricted_sites, amino_acids, number_of_individuals)

        # Print and save the sequence and its length to the result file.
        log_and_print(f'Wild-Type Sequence: {wild_type_sequence}', f)
        log_and_print(f'Wild-Type Length: {wild_type_length}', f)

        # Prepare a list where the best (lowest) fitness of every generation will be stored.
        best_fitness = []
        # Prepare a list for average fitness values over generations.
        average_fitness = []

        # Evolution loop.
        for i in range(number_of_generations):
            # Remove directories and files produced by foldX for previous generation.
            cleanup_foldx_directories()
            
            # Evaluate fitness of the population.
            fitness = evaluate_population_fitness_parallel(population, wild_type_sequence, protein_code, i)
        
            # Sort a combined list of population and its fitness in ascending order, 
            # as lower fitness (lower ddG) means better individual.
            combined = sorted(zip(population, fitness), key=lambda x: x[1], reverse=False)
            # Print and save 5 fittest individuals in every generation to the result file.
            log_and_print(f"\nTop 5 individuals from generation number {i}:", f)
            for j, (indiviudal, fitness_value) in enumerate(combined[:5], 1):
                log_and_print(f"{j}. {indiviudal}: ddG = {fitness_value} kcal/mol", f)
            
            # Extract the best (lowest) fitness of current generation.
            best_fitness_of_generation = min(fitness)
            # Compute the average fitness of this generation.
            average_fitness_of_generation = sum(fitness) / len(fitness)
            # Print and save the average fitness of current generation to the result file.
            log_and_print(f"Average fitness of generation: {average_fitness_of_generation}", f)

            # Append the best (lowest) fitness value in this generation.
            best_fitness.append(best_fitness_of_generation)
            # Append the average fitness value in this generation.
            average_fitness.append(average_fitness_of_generation)
            # Generate new population via selection and mutations.
            population = perform_population_selection(population, fitness, restricted_sites, amino_acids, wild_type_sequence)

        # Remove directories and files produced by foldX for previous generation.
        cleanup_foldx_directories()

        # Compute fitness for the last generation and append best and average fitness values.
        last_generation_fitness = evaluate_population_fitness_parallel(population, wild_type_sequence, protein_code, number_of_generations)
        last_average_fitness = sum(last_generation_fitness) / len(last_generation_fitness)
        best_fitness.append(min(last_generation_fitness))
        average_fitness.append(last_average_fitness)
        
        # Sort a combined list of last generation and its fitness in ascending order to extract the best individual found through evolution.
        combined = sorted([(individual, fitness, index) for index, (individual, fitness) in enumerate(zip(population, last_generation_fitness))], key=lambda x: x[1], reverse=False)
        best_individual, best_fit, best_index = combined[0]
        # Save structure of the best individual produced by foldX into the working directory.
        save_best_structure(best_index, number_of_generations, protein_code)

        # Print and save the top 10 individuals along with their fitness to the result file.
        log_and_print("\nTop 10 individuals:", f)
        for i, (indiviudal, fitness_value, index) in enumerate(combined[:10], 1):
            log_and_print(f"{i}. {indiviudal}: ddG = {fitness_value} kcal/mol", f)

        # Print and save to result file the best (minimum) fitness values over generations.
        log_and_print("\nBest fitness (minimum) over generations:", f)
        log_and_print(str(best_fitness), f)

        # Plot the best fitness values and average fitness values over generations and save the plot into the working directory.
        plot_fitness(best_fitness, average_fitness, number_of_generations)
        # Remove directories and files produced by foldX at the end of the program.
        cleanup_foldx_directories()

if __name__ == "__main__":    main()
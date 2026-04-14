import warnings
import numpy as np
import random
import requests
import subprocess
from pathlib import Path
from Bio import SeqIO
import os
import glob

# Method to parse the parametrization file and extract the necessary parameters
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
    population = []

    # Select allowed positions for mutation by excluding the restricted sites from the wild-type sequence 
    # (i+1 because restricted sites are 1-based).
    allowed_positions = [i for i in range(wild_type_length) if (i+1) not in restricted_sites]

    # Create a number of individuals according to population size 
    # (number of indiviuals defined in the parametrization file).
    for _ in range(population_size):
        individual = []
        # Randomly select a position to mutate from the allowed positions.
        mutated_position = random.choice(allowed_positions)
        # Select allowed amino acids for mutation by excluding the wild-type amino acid.
        allowed_amino_acids = [amino_acid for amino_acid in amino_acids if amino_acid != wild_type_sequence[mutated_position]]
        # Randomly select an amino acid to mutate to from the allowed amino acids.
        mutated_amino_acid = random.choice(allowed_amino_acids)
        # Each individual in the population is represented as a tuple containing the 
        # mutated position and the mutated amino acid.
        single_mutation = (mutated_position, mutated_amino_acid)
        individual.append(single_mutation)
        population.append(individual.copy())
    return population

# Method to delete files produced by foldX before every foldX run.
def delete_foldx_files():
    for file in glob.glob("*.fxout") + glob.glob("*_*.pdb"):
        try:
            os.remove(file)
        except:
            pass

# Method to create a list of mutations of every individual in population, so the whole population
# will be evaulated together.
def create_mutation_list(population, wild_type_sequence):
    with open("individual_list.txt", "w") as f:
        for individual in population:
            variants = []
            for mutation in individual:
                position, mutated_amino_acid = mutation
                wild_type_amino_acid = wild_type_sequence[position]
                # Pass the variant to foldX in correct format.
                variant = f"{wild_type_amino_acid}A{position+1}{mutated_amino_acid}"
                variants.append(variant)
            
            # Add comma between the mutations (variants), and semicolon at the end of the line,
            # as this format foldX requires.
            variants_string = ','.join(variants)
            f.write(variants_string + ";" + "\n")


# Method to calculate the fitness of population using protein stability predictor.
def evaluate_population_fitness(population, protein_code):
    # Delete files produced bt foldX from previous calculation.
    delete_foldx_files()
    pdb_filename = f"{protein_code}.pdb"

    # Create a command passed to foldX to evaluate the population of mutations
    # given the wild-type structure and a list of mutations.
    cmd = [
        "./foldx/foldx_20270131",
        "--command=BuildModel",
        f"--pdb={pdb_filename}",
        "--mutant-file=individual_list.txt",
        "--noHeader=1",
        "--numberOfRuns=3"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Output foldX results in every iteration (DEBUG THING) #
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    #########################################################

    # Ensure to know if foldX fails the evaluation. If so, assign as all ddGs very large number.
    if result.returncode != 0:
        print("FoldX crashed!")
        return [1e6] * len(population)

    # List of fitness values (average ddG) for the whole population.
    fitness = []
    # List for ddG of one individual through the runs of foldX.
    ddG_over_runs = []

    output_file = f"Dif_{protein_code}.fxout"
    # Read from foldX output file with energy differencies for the mutant structures.
    with open(output_file) as f:
        lines = f.readlines()

    for line in lines:
        # Select only lines which start with the uniprot protein code, as in these 
        # lines are stored the energy differences.
        if line.startswith(protein_code):
            parts = line.split()

            if len(parts) < 2:
                continue
                
            # Extract the energy difference used as fitness.
            ddg = float(parts[1])
            ddG_over_runs.append(ddg)

            # Every 3 runs correspond to one mutation (numberOfRuns=3), thus after 3 runs compute the average ddG.
            if len(ddG_over_runs) == 3:
                # Compute the average ddG of the three runs of foldX.
                average_ddg = sum(ddG_over_runs) / 3
                # Append the energy difference to fitness list.
                # Negative ddG means the mutation is stabilizing, so it is
                # essential to minimize the fitness values.
                fitness.append(average_ddg)
                ddG_over_runs = []

    # Check if the fitness was correctly computed for the whole population.
    if(len(fitness) != len(population)): 
        print("Fitness and population size is NOT the same! Fitness length:", len(fitness), "population length:", len(population)) 

    return fitness

# Method to perform selection of individuals for the next generation using elitism and tournament selection.
def perform_population_selection(population, fitness, restricted_sites, amino_acids, wild_type_sequence, elitism_rate=0.05):
    # Combine lists of population and fitness into a single list sorted by ascending order 
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

        # With a 5% probability, create the offspring by deleting one of the individuals 
        # mutations (only if the individual has more than one mutation).
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
        
        # With a 30% probability, select one of individuals mutations and mutate its position 
        # or change its amino acid.
        else:
            offspring = change_mutation(parent, restricted_sites, amino_acids, wild_type_sequence)

        new_generation.append(offspring)

    return new_generation

# Method add one new mutation to provided individual.
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

# Method to perform mutation on an individual. 
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

# Method to print and also save to result file information from the parametrization file and
# also best individuals along with their fitness over generations.
def log_and_print(text, file):
    print(text)
    file.write(text + "\n")

def main():
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

        # Initialize random population of individuals.
        population = initialize_random_population(wild_type_sequence, wild_type_length, restricted_sites, amino_acids, number_of_individuals)

        # Print and save to result file the sequence and its length.
        log_and_print(f'Wild-Type Sequence: {wild_type_sequence}', f)
        log_and_print(f'Wild-Type Length: {wild_type_length}', f)
        log_and_print(f'Initial Population: {population}', f)

        # Prepare a list where best (lowest) fitness of every generation will be stored.
        best_fitness = []
        # Prepare a list for average fitness values over generations.
        average_fitness = []

        # Evolution loop.
        for i in range(number_of_generations):
            # Create a list of individuals and their mutations.
            create_mutation_list(population, wild_type_sequence)
            # Evaluate fitness of the population.
            fitness = evaluate_population_fitness(population, protein_code)
        
            # Print and save to result file the  5 fittest individuals in every generation.
            # Sort in ascending order, as lower fitness (lower ddG) means better individual.
            combined = sorted(zip(population, fitness), key=lambda x: x[1], reverse=False)
            log_and_print(f"\nTop 5 individuals from generation number {i}:", f)
            for j, (indiviudal, fitness_value) in enumerate(combined[:5], 1):
                log_and_print(f"{j}. {indiviudal}: ddG = {fitness_value} kcal/mol", f)
            
            # Extract the best (lowest) fitness of current generation.
            best_fitness_of_generation = min(fitness)
            # Compute the average fitness of this generation.
            average_fitness_of_generation = sum(fitness) / len(fitness)
            # Print and save to result file average fitness of current generation.
            log_and_print(f"Average fitness of generation: {average_fitness_of_generation}", f)

            # Append the best (lowest) fitness value in this generation.
            best_fitness.append(best_fitness_of_generation)
            # Append the average fitness value in this generation.
            average_fitness.append(average_fitness_of_generation)
            # Generate new population via selection and mutations.
            population = perform_population_selection(population, fitness, restricted_sites, amino_acids, wild_type_sequence)
    
        # Extract fitness values for the last generation.
        create_mutation_list(population, wild_type_sequence)
        last_generation_fitness = evaluate_population_fitness(population, protein_code)
        last_average_fitness = sum(last_generation_fitness) / len(last_generation_fitness)
        best_fitness.append(min(last_generation_fitness))
        average_fitness.append(last_average_fitness)
        combined = sorted(zip(population, last_generation_fitness), key=lambda x: x[1], reverse=False)

        # Print and save to result file the top 10 individuals along with their fitness.
        log_and_print("\nTop 10 individuals:", f)
        for i, (indiviudal, fitness_value) in enumerate(combined[:10], 1):
            log_and_print(f"{i}. {indiviudal}: ddG = {fitness_value} kcal/mol", f)

        # Print and save to result file the best (minimum) fitness values over generations.
        log_and_print("\nMax fitness over generations:", f)
        log_and_print(str(best_fitness), f)

        delete_foldx_files()


if __name__ == "__main__":    main()
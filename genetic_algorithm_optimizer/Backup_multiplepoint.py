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

# Method to initialize a random population of mutants.
def random_population_initialization(wild_type_sequence, wild_type_length, restricted_sites, amino_acids, population_size):
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
            
            variants_string = ','.join(variants)
            f.write(variants_string + ";" + "\n")


# Method to calculate the fitness of population using protein stability predictor.
def fitness_function(population, protein_code):
    delete_foldx_files()
    pdb_filename = f"{protein_code}.pdb"

    # Create a command passed to foldX to evaluate the population of mutations
    # given the wild-type strcuture and a list of mutations.
    cmd = [
        "./foldx/foldx_20270131",
        "--command=BuildModel",
        f"--pdb={pdb_filename}",
        "--mutant-file=individual_list.txt",
        "--noHeader=1"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Output foldX results in every iteration (DEBUG THING).
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Ensure to know if foldX fails the evaluation.
    if result.returncode != 0:
        print("FoldX crashed!")
        return [1e6] * len(population)

    fitness = []

    try:
        output_file = f"Dif_{protein_code}.fxout"
        # Read from foldX output file with energy differencies for the mutant structures.
        with open(output_file) as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith(protein_code):
                parts = line.split()

                if len(parts) < 2:
                    continue
                
                # Extract the energy difference used as fitness.
                ddg = float(parts[1])

                # Append the energy difference to fitness list with opposite sign.
                # Negative ddG means the mutation is stabilizing, but we want to 
                # maximaze the fitness so pass ddG with opposite sign.
                fitness.append(-ddg)
        # Check if the fitness was correctly computed for the whole population.
        if(len(fitness) != len(population)): 
            print("Fitness and population size is NOT the same! Fitness length:", len(fitness), "population length:", len(population)) 
    except Exception as e: 
        print("Error reading ddG:", e) 
        return [1e6] * len(population)

    return fitness

# Method to perform selection of individuals for the next generation using elitism and tournament selection.
def selection(population, fitness, restricted_sites, amino_acids, wild_type_sequence, elitism_rate=0.2):
    # Combine population and fitness into a single list of tuples.
    combined = list(zip(population, fitness))
    # Sort the combined list by fitness in descending order (maximum fitness first).
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True) 
    # Calculate the number of elite individuals to keep and extract them to next generation.
    elite_count = max(1, int(len(population) * elitism_rate)) 
    elite_individuals = [individual for individual, _ in sorted_combined[:elite_count]] 
    # Extract the remaining individuals for selection.
    remaining_individuals = sorted_combined[elite_count:] 

    # For remaining individuals perform the tournament selection.
    selected_population = []
    # To prevent infinite loops in case of issues with mutation or selection.
    attempts = 0
    # To prevent infinite loops in case of issues with mutation or selection.
    max_attempts = 1000  

    while len(selected_population) < len(population) - elite_count and attempts < max_attempts:
        probability = random.random()
        # Choose the fittest parent from 3 randomly picked individuals.
        tournament = random.sample(remaining_individuals, 3) 
        parent = max(tournament, key=lambda x: x[1])[0] 

        # With 10% probability, create the offspring by deleting
        # one of the individuals mutations (only if the individual has more than one mutation).
        if probability < 0.1 and len(parent) > 1:
            upper_bound = len(parent) - 1
            delete_index = random.randint(0, upper_bound)
            offspring = parent.copy()
            offspring.pop(delete_index)
        
        # With 20% probability, create the offspring by combining mutations
        # from two different parents.
        elif 0.1 <= probability < 0.3:
            second_tournament = random.sample(remaining_individuals, 3)
            second_parent = max(second_tournament, key=lambda x: x[1])[0]
            offspring = crossover(parent, second_parent)
        
        # With 30% probability, create the offspring by inheriting randomly one mutation
        # from the parent and obtaining one new mutation.
        elif 0.3 <= probability < 0.6 and len(parent) < 3:
            offspring = add_new_mutation(parent, restricted_sites, amino_acids, wild_type_sequence)
        
        # With 40% probability, select one of individuals mutations and mutate its position 
        # or change its amino acid.
        else:
            offspring = mutation(parent, restricted_sites, amino_acids, wild_type_sequence)

        # Ensure the offspring is unique in the next generation.
        if offspring not in selected_population and offspring not in elite_individuals: 
            selected_population.append(offspring)
        attempts += 1
    
    next_generation = elite_individuals + selected_population  
    return next_generation

def add_new_mutation(parent, restricted_sites, amino_acids, wild_type_sequence):
    offspring = []
    # Select randomly which mutation from parent will be inherited.
    parent_upper_bound = len(parent) - 1
    parent_mutation_index = random.randint(0, parent_upper_bound)
    parent_mutation = parent[parent_mutation_index]
    offspring.append(parent_mutation)

    # Extract the position and amino acid of the parent mutation.
    parent_position, parent_amino_acid = parent_mutation
    # Define allowed positions which are not in restricted sites or the position is not the parent mutation.
    allowed_positions = [i for i in range(len(wild_type_sequence)) if (i+1) not in restricted_sites and i != parent_position]
    # Randomly select a position to mutate from the allowed positions.
    mutated_position = random.choice(allowed_positions)
    # Define allowed amino acids for mutation by excluding the wild-type amino acid and the parent amino acid.
    allowed_amino_acids = [amino_acid for amino_acid in amino_acids if amino_acid != wild_type_sequence[mutated_position] and amino_acid != parent_amino_acid]
    # Randomly select an amino acid to mutate to from the allowed amino acids.
    mutated_amino_acid = random.choice(allowed_amino_acids)
    # Each individual in the population is represented as a tuple containing the 
    # mutated position and the mutated amino acid.
    single_mutation = (mutated_position, mutated_amino_acid)
    offspring.append(single_mutation)

    return offspring


def crossover(first_parent, second_parent):
    offspring = []

    # Choose randomly which mutation from the first parent will be 
    # passed to the offspring.
    first_parent_upper_bound = len(first_parent) - 1
    first_parent_mutation_index = random.randint(0, first_parent_upper_bound)
    first_parent_mutation = first_parent[first_parent_mutation_index]
    offspring.append(first_parent_mutation)

    # Choose randomly which mutation from the second parent will be 
    # passed to the offspring.
    second_parent_upper_bound = len(second_parent) - 1
    second_parent_mutation_index = random.randint(0, second_parent_upper_bound)
    second_parent_mutation = second_parent[second_parent_mutation_index]
    # Check if the two mutation are not in the same position. If so, do not add the second mutation.
    if second_parent_mutation[0] != first_parent_mutation[0]:
        offspring.append(second_parent_mutation)
    
    return offspring

# Method to perform mutation on an individual. 
def mutation(individual, restricted_sites, amino_acids, wild_type_sequence):
    offspring = individual.copy()
    offspring_upper_bound = len(offspring) - 1
    # Choose randomly which mutation will be modified.
    offspring_mutation_index = random.randint(0, offspring_upper_bound)
    offspring_mutation = offspring[offspring_mutation_index]

    mutated_position, mutated_amino_acid = offspring_mutation
    probability = random.random()
    # With a 40% chance, mutate the position of the mutation while keeping the amino acid the same.
    if probability < 0.4:
        allowed_positions = [i for i in range(len(wild_type_sequence)) if (i+1) not in restricted_sites and i != mutated_position]
        mutated_position = random.choice(allowed_positions)
    # With a 60% chance, mutate the amino acid while keeping the same position of the mutation.
    else:  
        allowed_amino_acids = [amino_acid for amino_acid in amino_acids if amino_acid != wild_type_sequence[mutated_position] and amino_acid != mutated_amino_acid]
        mutated_amino_acid = random.choice(allowed_amino_acids)
    
    modified_mutation = (mutated_position, mutated_amino_acid) 
    offspring[offspring_mutation_index] = modified_mutation
    return offspring

def main():
    file_path = 'Parametrization_file.txt'
    protein_code, restricted_sites, amino_acids, number_of_individuals, number_of_generations = parse_parametization_file(file_path)

    print(f'Protein Code: {protein_code}')
    print(f'Restricted Sites: {restricted_sites}')
    print(f'Amino Acids: {amino_acids}')
    print(f'Number of Individuals: {number_of_individuals}')
    print(f'Number of Generations: {number_of_generations}')

    fasta_path, pdb_path = download_protein_data(protein_code)

    wild_type_sequence, wild_type_length = get_wild_type_sequence(fasta_path)
    print(f'Wild-Type Sequence: {wild_type_sequence}')
    print(f'Wild-Type Length: {wild_type_length}')

    population = random_population_initialization(wild_type_sequence, wild_type_length, restricted_sites, amino_acids, number_of_individuals)
    print(f'Initial Population: {population}')
    max_fitness = []

    for _ in range(number_of_generations):
        create_mutation_list(population, wild_type_sequence)
        fitness = fitness_function(population, protein_code)
        print(list(zip(population[:5], fitness[:5])))
        max_fitness.append(max(fitness))
        population = selection(population, fitness, restricted_sites, amino_acids, wild_type_sequence)
    
    create_mutation_list(population, wild_type_sequence)
    last_gen_fitness = fitness_function(population, protein_code)
    max_fitness.append(max(last_gen_fitness))
    
    # Combine population with fitness
    combined = list(zip(population, last_gen_fitness))

    # Sort by fitness (descending)
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

    # Print top 10
    top_n = 10
    print(f"\nTop {top_n} individuals:")
    for i, (individual, fit) in enumerate(sorted_combined[:top_n], 1):
        print(f"{i}. Individual: {individual}, Fitness: {fit}")

    print(f'Max Fitness over Generations: {max_fitness}')


if __name__ == "__main__":    main()
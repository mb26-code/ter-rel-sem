import csv
from collections import Counter



# File path to the actual CSV (uncomment and update with your file path)
FILE_PATH = "semantic_relations_rows.csv"

# Threshold for frequency of terms to be considered culinary
FREQUENCY_THRESHOLD = 3

# Useful dependency types for culinary relations
USEFUL_DEPS = ['nmod', 'amod', 'compound', 'nsubj', 'obj']

def extract_culinary_terms(csv_data=None, file_path=None):
    # Initialize counters and term lists
    term_counts = Counter()
    culinary_terms = set()

    # Read the CSV data
    if file_path:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
    else:
        csv_file = csv.DictReader(csv_data.splitlines())
        rows = list(csv_file)

    # Step 1: Extract and count terms from node1 and node2
    for row in rows:
        node1 = row['node1'].lower().strip()
        node2 = row['node2'].lower().strip()
        pos = row.get('pos', '').strip()
        dep = row.get('dep', '').strip()

        # Only consider terms where pos is NOUN and dep is useful
        if pos == 'NOUN' and dep in USEFUL_DEPS:
            term_counts[node1] += 1
            term_counts[node2] += 1

    # Step 2: Filter terms based on frequency
    for term, count in term_counts.items():
        if count >= FREQUENCY_THRESHOLD:
            culinary_terms.add(term)

    # Step 3: Sort terms and format as a Python list
    culinary_terms = sorted(list(culinary_terms))
    return culinary_terms

def generate_culinary_terms_list():
    # Extract terms (use file_path=FILE_PATH if reading from file)
    terms = extract_culinary_terms(file_path=FILE_PATH)
    # Replace with file_path=FILE_PATH as needed

    # Format as Python code
    terms_str = ",\n    ".join(f'"{term}"' for term in terms)
    python_code = f"CULINARY_TERMS = [\n    {terms_str}\n]"
    return python_code

# Generate and print the new CULINARY_TERMS list
if __name__ == "__main__":
    new_culinary_terms = generate_culinary_terms_list()
    print(new_culinary_terms)  # <- Affiche dans la console

    # Écrit aussi dans un fichier Python
    with open("generated_culinary_terms.py", "w", encoding="utf-8") as f:
        f.write(new_culinary_terms)
    print("✅ Fichier 'generated_culinary_terms.py' créé avec succès.")

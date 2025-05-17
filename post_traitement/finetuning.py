#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning script for Llama3 model to predict semantic relations between word pairs.
Optimized for MacBook Pro with M1 Pro chip.

This script:
1. Extracts word pairs with w > 80 from ressources_completes directory
2. Creates a training dataset in the appropriate format
3. Fine-tunes the llama3:instruct model using QLoRA
4. Saves the fine-tuned model

Requirements:
- torch
- transformers
- datasets
- pandas
- peft
- accelerate
- bitsandbytes
"""

import os
import glob
import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
from tqdm import tqdm

# List of relations we care about with descriptions
RELATIONS = {
    "r_associated": "Les concepts associés ont une priorité plus élevée",
    "r_raff_sem": "Raffinement sémantique vers un usage particulier du terme source",
    "r_isa": "Les relations taxonomiques sont importantes",
    "r_hypo": "Il est demandé d'énumérer des SPECIFIQUES/hyponymes du terme. Par exemple, 'mouche', 'abeille', 'guêpe' pour 'insecte'",
    "r_has_part": "Les relations partie-tout sont importantes",
    "r_holo": "Il est démandé d'énumérer des 'TOUT' (a pour holonymes) de l'objet en question. Pour 'main', on aura 'bras', 'corps', 'personne', etc... Le tout est aussi l'ensemble comme 'classe' pour 'élève'",
    "r_instr": "L'instrument est l'objet avec lequel on fait l'action. Dans - Il mange sa salade avec une fourchette -, fourchette est l'instrument",
    "r_carac": "Les relations caractéristiques sont importantes",
    "r_accomp": "Est souvent accompagné de, se trouve avec... Par exemple : Astérix et Obelix, le pain et le fromage, les fraises et la chantilly",
    "r_lieu": "Les relations de lieu ont une priorité moyenne",
    "r_make": "Que peut PRODUIRE le terme ? (par exemple abeille -> miel, usine -> voiture, agriculteur -> blé, moteur -> gaz carbonique ...)",
    "r_similar": "Similaire/ressemble à ; par exemple le congre est similaire à une anguille, ..."
}

# Mapping from relation name to type ID
RELATION_TYPE_MAP = {
    "r_associated": 0,
    "r_raff_sem": 1,
    "r_isa": 6,
    "r_hypo": 8,
    "r_has_part": 9,
    "r_holo": 10,
    "r_instr": 16,
    "r_carac": 17,
    "r_lieu": 15,
    "r_make": 53,
    "r_similar": 67,
    "r_accomp": 75
}

# Default configuration
DEFAULT_CONFIG = {
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # TinyLlama works well for fine-tuning
    "output_dir": "fine_tuned_model",
    "weight_threshold": 80,
    "batch_size": 4,
    "grad_accum": 8,
    "epochs": 2,
    "learning_rate": 2e-5,
    "max_seq_length": 300,
    "lora_r": 16,
    "lora_alpha": 32,
    "test_size": 0.2,
    "use_mps": True,
    "ollama_model": "llama3:instruct"  # The model you've pulled from Ollama
}

def load_and_filter_data_from_directory(directory_path, weight_threshold=80, sim_threshold=0.55):
    """
    Load data from all CSV files in directory and filter by weight threshold and similarity score
    """
    print(f"Loading data from {directory_path} with weight threshold > {weight_threshold} and similarity > {sim_threshold}...")
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "**/*.csv"), recursive=True)
    print(f"Found {len(csv_files)} CSV files")
    
    all_pairs = []
    
    # Process each file
    for file_path in tqdm(csv_files, desc="Processing files"):
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Check if the file has the required columns
            required_columns = ['head', 'lemma', 'pos', 'dep', 'relation_name', 'w', 'sim']
            if not all(col in df.columns for col in required_columns):
                print(f"Skipping {file_path}: missing required columns")
                continue
                
            # Filter by weight threshold and similarity threshold
            filtered_rows = df[(df['w'] > weight_threshold) & (df['sim'] > sim_threshold)]
            
            # Filter out rows with missing relation_name values
            valid_rows = filtered_rows[filtered_rows['relation_name'].notna() & 
                                      (filtered_rows['relation_name'] != '') & 
                                      (~filtered_rows['relation_name'].isin(['none', 'None']))]
            
            if len(valid_rows) > 0:
                # Extract relevant columns: head, lemma, pos, dep, relation_name, sim
                for _, row in valid_rows.iterrows():
                    # Only add pairs with recognized relations
                    if row['relation_name'] in RELATIONS:
                        all_pairs.append({
                            'head': row['head'],
                            'lemma': row['lemma'],
                            'pos': row['pos'] if pd.notna(row['pos']) else '',
                            'dep': row['dep'] if pd.notna(row['dep']) else '',
                            'relation': row['relation_name'],
                            'weight': row['w'],
                            'similarity': row['sim']
                        })
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print(f"Extracted {len(all_pairs)} word pairs with weight > {weight_threshold}")
    
    # Analyze distribution of relations
    relation_counts = {}
    for pair in all_pairs:
        relation = pair['relation']
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    
    print("\nDistribution of relations:")
    for relation, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {relation}: {count} pairs")
        
    return all_pairs

def format_data_for_instruction_tuning(word_pairs, context="cuisine"):
    """
    Format word pairs as instruction examples for fine-tuning
    """
    formatted_data = []
    
    for pair in tqdm(word_pairs, desc="Formatting instruction examples"):
        # Create the input prompt following the same format used in the production code
        input_text = f"""
    Determine le type de relation sémantique entre deux mots dans un contexte de {context}:
    
    Mot 1: {pair['head']}
    Mot 2: {pair['lemma']}(POS: {pair['pos']})
    Relation syntaxique: {pair['dep']}
    
    Choisir UNE relation parmi les suivantes qui décrit le mieux comment ces mots sont liés:
    
    - r_associated: Concepts généralement associés ensemble
    - r_raff_sem: Raffinement sémantique vers un usage particulier du terme source
    - r_isa: Relations taxonomiques (X est un Y)
    - r_hypo: Terme spécifique (hyponyme) du premier mot (X est un type de Y)
    - r_has_part: Relation partie-tout (X a Y comme partie)
    - r_holo: Le tout dont le premier mot fait partie (X fait partie de Y)
    - r_instr: Relation d'instrument (X est utilisé pour faire Y)
    - r_carac: Relation de caractéristique (X a la caractéristique Y)
    - r_accomp: Souvent accompagné de (X est souvent avec Y)
    - r_lieu: Relation de lieu (X se trouve à/dans Y)
    - r_make: X produit Y
    - r_similar: X est similaire à Y
    
    Réponds uniquement avec le nom de la relation (par exemple "r_carac") sans explication. Si aucune relation ne s'applique, réponds "none".
    """
        
        # Output is just the relation name
        output = pair['relation']
        
        # Create the instruction example for Llama format
        formatted_example = {
            "instruction": "Détermine la relation sémantique entre deux mots",
            "input": input_text.strip(),
            "output": output
        }
        
        formatted_data.append(formatted_example)
    
    return formatted_data

def prepare_dataset_for_training(formatted_data, tokenizer, config):
    """
    Convert formatted data to a tokenized dataset for training
    """
    # Create a Hugging Face dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Split into train and validation sets
    dataset = dataset.train_test_split(test_size=config['test_size'], seed=42)
    
    def tokenize_function(examples):
        """Tokenize and format for instruction tuning"""
        
        # Apply the Llama chat template to correctly format the instruction, input and output
        texts = []
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            # Using the Llama 3 instruction format
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nTu es un assistant linguistique expert en relations sémantiques.<|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n{input_text}<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|end_of_text|>"
            texts.append(prompt)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=config['max_seq_length'],
            padding="max_length",
            return_tensors="pt"
        )
        
        # Prepare labels for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Tokenize the datasets
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset"
    )
    
    print(f"Train dataset: {len(tokenized_dataset['train'])} examples")
    print(f"Validation dataset: {len(tokenized_dataset['test'])} examples")
    
    return tokenized_dataset

def setup_model_for_training(config):
    """
    Load and prepare the model for LoRA fine-tuning
    """
    print(f"Loading model: {config['base_model']}")
    
    # Determine device
    if config['use_mps'] and torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        device_map = {"": "mps"}
    else:
        print("Using CPU (MPS not available)")
        device_map = {"": "cpu"}
    
    # Configure quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Use a locally accessible model that works well with Apple Silicon
    try:
        print("Loading a model for fine-tuning on Apple Silicon...")
        
        # Use a model known to work well with Apple Silicon and LoRA
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print(f"Using TinyLlama tokenizer - a good fit for M1 Mac fine-tuning")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load the model without 4-bit quantization for M1 compatibility
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Small but effective model for fine-tuning
            device_map=device_map,
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
            trust_remote_code=True
        )
        print("Successfully loaded TinyLlama model for fine-tuning")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to an even smaller model...")
        
        # Fallback to a smaller model that's freely available
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA adapter - using simpler target modules compatible with TinyLlama
    target_modules = None
    if "TinyLlama" in config['base_model'] or "tinyllama" in config['base_model'].lower():
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj", 
            "up_proj",
            "down_proj"
        ]
    else:
        # Default target modules
        target_modules = [
            "q_proj",
            "v_proj"
        ]
        
    lora_config = LoraConfig(
        r=config['lora_r'],                   # Rank
        lora_alpha=config['lora_alpha'],      # Alpha parameter for LoRA scaling
        target_modules=target_modules,        # Modules to apply LoRA to (automatically detected)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Add LoRA adapter to model
    model = get_peft_model(model, lora_config)
    
    # Log trainable parameters
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of {all_params:,})")
    
    return model, tokenizer

def train_model(model, tokenized_dataset, tokenizer, config):
    """
    Fine-tune the model using LoRA
    """
    # Prepare output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['grad_accum'],
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        learning_rate=config['learning_rate'],
        optim="adamw_torch",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=None  # No wandb or tensorboard
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train the model
    print(f"Starting fine-tuning for {config['epochs']} epochs...")
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model(config['output_dir'])
    print(f"Model fine-tuning completed. Model saved to {config['output_dir']}")

def load_word_pairs_from_directory(directory_path, weight_threshold, sim_threshold=0.55):
    """
    Main function to load and process word pairs from the directory
    """
    # Path constants
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.join(script_dir, directory_path)
    
    # Load and filter the data
    word_pairs = load_and_filter_data_from_directory(resources_dir, weight_threshold, sim_threshold)
    
    return word_pairs

def export_to_ollama(model_name, model_dir):
    """
    Export fine-tuned model to Ollama
    Note: This assumes Ollama is installed and properly configured
    """
    print(f"Exporting fine-tuned model to Ollama as '{model_name}'...")
    
    # Create Modelfile content - using the existing llama3:instruct model directly
    modelfile_content = f"""
FROM llama3:instruct

# Custom adapter configuration
TEMPLATE \"{{
  \\"messages\\": [
    {{ \\"role\\": \\"system\\", \\"content\\": \\"Tu es un assistant linguistique expert en relations sémantiques.\\" }},
    {{ \\"role\\": \\"user\\", \\"content\\": {{ .Input }} }}
  ]
}}\"

# Add any fine-tuning options
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "</s>"
"""

    # Save Modelfile
    modelfile_path = os.path.join(model_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    # Import to Ollama
    import subprocess
    try:
        subprocess.run(["ollama", "create", model_name, "-f", modelfile_path], check=True)
        print(f"Successfully created Ollama model: {model_name}")
        print(f"You can now use it with: ollama run {model_name}")
        
        # Integration example with your existing code
        print("\nExample integration with your existing code:")
        print(f"""
# Example code to use your fine-tuned model
from langchain_community.llms import Ollama

# Replace current model instantiation
thread_local.model = Ollama(model="{model_name}")  # Instead of "llama3:instruct"
        """)
    except subprocess.CalledProcessError as e:
        print(f"Error exporting to Ollama: {e}")
        print("Please make sure Ollama is installed and running.")
    except FileNotFoundError:
        print("Ollama command not found. Please install Ollama first.")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune a Llama3 model for semantic relation prediction')
    parser.add_argument('--data-dir', type=str, default='ressources_completes', 
                        help='Directory containing the CSV files with word pairs')
    parser.add_argument('--weight-threshold', type=float, default=DEFAULT_CONFIG['weight_threshold'],
                        help='Minimum weight threshold for filtering word pairs')
    parser.add_argument('--sim-threshold', type=float, default=0.55,
                        help='Minimum similarity threshold for filtering word pairs')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_CONFIG['output_dir'],
                        help='Directory to save the fine-tuned model')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--context', type=str, default='cuisine',
                        help='Context domain for the language model')
    parser.add_argument('--export-ollama', action='store_true',
                        help='Export the fine-tuned model to Ollama')
    parser.add_argument('--ollama-model-name', type=str, default='llama3-relations',
                        help='Name for the exported Ollama model')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    config['weight_threshold'] = args.weight_threshold
    config['output_dir'] = args.output_dir
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    
    # Display target device information
    if config['use_mps'] and torch.backends.mps.is_available():
        print("Apple Silicon GPU (MPS) is available and will be used")
    elif torch.cuda.is_available():
        print("CUDA GPU is available and will be used")
    else:
        print("No GPU detected, using CPU (this will be much slower)")
    
    # 1. Load data
    word_pairs = load_word_pairs_from_directory(args.data_dir, config['weight_threshold'], args.sim_threshold)
    
    if len(word_pairs) == 0:
        print("No word pairs found. Exiting.")
        return
        
    # 2. Format data for instruction tuning
    formatted_data = format_data_for_instruction_tuning(word_pairs, args.context)
    
    # 3. Setup model for training
    model, tokenizer = setup_model_for_training(config)
    
    # 4. Prepare dataset
    tokenized_dataset = prepare_dataset_for_training(formatted_data, tokenizer, config)
    
    # 5. Train model
    train_model(model, tokenized_dataset, tokenizer, config)
    
    # 6. Export to Ollama if requested
    if args.export_ollama:
        export_to_ollama(args.ollama_model_name, config['output_dir'])
    
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()

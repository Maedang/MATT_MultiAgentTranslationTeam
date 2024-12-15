import json
import os
import random
from dotenv import load_dotenv
import sys
import time
import logging
from anthropic import Anthropic

load_dotenv()
LLM_model = "claude-3-5-sonnet-20241022" # Replace with the preferable model name\
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))



    
def load_data(filename):
    """
    Load data from a JSON file and return a dictionary mapping IDs to a tuple of (source_text, translation).
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Assuming data is a list of dictionaries with 'source_text' and 'translation' keys
    if isinstance(data, list) and isinstance(data[0], dict):
        # Start IDs from 1 to 100
        data_dict = {str(idx + 1): (item['source_text'], item['translation']) for idx, item in enumerate(data)}
    else:
        raise ValueError(f"Unsupported data format in {filename}. Expected a list of dictionaries with 'source_text' and 'translation' keys.")
    
    return data_dict

def load_existing_results(output_filepath):
    """
    Load existing results from a JSON file and return a set of processed IDs.
    If the file does not exist, create it with an empty list.
    """
    if not os.path.exists(output_filepath):
        # Create the file with an empty list
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        return set()
    
    with open(output_filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            processed_ids = {entry['id'] for entry in data if 'id' in entry}
            return processed_ids
        except json.JSONDecodeError:
            logging.warning(f"Malformed JSON in {output_filepath}. Starting with an empty list.")
            return set()

def append_result(output_filepath, result):
    """
    Append a single result to a JSON file.
    """
    try:
        with open(output_filepath, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            data.append(result)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.truncate()
    except json.JSONDecodeError:
        logging.error(f"Malformed JSON in {output_filepath}. Cannot append result.")
    except Exception as e:
        logging.error(f"Error appending result to {output_filepath}: {e}")

def setup_logging():
    """
    Configure the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("evaluation.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_evaluation_response(response):
    """
    Parse Claude's response to extract the evaluation option.
    Expected options: 'A', 'B', 'Tie', 'None'
    """
    response = response.strip().lower()

    if "translation a" in response:
        return 'A'
    elif "translation b" in response:
        return 'B'
    elif "Tie" in response:
        return 'Tie'
    elif "None" in response:
        return 'None'
    else:
        return "Could not determine"

def main():
    setup_logging()
    
    # Define models, sizes, and languages
    models = [
        'Baseline',
        # 'Multiagents',
        'Google'
    ]
    sizes = [
        'chunks', 
        # 'flores'
    ]
    languages = {
        # "vie": ["Vietnam", "Vietnamese"],
        "hin": ["India", "Hindi"],
        # "mal": ["India", "Malayalam"],
    }

    # Loop over sizes and languages
    for size in sizes:
        for lang_code, lang_info in languages.items():
            target_lang = lang_info[1]
            logging.info(f"Processing Size: '{size}', Language: '{target_lang}' ({lang_code})")

            # Load translation data for each model
            translations = {}
            source_texts = {}
            for model_name in models:
                filepath = f'../translations/{model_name}/{model_name}_{size}_{lang_code}.json'
                if not os.path.exists(filepath):
                    logging.warning(f"Translation file {filepath} not found. Skipping model '{model_name}'.")
                    continue
                try:
                    translations[model_name] = load_data(filepath)
                    logging.info(f"Loaded translations for model '{model_name}' from '{filepath}'")
                except Exception as e:
                    logging.error(f"Error loading data from {filepath}: {e}")
                    continue
                
                # Extract source texts from the first model loaded
                if not source_texts:
                    source_texts = {idx: data[0] for idx, data in translations[model_name].items()}

            # Get common IDs between all models
            common_ids = set(source_texts.keys())
            for t in translations.values():
                common_ids &= set(t.keys())

            # Prepare model pairs
            model_pairs = [(m1, m2) for i, m1 in enumerate(models) for m2 in models[i+1:]]
            logging.info(f"Model pairs to evaluate: {model_pairs}")

            # Evaluate each pair
            for model1, model2 in model_pairs:
                if model1 not in translations or model2 not in translations:
                    logging.warning(f"Missing translations for models '{model1}' or '{model2}'. Skipping pair...")
                    continue

                translation1_data = translations[model1]
                translation2_data = translations[model2]

                # Define output file path
                output_dir = 'claude_evaluation_results'
                os.makedirs(output_dir, exist_ok=True)
                output_filename = f'evaluation_{model1}_vs_{model2}_{size}_{lang_code}.json'
                output_filepath = os.path.join(output_dir, output_filename)

                # Load existing results
                processed_ids = load_existing_results(output_filepath)
                logging.info(f"Found {len(processed_ids)} already processed IDs in '{output_filename}'.")

                # Sort and filter IDs
                sorted_common_ids = sorted(common_ids, key=lambda x: int(x))
                remaining_ids = [idx for idx in sorted_common_ids if idx not in processed_ids]

                if not remaining_ids:
                    logging.info(f"All sentences already evaluated for pair '{model1}' vs '{model2}'. Skipping...")
                    continue

                # Evaluate each remaining ID
                for idx in remaining_ids:
                    source_text = source_texts[idx]
                    translation1 = translation1_data[idx][1]
                    translation2 = translation2_data[idx][1]

                    # Randomly assign translations
                    translations_list = [('A', translation1, model1), ('B', translation2, model2)]
                    random.shuffle(translations_list)
                    translation_a_label, shuffled_translation_a, model_a = translations_list[0]
                    translation_b_label, shuffled_translation_b, model_b = translations_list[1]

                    # Construct the prompt for Claude
                    prompt = f"""As a translation expert, evaluate these two translations of the same text.

Source text (English):
"{source_text}"

Translation A: "{shuffled_translation_a}"
Translation B: "{shuffled_translation_b}"

Evaluate which translation matches the meaning of the english sentence, called {source_text} best considering the accuracy, fluency, terminology, style, and cultural adaptability. Only respond with one of the following answers: Translation A,Translation B, Tie, None

"""

                    try:
                        # Generate Claude's response
                        message = client.messages.create(
                            model=LLM_model,
                            max_tokens=10,
                            temperature=0,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ]
                        )
                        
                        # Extract and process response
                        evaluation_response = message.content[0].text
                        logging.info(f"ID {idx}: {evaluation_response}")
                        
                        # Parse the evaluation response
                        evaluation = parse_evaluation_response(evaluation_response)
                        
                        # Map the evaluation to the winner
                        winner_map = {
                            'A': model_a,
                            'B': model_b,
                            'Tie': 'Tie',
                            'None': 'None',
                            'Could not determine': 'Could not determine'
                        }
                        winner = winner_map.get(evaluation)

                        # Prepare and save result
                        result = {
                            "id": idx,
                            "winner": winner,
                            "model_A": model_a, 
                            "model_B": model_b, 
                            "evaluation": evaluation,
                            "model_response": evaluation_response
                        }
                        append_result(output_filepath, result)
                        logging.info(f"Appended result for ID {idx} to '{output_filename}'.")

                    except Exception as e:
                        logging.error(f"Error evaluating translations for ID {idx}: {e}")
                        continue

                    # Rate limiting
                    time.sleep(0.5)  # Adjust based on your API rate limits

                logging.info(f"Completed evaluations for pair '{model1}' vs '{model2}'.")

if __name__ == '__main__':
    main()
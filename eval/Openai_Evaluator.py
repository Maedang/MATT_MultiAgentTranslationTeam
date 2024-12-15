import json
import os
import random
from groq import Groq  
from dotenv import load_dotenv
import sys
import time
import logging
import openai
from pydantic import BaseModel, Field
from typing import Literal, List

load_dotenv()

# Set the API key
openai.api_key = os.getenv("OPENAI_API_KEY")


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

def main():
    setup_logging()
    
    # Define models, sizes, and languages
    models = [
        'Baseline', 
        'Multiagents',
        #'Google'
              ]
    sizes = [
    'chunks', 
    'flores'
    ]
    languages = {
        # "eng": ["United States", "English"],
        "vie": ["Vietnam", "Vietnamese"],
        "hin": ["India", "Hindi"],
        "mal": ["India", "Malayalam"],
    }
    # Loop over sizes and languages
    for size in sizes:
        for lang_code, lang_info in languages.items():
            source_lang = 'English'
            country = lang_info[0]
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

            # Ensure at least two models have data
            if len(translations) < 2:
                logging.warning(f"Not enough translation files for size '{size}' and language '{target_lang}'. Skipping...")
                continue

            # Get common IDs
            common_ids = set(source_texts.keys())
            for t in translations.values():
                common_ids &= set(t.keys())

            if not common_ids:
                logging.warning(f"No common IDs found for size '{size}' and language '{target_lang}'. Skipping...")
                continue

            # Prepare model pairs (e.g., Baseline vs Multiagents)
            model_pairs = [(m1, m2) for i, m1 in enumerate(models) for m2 in models[i+1:]]
            logging.info(f"Model pairs to evaluate: {model_pairs}")

            # Evaluate each pair
            for model1, model2 in model_pairs:
                if model1 not in translations or model2 not in translations:
                    logging.warning(f"Missing translations for models '{model1}' or '{model2}'. Skipping pair...")
                    continue

                translation1_data = translations[model1]
                translation2_data = translations[model2]

                # Define output file path for the model pair
                output_dir = 'openai_evaluation_results_3'
                os.makedirs(output_dir, exist_ok=True)
                output_filename = f'evaluation_{model1}_vs_{model2}_{size}_{lang_code}.json'
                output_filepath = os.path.join(output_dir, output_filename)

                # Load existing results to determine processed IDs
                processed_ids = load_existing_results(output_filepath)
                logging.info(f"Found {len(processed_ids)} already processed IDs in '{output_filename}'.")

                # Sort the common IDs to ensure order from 1 to 100
                sorted_common_ids = sorted(common_ids, key=lambda x: int(x))

                # Filter out already processed IDs
                remaining_ids = [idx for idx in sorted_common_ids if idx not in processed_ids]
                logging.info(f"Evaluating {len(remaining_ids)} remaining sentences for pair '{model1}' vs '{model2}'.")

                if not remaining_ids:
                    logging.info(f"All sentences already evaluated for pair '{model1}' vs '{model2}'. Skipping...")
                    continue

                # Evaluate each remaining ID
                for idx in remaining_ids:
                    source_text = source_texts[idx]
                    translation1 = translation1_data[idx][1]
                    translation2 = translation2_data[idx][1]

                    # Randomly assign translations to Translation A and B
                    translations_list = [(translation1, model1), (translation2, model2)]
                    random.shuffle(translations_list)
                    shuffled_translation_a, model_a = translations_list[0]
                    shuffled_translation_b, model_b = translations_list[1]

                    # Construct the prompt with detailed criteria
                    prompt = f"""You are an expert in evaluating translation quality.

Your task is to carefully read a source text and two translations from {source_lang} to {target_lang}, and choose which translation is best keeps the meaning of the {source_text}. \

The source text and two translations, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION_A></TRANSLATION_A>, and <TRANSLATION_A></TRANSLATION_A> are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION_A>
{shuffled_translation_a}
</TRANSLATION_A>

<TRANSLATION_B>
{shuffled_translation_b}
</TRANSLATION_B>

When choosing between Translation A and Translation B, pay attention to which translation keeps the meaning of the <SOURCE_TEXT> considering the following:\
(i) accuracy (no deviations from the original meaning; all key details and nuances are conveyed exactly as intended)\
(ii) fluency (fluent and natural; no grammatical errors or awkward expressions; flows as if originally written in the {target_lang})\
(iii) style (maintained the original style, tone, and voice of the <SOURCE_TEXT>; the translation mirrors the source text perfectly in its stylistic delivery)\
(iv) terminology (terminology use is consistent and reflects the source text domain; and equivalent idioms in {target_lang} are used)\
(iv) cultural adaptablility (Cultural references, idioms, and context are perfectly adapted to the {target_lang} and {country})\

Write which translation is prefered or if its a tie or none and explain why.

"""
                    
                    # Generate the model's response with strict parameters
                    try:
                        class LLM_evlauation_response(BaseModel):
                            chain_of_thought: str = Field(description="The chain of thought that led to the better translation.")
                            winning_model: Literal['Translation A', 'Translation B', 'Tie','None'] = Field(description="The translation that was better.")

                        chat_completion = openai.beta.chat.completions.parse(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an expert in evaluating translation quality."
                                },
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            # model="mixtral-8x7b-32768",
                            model = 'gpt-4o',
                            temperature=0.2,  # Set temperature to 0 for deterministic output
                            top_p = 1,     # Limit the response length
                            stop=None,          # No specific stop sequence
                            response_format = LLM_evlauation_response
                        )
                        # Extract and clean the response
                        evaluation_response = chat_completion.choices[0].message.parsed
                        logging.info(f"ID {idx}: {evaluation_response}")
                    except Exception as e:
                        logging.error(f"Error evaluating translations for ID {idx}: {e}")
                        # Optionally, implement retry logic here
                        continue

                
                    # Define expected responses and map to winners
                    expected_responses = {
                        'Translation A': model_a,
                        'Translation B': model_b,
                        'Tie': 'Tie',
                        'None': 'None'
                    }

                    winner = expected_responses.get(evaluation_response.winning_model, "Could not determine")

                    #Prepare the result dictionary
                    result = {
                        "id": idx,
                        'winner': winner,
                        "Decision": evaluation_response.winning_model,
                        "evaluation reasoning": evaluation_response.chain_of_thought,
                        "model_a": model_a,
                        "model_b": model_b
                    }
                    # Append the result to the JSON file
                    try:
                        append_result(output_filepath, result)
                        logging.info(f"Appended result for ID {idx} to '{output_filename}'.")
                    except Exception as e:
                        logging.error(f"Error saving result for ID {idx} to '{output_filename}': {e}")

                    # Optional: Add a short delay to respect API rate limits
                    time.sleep(0.1)  # Adjust as needed based on API guidelines

                logging.info(f"Completed evaluations for pair '{model1}' vs '{model2}'.")



if __name__ == '__main__':
    main()

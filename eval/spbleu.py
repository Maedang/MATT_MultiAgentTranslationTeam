import os
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

def calculate_sp_bleu_for_language_model(language_name, model_name, agent_file_path, groundtruth_file_path):
   #translation
    agent_translations = pd.read_csv(agent_file_path)
    ground_truth_translations = pd.read_csv(groundtruth_file_path) 

    bleu_scores = []
    total_score = 0  # For averaging BLEU scores

    for _, agent_row in agent_translations.iterrows():
        agent_sentence = agent_row['source_text']
        agent_translation = agent_row['translation']  

        
        gt_row = ground_truth_translations[ground_truth_translations['source_text'] == agent_sentence]
        if not gt_row.empty:
            reference = [gt_row.iloc[0]['translation'].split()] 
            candidate = agent_translation.split()

           
            bleu_score = sentence_bleu(reference, candidate)
            bleu_scores.append((agent_sentence, agent_translation, bleu_score))

            
            total_score += bleu_score
        else:
            print(f"No ground truth found for: {agent_sentence}")

    
    if bleu_scores:
        average_bleu_score = total_score / len(bleu_scores)
    else:
        average_bleu_score = 0.0

   
    print(f"Average sp-BLEU score for {language_name} ({model_name}): {average_bleu_score}")


if __name__ == "__main__":
    # Language codes
    language =  "Vietnamese"
    #     "Hindi": "hin"
    #     "Malayalam": "mal"
    

    model =   "Multiagents"
    # "Baseline"
    # "Google"

    # Paths to folders
    translation_folder = "../translations"
    groundtruth_folder = "../data/Ground_Truth"
    agent_file_path = os.path.join(translation_folder, f"{model}/{model}_flores_{language}.csv")
    groundtruth_file_path = os.path.join(groundtruth_folder, f"ground_truth.csv")
    calculate_sp_bleu_for_language_model(language, model, agent_file_path, groundtruth_file_path)
   

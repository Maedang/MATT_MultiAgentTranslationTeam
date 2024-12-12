import datetime
import json
import random
import logging
import gradio as gr

import json

import json

with (
    open("../translations/Multiagents/Multiagents_flores_vie.json", "r") as multiagents_flores_vn,
    open("../translations/Multiagents/Multiagents_chunks_vie.json", "r") as multiagents_chunks_vn,
    open("../translations/Baseline/Baseline_flores_vie.json", "r") as baseline_flores_vn,
    open("../translations/Baseline/Baseline_chunks_vie.json", "r") as baseline_chunks_vn,
    open("../translations/Google/Google_flores_vie.json", "r") as google_flores_vn,
    open("../translations/Google/Google_chunks_vie.json", "r") as google_chunks_vn,
    
    open("../translations/Multiagents/Multiagents_flores_hin.json", "r") as multiagents_flores_hin,
    open("../translations/Multiagents/Multiagents_chunks_hin.json", "r") as multiagents_chunks_hin,
    open("../translations/Baseline/Baseline_flores_hin.json", "r") as baseline_flores_hin,
    open("../translations/Baseline/Baseline_chunks_hin.json", "r") as baseline_chunks_hin,
    open("../translations/Google/Google_flores_hin.json", "r") as google_flores_hin,
    open("../translations/Google/Google_chunks_hin.json", "r") as google_chunks_hin,

    open("../translations/Multiagents/Multiagents_flores_mal.json", "r") as multiagents_flores_mal,
    open("../translations/Multiagents/Multiagents_chunks_mal.json", "r") as multiagents_chunks_mal,
    open("../translations/Baseline/Baseline_flores_mal.json", "r") as baseline_flores_mal,
    open("../translations/Baseline/Baseline_chunks_mal.json", "r") as baseline_chunks_mal,
    open("../translations/Google/Google_flores_mal.json", "r") as google_flores_mal,
    open("../translations/Google/Google_chunks_mal.json", "r") as google_chunks_mal, 

    # open("../translations/Multiagents/Multiagents_flores_spa.json", "r") as multiagents_flores_spa,
    # open("../translations/Multiagents/Multiagents_chunks_spa.json", "r") as multiagents_chunks_spa,
    # open("../translations/Baseline/Baseline_flores_spa.json", "r") as baseline_flores_spa,
    # open("../translations/Baseline/Baseline_chunks_spa.json", "r") as baseline_chunks_spa,
    # open("../translations/Google/Google_flores_spa.json", "r") as google_flores_spa,
    # open("../translations/Google/Google_chunks_spa.json", "r") as google_chunks_spa,

    # open("../translations/Multiagents/Multiagents_flores_zho_simpl.json", "r") as multiagents_flores_zho_simpl,
    # open("../translations/Multiagents/Multiagents_chunks_zho_simpl.json", "r") as multiagents_chunks_zho_simpl,
    # open("../translations/Baseline/Baseline_flores_zho_simpl.json", "r") as baseline_flores_zho_simpl,
    # open("../translations/Baseline/Baseline_chunks_zho_simpl.json", "r") as baseline_chunks_zho_simpl,
    # open("../translations/Google/Google_flores_zho_simpl.json", "r") as google_flores_zho_simpl,
    # open("../translations/Google/Google_chunks_zho_simpl.json", "r") as google_chunks_zho_simpl,

    
):
    # Vietnamese
    multiagents_en_vn_f = json.load(multiagents_flores_vn)
    multiagents_en_vn = json.load(multiagents_chunks_vn)
    baseline_en_vn_f = json.load(baseline_flores_vn)
    baseline_en_vn = json.load(baseline_chunks_vn)
    google_en_vn_f = json.load(google_flores_vn)
    google_en_vn = json.load(google_chunks_vn)

    # Hindi
    multiagents_en_hi_f = json.load(multiagents_flores_hin)
    multiagents_en_hi = json.load(multiagents_chunks_hin)
    baseline_en_hi_f = json.load(baseline_flores_hin)
    baseline_en_hi = json.load(baseline_chunks_hin)
    google_en_hi_f = json.load(google_flores_hin)
    google_en_hi = json.load(google_chunks_hin)

    # Malayalam
    multiagents_en_mal_f = json.load(multiagents_flores_mal)
    multiagents_en_mal = json.load(multiagents_chunks_mal)
    baseline_en_mal_f = json.load(baseline_flores_mal)
    baseline_en_mal = json.load(baseline_chunks_mal)
    google_en_mal_f = json.load(google_flores_mal)
    google_en_mal = json.load(google_chunks_mal)

    # # # Spanish
    # multiagents_en_spa_f = json.load(multiagents_flores_spa)
    # multiagents_en_spa = json.load(multiagents_chunks_spa)
    # baseline_en_spa_f = json.load(baseline_flores_spa)
    # baseline_en_spa = json.load(baseline_chunks_spa)
    # google_en_spa_f = json.load(google_flores_spa)
    # google_en_spa = json.load(google_chunks_spa)
 
    # # Mandarin
    # multiagents_en_zho_simpl_f = json.load(multiagents_flores_zho_simpl)
    # multiagents_en_zho_simpl = json.load(multiagents_chunks_zho_simpl)
    # baseline_en_zho_simpl_f = json.load(baseline_flores_zho_simpl)
    # baseline_en_zho_simpl = json.load(baseline_chunks_zho_simpl)
    # google_en_zho_simpl_f = json.load(google_flores_zho_simpl)
    # google_en_zho_simpl = json.load(google_chunks_zho_simpl)


#Vietnamese
multiagents_en_vn += multiagents_en_vn_f
baseline_en_vn += baseline_en_vn_f
google_en_vn += google_en_vn_f

#Hindi
multiagents_en_hi += multiagents_en_hi_f
baseline_en_hi += baseline_en_hi_f
google_en_hi += google_en_hi_f

#Malayalam
multiagents_en_mal += multiagents_en_mal_f
baseline_en_mal += baseline_en_mal_f
google_en_mal += google_en_mal_f

# # Mandarin
# multiagents_en_zho_simpl += multiagents_en_zho_simpl_f
# baseline_en_zho_simpl += baseline_en_zho_simpl_f
# google_en_zho_simpl += google_en_zho_simpl_f

# #Spanish 
# multiagents_en_spa += multiagents_en_spa_f
# baseline_en_spa += baseline_en_spa_f
# google_en_spa += google_en_spa_f



# novoting_en_vn += novoting_en_vn_f

share_js = """
function () {
    const captureElement = document.querySelector('#share-region-annoy');
    // console.log(captureElement);
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'guardrails-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [];
}
"""

css = """
# language_box {width: 20%}
"""


def gen_random():
    """
    Generate a random index within the range of available translations in gpt4_en_spa.

    Returns:
        int: A random integer between 0 (inclusive) and len(gpt4_En_Spa) - 1.
    """
    return random.randint(0, len(multiagents_en_vn) - 1)

model_info = [
    {
        "model": "multiagents",
        "company": "Llama3.1 and OpenAI",
        "type": "Multiagents with voting",
        "link": "https://github.com/Maedang/Translation_MultiAgent",
    },
    {
        "model": "baseline",
        "company": "Llama3.1",
        "type": "Translation agents by Andrew Ng",
        "link": "https://github.com/andrewyng/translation-agent",
    },
    {
        "model": "google_translate",
        "company": "Google Translate only",
        "type": "NMT",
        "link": "https://google.com",
    },
]

models = ["multiagents",  "baseline", "google_translate"]

def get_model_description_md(models):
    """
    Generate a Markdown description of the models in `models`.

    Args:
        models (list): A list of model names.

    Returns:
        str: A Markdown string containing descriptions of each model.
    """
    model_description_md = "_________________________________ <br>"
    visited = set()
    for name in models:
        if name in visited:
            continue
        else:
            # Find the matching model info
            minfo = [x for x in model_info if x["model"] == name]
            if minfo:
                visited.add(name)
                one_model_md = f"[{minfo[0]['model']}]({minfo[0]['link']}): {minfo[0]['type']}"
                new_line = "_________________________________ <br>"
                model_description_md += f" {one_model_md} <br> {new_line}"
    return model_description_md

# Call the function
model_description_md = get_model_description_md(models)
print(model_description_md)



multiagent_dict = {
    "Vietnamese": multiagents_en_vn,
    "Hindi": multiagents_en_hi,
    "Malayalam": multiagents_en_mal,
    # "Simplified Mandarin": multiagents_en_zho_simpl,
    # "Spanish": multiagents_en_spa,
}
google_language_dict = {
    "Vietnamese": google_en_vn,
    "Hindi": google_en_hi,
    "Malayalam": google_en_mal,
    # "Simplified Mandarin": google_en_zho_simpl,
    # "Spanish": google_en_spa,
}
baseline_dict = {
    "Vietnamese": baseline_en_vn,
    "Hindi": baseline_en_hi,
    "Malayalam": baseline_en_mal,
#     "Simplified Mandarin": baseline_en_zho_simpl,
#     "Spanish": baseline_en_spa,
}

def change_language(txtbox1, txtbox2, src_txtbox, new_lang):
    new_idx = gen_random()
    print(new_idx)
    txtbox1 = multiagent_dict[new_lang][new_idx]["translation"]
    txtbox2 = baseline_dict[new_lang][new_idx]["translation"]
    src_txtbox = multiagent_dict[new_lang][new_idx]["source_text"]
    new_idx_random = gr.State(value=new_idx, render=False)
    print(new_idx_random.value)
    return txtbox1, txtbox2, src_txtbox


def write_answer(component, model, source_text, translation_a, translation_b):
    print(model)
    match component:
        case "👈  A is better":
            output = "A"
            model_name = model
            gr.Info(f"{model_name} won!")
        case "👉  B is better":
            output = "B"
            model_name = [x for x in models if x != model][0]
            gr.Info(f"{model_name} won!")
        case "🤝  Tie":
            output = "tie"
            model_name = "tie"
            gr.Info("'Tis a tie")
        case "👎  Equally bad":
            output = "both-bad"
            model_name = "both-bad"
            gr.Info("Both were bad!")
        case _:
            output = None
            model_name = None
    new_dict = {
        "time": datetime.datetime.now(),
        "output": output,
        "win_model": model_name,
    }
    print(new_dict)
    log_entry = {
        "time": datetime.datetime.now().isoformat(),
        "user_choice": output,
        "winning_model": model_name,
        "model_a": model,
        "model_b": [x for x in models if x != model][0],
        "source_text": source_text,
        "translation_a": translation_a,
        "translation_b": translation_b,
    }
    print(log_entry)

    # Write the log entry to a file - Vietnamese characters
    log_file_path = "translation_arena_log.jsonl"  # Adjust the path as needed
    with open(log_file_path, "a",  encoding='utf-8') as log_file:
        log_file.write(json.dumps(log_entry, ensure_ascii=False, indent=2) + "\n")


def regen(language, model):
    new_idx = gen_random()
    print(new_idx)
    model_a = random.choice(models)
    txtbox1_model = gr.State(value=model_a, render=True)
    
    match txtbox1_model.value:
        case "multiagents":
            init_value_a = multiagent_dict[language][new_idx]["translation"]
            init_value_b = baseline_dict[language][new_idx]["translation"]
        case "baseline":
            init_value_a = baseline_dict[language][new_idx]["translation"]
            init_value_b = multiagent_dict[language][new_idx]["translation"]
        case "google_translate":
            init_value_a = google_language_dict[language][new_idx]["translation"]
            init_value_b = multiagent_dict[language][new_idx]["translation"]
        
        case _:
            init_value_a = None
            init_value_b = None

    txtbox1 = init_value_a
    txtbox2 = init_value_b
    src_txtbox = multiagent_dict[language][new_idx]["source_text"]
    return txtbox1, txtbox2, src_txtbox, txtbox1_model.value

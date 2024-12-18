import random

import gradio as gr
from dotenv import load_dotenv
from utils import change_language
from utils import css
from utils import gen_random
from utils import get_model_description_md
from utils import multiagent_dict
from utils import baseline_dict
# from utils import novoting_dict
from utils import google_language_dict
from utils import models
from utils import regen
from utils import write_answer


load_dotenv()


with gr.Blocks(
    title="Translation Arena Battle",
    theme=gr.themes.Soft(secondary_hue=gr.themes.colors.sky),
    css=css,
) as demo:
    num_sides = 2
    states = [gr.State() for _ in range(num_sides)]
    chatbots = [None] * num_sides
    intro_num = gen_random()
    idx_random = gr.State(value=intro_num, render=True)
    model_a = random.choice(models)
    txtbox1_model = gr.State(value=model_a, render=True)
    print(idx_random.value)
    gr.Markdown(
        "# Translation Model Arena\n\nCompare and evaluate the translations of multiple models."
    )
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion(
                "Language", open=True, elem_id="language_box"
            ) as lang_row:
                lang = gr.Dropdown(
                    label="Choose Preferred Language",
                    choices=["Vietnamese", "Hindi", "Malayalam", "Spanish", "Simplified Mandarin"],
                    value="Vietnamese",
                    interactive=True,
                )
    with gr.Tabs() as tabs:
        with gr.Tab("Text Arena", id=0):
            with gr.Tab("⚔️  Arena (battle)", id=0):
                with gr.Group(elem_id="share-region-annoy"):
                    with gr.Column(scale=4):
                        chosen_lang = lang.value
                        print(chosen_lang)
                        source_txtbox = gr.Textbox(
                            label="Source Text",
                            value= multiagent_dict[chosen_lang][idx_random.value]["source_text"],
                            show_copy_button=True,
                        )

                with gr.Group(elem_id="share-region-annoy"):
                    with gr.Accordion(
                        "🔍 Expand to see the models", open=False
                    ):
                        all_models = [
                            {"name": "multiagents"},
                            {"name": "baseline"},
                            {"name": "google_translate"}, 
                           
                        ]
                        models = ["multiagents",  "baseline", "google_translate"]
                        model_description_md = get_model_description_md(models)
                        gr.Markdown(
                            model_description_md,
                            elem_id="model_description_markdown",
                        )
                    with gr.Row():
                        print(txtbox1_model.value)
                        match txtbox1_model.value:
                                case "multiagents":
                                    init_value_a = multiagent_dict[chosen_lang][idx_random.value]["translation"]
                                    init_value_b = baseline_dict[chosen_lang][idx_random.value]["translation"]
                                case "baseline":
                                    init_value_a = baseline_dict[chosen_lang][idx_random.value]["translation"]
                                    init_value_b = multiagent_dict[chosen_lang][idx_random.value]["translation"]
                                case "google_translate":
                                    init_value_a = google_language_dict[chosen_lang][idx_random.value]["translation"]
                                    init_value_b = multiagent_dict[chosen_lang][idx_random.value]["translation"]
                                
                                case _:
                                    pass
                        
                        with gr.Column():
                            label = "Model A"
                            txtbox1 = gr.Textbox(
                                label=label,
                                elem_id="chatbot",
                                value=init_value_a,
                                show_copy_button=True,
                            )

                        with gr.Column():
                            label = "Model B"
                            txtbox2 = gr.Textbox(
                                label=label,
                                elem_id="chatbot",
                                value=init_value_b,
                                show_copy_button=True,
                            )
                            lang.change( # type: ignore
                                fn=change_language,
                                inputs=[txtbox1, txtbox2, source_txtbox, lang],
                                outputs=[
                                    txtbox1,
                                    txtbox2,
                                    source_txtbox,
                                ],
                            )
                with gr.Row() as button_row:
                    a_better = gr.Button(
                        value="👈  A is better",
                        interactive=True,
                    )
                    a_better.click(
                        fn=write_answer, inputs=[a_better, txtbox1_model, source_txtbox, txtbox1, txtbox2]
                    )

                    b_better = gr.Button(
                        value="👉  B is better",
                        interactive=True,
                    )
                    b_better.click(
                        fn=write_answer, inputs=[b_better, txtbox1_model, source_txtbox, txtbox1, txtbox2]
                    )

                    tie_btn = gr.Button(
                        value="🤝  Tie", visible=True, interactive=True
                    )

                    tie_btn.click(
                        fn=write_answer, inputs=[tie_btn, txtbox1_model, source_txtbox, txtbox1, txtbox2]
                    )

                    bothbad_btn = gr.Button(
                        value="👎  Both are bad",
                        visible=True,
                        interactive=True,
                    )

                    bothbad_btn.click(
                        fn=write_answer, inputs=[bothbad_btn, txtbox1_model, source_txtbox, txtbox1, txtbox2]
                    )

            with gr.Row():
                regenerate_btn = gr.Button(
                    value="Regenerate", visible=True, interactive=True
                )

                regenerate_btn.click(
                    fn=regen,
                    inputs=[lang, txtbox1_model],
                    outputs=[txtbox1, txtbox2, source_txtbox, txtbox1_model],
                )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=10)
    demo.launch(share=True)

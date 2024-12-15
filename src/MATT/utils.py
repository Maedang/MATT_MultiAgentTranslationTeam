from typing import List
from typing import Union
import tiktoken
# from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
import json
import openai
from dotenv import load_dotenv
from deep_translator import GoogleTranslator 
from deep_translator.constants import GOOGLE_LANGUAGES_TO_CODES
import os 

load_dotenv()
client = Groq(api_key = os.getenv("GROQ_API_KEY"))
client_openai = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))


MAX_TOKENS_PER_CHUNK = (
    500  # if text is more than this many tokens, we'll break it up into
)
# discrete chunks to translate one chunk at a time

def get_language_code(language_name):
    """Convert a language name to its corresponding language code."""
    language_name = language_name.lower() 
    if language_name in GOOGLE_LANGUAGES_TO_CODES:
        return GOOGLE_LANGUAGES_TO_CODES[language_name]
    else:
        raise ValueError(f"Language '{language_name}' is not supported.")

def Google_Translator(source_lang, target_lang, text):
    try:
        # Convert language names to language codes
        source_lang_code = get_language_code(source_lang)
        target_lang_code = get_language_code(target_lang)

        # Use GoogleTranslator with the language codes
        GL_translator = GoogleTranslator(source=source_lang_code, target=target_lang_code)
        GL_translation = GL_translator.translate(text)
        return GL_translation
    
    except Exception as e:
        print(f"Error during translation: {e}")
        return None

# LLmama 3 
def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = "llama-3.1-70b-versatile",
    temperature: float = 0.3,
    json_mode: bool = False,
) -> Union[str, dict]:
    """
        Generate a completion using the OpenAI API.

    Args:
        prompt (str): The user's prompt or query.
        systekm_message (str, optional): The system message to set the context for the assistant.
            Defaults to "You are a helpful assistant.".
        model (str, optional): The name of the OpenAI model to use for generating the completion.
            Defaults to "mixtral-8x7b-32768".
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
            Defaults to 0.3.
        json_mode (bool, optional): Whether to return the response in JSON format.
            Defaults to False.

    Returns:
        Union[str, dict]: The generated completion.
            If json_mode is True, returns the complete API response as a dictionary.
            If json_mode is False, returns the generated text as a string.
    """

    if json_mode:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


# OpenAI
def get_completion_openAI(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-4o",
    temperature: float = 0.3,
) -> Union[str, dict]:
    """
        Generate a completion using the OpenAI API.

    Args:
        prompt (str): The user's prompt or query.
        systekm_message (str, optional): The system message to set the context for the assistant.
            Defaults to "You are a helpful assistant.".
        model (str, optional): The name of the OpenAI model to use for generating the completion.
            Defaults to "mixtral-8x7b-32768".
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
            Defaults to 0.3.
        json_mode (bool, optional): Whether to return the response in JSON format.
            Defaults to False.

    Returns:
        Union[str, dict]: The generated completion.
            If json_mode is True, returns the complete API response as a dictionary.
            If json_mode is False, returns the generated text as a string.
    """

    response = client_openai.chat.completions.create(
        model=model,
        temperature=temperature,
        top_p=1,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},

        ],
    )
    return response.choices[0].message.content    



def one_chunk_LLM_translator(
    source_lang: str, target_lang: str, source_text: str
) -> str:
    """
    Translate the entire text as one chunk using an LLM.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text (str): The text to be translated.
       

    Returns:
        str: The translated text.
    """

    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

    translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""

    prompt = translation_prompt.format(source_text=source_text)

    LLM_translation = get_completion(prompt, system_message=system_message)

    return LLM_translation


def evaluation_agent(
    source_lang: str,
    target_lang: str,
    source_text: str,
    LLM_translation: str,
    GL_translation: str,
) -> str:
    """
    Use an LLM to evaluate two translations and decide which one is better.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translations.
        source_text (str): The original text to be translated.
        LLM_translation (str): The first translation.
        GL_translation (str): Google Translation


    Returns:
        str: The better translation
    """

    system_message = (
        f"You are an expert linguist proficient in both {source_lang} and {target_lang}. "
        "Your task is to evaluate two translations of a given text and decide which one is better "
        "based on accuracy, fluency, and faithfulness to the source text."
    )

    evaluation_prompt = f"""
Given the following {source_lang} text:

"{source_text}"

Here are two {target_lang} translations:

A:
"{LLM_translation}"

B:
"{GL_translation}"

Please analyze both translations and determine which one is better overall. Consider the following criteria:

- **Accuracy**: How accurately does the translation convey the meaning of the source text?
- **Fluency**: Is the translation grammatically correct and natural-sounding in {target_lang}?
- **Faithfulness**: Does the translation preserve the nuances and tone of the original text?

Please choose the better translation by responding with either A or B. Response only includes one letter (A or B) of the most accurate, fluent, and faithful translation compared to the original text. no further information is needed.
"""
    
    evaluation = get_completion_openAI(evaluation_prompt, system_message=system_message)

    winning_translation = []

    if evaluation == "A": 
        winning_translation = LLM_translation
    elif evaluation == "B":
        winning_translation = GL_translation
    else:
        winning_translation = LLM_translation

    return winning_translation , evaluation


def one_chunk_proofreader(
    source_lang: str,
    target_lang: str,
    source_text: str,
    winning_translation: str,
    country: str = "",
) -> str:
    """
    Use an LLM to reflect on the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text (str): The original text in the source language.
        winning_translation (str): The winning translation between google translate and the LLM.
        country (str): Country specified for target language.

    Returns:
        str: The LLM's reflection on the translation, providing constructive criticism and suggestions for improvement.
    """

    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    if country != "":
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}. 

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{winning_translation}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text).\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    else:
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{winning_translation}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text,\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    prompt = reflection_prompt.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        winning_translation=winning_translation,
    )
    reflection = get_completion(prompt, system_message=system_message)
    return reflection


def one_chunk_editor(
    source_lang: str,
    target_lang: str,
    source_text: str,
    current_translation: str,
    reflection: str,
) -> str:
    """
    Use the reflection to improve the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The original text in the source language.
        current_translation (str): The current translation of the source text.
        reflection (str): Expert suggestions and constructive criticism for improving the translation.

    Returns:
        str: The improved translation based on the expert suggestions.
    """

    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."

    prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{current_translation}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""

    improved_translation = get_completion(prompt, system_message)

    return improved_translation



def loss_in_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation: str,
) -> dict:
    """
    Measures loss in translation by comparing the source text and the translation.
    
    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The original text in the source language.
        translation (str): The current improved translation of the source text.

    Returns:
        dict: Returns a dictionary with scores for Accuracy, Fluency, Style, Terminology, and a combined reason.
    """
    system_message = f"You are an expert linguist specializing in translation evaluation from {source_lang} to {target_lang}."
    
    prompt = f"""You are to act as a judge and compare the following {source_lang} text and its {target_lang} translation.

The source text and the translation are provided below:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation}
</TRANSLATION>

Please evaluate how well the translation preserves the meaning of the source text using the following rubrics for each category:

*(i) Accuracy*
Level         | Description
------------- | ----------- 
Excellent     | Completely accurate with no deviations from the original meaning; all key details and nuances are conveyed exactly as intended.
Satisfactory  | Largely accurate with only minor errors that do not affect overall comprehension.
Unacceptable  | Significant distortions of meaning; important information is incorrect or missing; mistranslations leading to misunderstanding of the content.

*(ii) Fluency*
Level         | Description
------------- | ----------- 
Excellent     | Completely fluent and natural; no grammatical errors or awkward expressions; flows as if originally written in the target language.
Satisfactory  | Mostly fluent with only a few unnatural expressions or isolated grammatical errors; reads smoothly overall.
Unacceptable  | Extremely awkward and unnatural; numerous grammatical mistakes; difficult to read and follow.

*(iii) Style*
Level         | Description
------------- | ----------- 
Excellent     | The original style, tone, and voice are fully maintained; the translation mirrors the source text perfectly in its stylistic delivery.
Satisfactory  | The style is well preserved with only slight deviations; the translation maintains the appropriate tone and reflects the original text’s voice effectively.
Unacceptable  | Significant loss of the original style; tone and voice are inappropriate or completely inconsistent with the source text.

*(iv) Terminology*
Level         | Description
------------- | ----------- 
Excellent     | Terminology is consistently accurate and precise throughout; all domain-specific terms are used correctly and appropriately.
Satisfactory  | Terminology is accurate and consistent, with only minor and isolated errors that don’t impact the overall meaning or coherence.
Unacceptable  | Incorrect or inconsistent use of key terminology; domain-specific terms are either mistranslated or completely missing.

*(v) Cultural Adaptation*
Level         | Description
------------- | ----------- 
Excellent     | Cultural references, idioms, and context are perfectly adapted to the target language.
Satisfactory  | Minor cultural nuances are missed but do not affect overall comprehension.
Unacceptable  | Cultural elements are entirely mishandled, causing significant loss of meaning.


**Your Task:**
- Assign a label of Excellent, Satisfactory, or Unacceptable for each category.
- Provide a **combined reason** for the assigned scores and what was lost in translation, citing specific examples from the texts.
- Present your evaluation in the following JSON format:

{{
    "Accuracy": "label",
    "Fluency": "label",
    "Style": "label",
    "Terminology": "label",
    "Cultural Adaptation": "label",
    "Reason": "Combined reason text with specific examples."
}}

**Note:**
- Replace "label" with one of the rankings: Excellent, Satisfactory, or Unacceptable.
- Replace 'Combined reason text with specific examples' with your brief combined reason for the assigned scores.
- Ensure the dictionary is properly formatted.
- Do not include any additional text outside the JSON response.
"""

    # Call to OpenAI API (assuming this function handles communication with OpenAI API)
    evaluation = get_completion_openAI(
        prompt,
        system_message=system_message,
    )

    if isinstance(evaluation, str):
        stripped_eval = evaluation.strip("```json").strip("```").strip()
        
        try:
            evaluation_dict = json.loads(stripped_eval)
            return evaluation_dict
        except json.JSONDecodeError as e:
            print("Error decoding JSON. Response was:")
            print(stripped_eval)
            print("JSONDecodeError:", e)
            return None
    else:
        print("Evaluation is not a string. Response was:")
        print(evaluation)
        return None



def calculate_loss_in_translation_percentage(evaluation_dict):
    """
    Calculate the current loss percentage based on the evaluation scores.

    Args:
        evaluation (dict): A dictionary containing evaluation scores.

    Returns:
        float: The calculated loss percentage.
    """
    label_to_score = {
        'Unacceptable': 1,
        'Satisfactory': 2,
        'Excellent': 3
    }


    if isinstance(evaluation_dict, dict):
        # Extract scores from evaluation
        scores = {
            'Accuracy': label_to_score[evaluation_dict['Accuracy']],
            'Fluency': label_to_score[evaluation_dict['Fluency']],
            'Terminology': label_to_score[evaluation_dict['Terminology']],
            'Style': label_to_score[evaluation_dict['Style']],
            'Cultural Adaptation': label_to_score[evaluation_dict['Cultural Adaptation']]
        }

        # Use the provided weights (sum to 1.0)
        weights = {
            'Accuracy': 0.4,
            'Fluency': 0.2,
            'Terminology': 0.2,
            'Style': 0.1,
            'Cultural Adaptation': 0.1
        }

        # Calculate the weighted total score
        weighted_score = sum(scores[metric] * weights[metric] for metric in scores)

        # The maximum possible weighted score is 5.0
        max_weighted_score = 3.0

        # Calculate the loss percentage
        current_loss_percentage = round(1 - (weighted_score / max_weighted_score), 2)
    else:
        print("Failed to get a valid evaluation.")
        # Assign maximum loss if evaluation fails
        current_loss_percentage = 1.0

    return current_loss_percentage



def one_chunk_editor_in_chief(source_lang, target_lang, source_text, improved_translation):
    """
    Improve the translation iteratively based on loss percentage.

    Args:
        source_lang (str): The source language.
        target_lang (str): The target language.
        source_text (str): The source text to translate.
        improved_translation (str): The initial translation after 1 reflection to improve on based on loss in translation.

    Returns:
        str: The final improved translation.
    """
    current_translation = improved_translation  # Start with the initial improved translation
    iteration = 0
    max_iterations = 3
    loss_threshold = 0.1

    # Initial evaluation
    evaluation = loss_in_translation(source_lang, target_lang, source_text, current_translation)
    current_loss_percentage = calculate_loss_in_translation_percentage(evaluation)

    print("Initial Loss Percentage:", current_loss_percentage)

    while iteration < max_iterations and current_loss_percentage > loss_threshold:
        iteration += 1
        print(f"Iteration {iteration}: Current Loss is {current_loss_percentage}")

        # Evaluate the current translation
        evaluation = loss_in_translation(source_lang, target_lang, source_text, current_translation)
        if not evaluation or not isinstance(evaluation, dict):
            print("Failed to get a valid evaluation. Stopping loop.")
            break

        # Calculate the current loss percentage
        current_loss_percentage = calculate_loss_in_translation_percentage(evaluation)

        # Check if the loss is zero (i.e., perfect translation)
        if current_loss_percentage == 0:
            print("Perfect translation achieved. Stopping loop.")
            break

        # Use the evaluation reason as feedback
        reflection = evaluation['Reason']

        # Improve the translation
        improved_translation = one_chunk_editor(
            source_lang, target_lang, source_text, current_translation, reflection
        )

        # Update current translation
        current_translation = improved_translation

        # Check if loss is acceptable
        if current_loss_percentage <= loss_threshold:
            print(f"Iteration {iteration}: Loss is acceptable (<= {loss_threshold}). Stopping loop.")
            break

    return current_translation

def one_chunk_translation_team(
    source_lang: str, target_lang: str, source_text: str, country: str = ""
) -> str:
    """
    Translate a single chunk of text from the source language to the target language.

    This function performs a two-step translation process:
    1. Get an initial LLM translation of the source text.
    2. Get a Google Translate Translation 
    3. Vote on if the GT translation or the LLM translation is better 
    4. Take the winning translation and reflect on it to generate an improved translation
    5. Take the improved translation and calculate loss in translation and improve until loss < 20%

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The text to be translated.
        country (str): Country specified for target language.
    Returns:
        str: The improved translation of the source text.
    """
    LLM_translation = one_chunk_LLM_translator(
        source_lang, target_lang, source_text
    )

    GL_translation = Google_Translator(source_lang, target_lang, source_text)

    winning_translation = evaluation_agent(source_lang,
    target_lang,
    source_text,
    LLM_translation,
    GL_translation)

    
    reflection = one_chunk_proofreader(
        source_lang, target_lang, source_text, winning_translation, country
    )
    initial_improved_translation = one_chunk_editor(
        source_lang, target_lang, source_text, winning_translation, reflection
    )
    final_translation = one_chunk_editor_in_chief(source_lang, target_lang, source_text, initial_improved_translation)

    return final_translation


def num_tokens_in_string(
    input_str: str, encoding_name: str = "cl100k_base"
) -> int:
    """
    Calculate the number of tokens in a given string using a specified encoding.

    Args:
        str (str): The input string to be tokenized.
        encoding_name (str, optional): The name of the encoding to use. Defaults to "cl100k_base",
            which is the most commonly used encoder (used by GPT-4).

    Returns:
        int: The number of tokens in the input string.

    Example:
        >>> text = "Hello, how are you?"
        >>> num_tokens = num_tokens_in_string(text)
        >>> print(num_tokens)
        5
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens


def multichunk_LLM_translator(
    source_lang: str, target_lang: str, source_text_chunks: List[str]
) -> List[str]:
    """
    Translate a text in multiple chunks from the source language to the target language.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): A list of text chunks to be translated.

    Returns:
        List[str]: A list of translated text chunks.
    """

    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

    translation_prompt = """Your task is provide a professional translation from {source_lang} to {target_lang} of PART of a text.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>. Translate only the part within the source text
delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS>. You can use the rest of the source text as context, but do not translate any
of the other text. Do not output anything other than the translation of the indicated part of the text.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, you should translate only this part of the text, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

Output only the translation of the portion you are asked to translate, and nothing else.
"""

    LLM_translation_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = translation_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
        )

        LLM_translation = get_completion(prompt, system_message=system_message)
        LLM_translation_chunks.append(LLM_translation)

    return LLM_translation_chunks


def multichunk_proofreader(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    winning_translation_chunks :List[str],
    country: str = "",
) -> List[str]:
    """
    Provides constructive criticism and suggestions for improving a partial translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text_chunks (List[str]): The source text divided into chunks.
        translation_1_chunks (List[str]): The translated chunks corresponding to the source text chunks.
        country (str): Country specified for target language.

    Returns:
        List[str]: A list of reflections containing suggestions for improving each translated chunk.
    """

    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    if country != "":
        reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{winning_translation_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    else:
        reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{winning_translation_chunk}

</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text), and considering the {GL_reference} as reference.\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    reflection_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )
        if country != "":
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                winning_translation_chunk = winning_translation_chunks[i],
                country=country,
            )
        else:
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                winning_translation_chunk=winning_translation_chunks[i],
            )

        reflection = get_completion(prompt, system_message=system_message)
        reflection_chunks.append(reflection)

    return reflection_chunks



def multichunk_editor(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    winning_translation_chunks: List[str],
    reflection_chunks: List[str],
) -> List[str]:
    """
    Improves the translation of a text from source language to target language by considering expert suggestions.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): The source text divided into chunks.
        translation_1_chunks (List[str]): The initial translation of each chunk.
        reflection_chunks (List[str]): Expert suggestions for improving each translated chunk.

    Returns:
        List[str]: The improved translation of each chunk.
    """

    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."

    improvement_prompt = """Your task is to carefully read, then improve, a translation from {source_lang} to {target_lang}, taking into
account a set of expert suggestions and constructive criticisms. Below, the source text, initial translation, and expert suggestions are provided.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context, but need to provide a translation only of the part indicated by <TRANSLATE_THIS> and </TRANSLATE_THIS>.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{winning_translation_chunk}
</TRANSLATION>

The expert translations of the indicated part, delimited below by <EXPERT_SUGGESTIONS> and </EXPERT_SUGGESTIONS>, is as follows:
<EXPERT_SUGGESTIONS>
{reflection_chunk}
</EXPERT_SUGGESTIONS>

Taking into account the expert suggestions rewrite the translation to improve it, paying attention
to whether there are ways to improve the translation's

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation of the indicated part and nothing else."""

    improved_translation_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = improvement_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
            winning_translation_chunk=winning_translation_chunks[i],
            reflection_chunk=reflection_chunks[i],
        )

        improved_translation = get_completion(prompt, system_message=system_message)
        improved_translation_chunks.append(improved_translation)

    return improved_translation_chunks

def multichunk_editor_in_chief(source_lang, target_lang, source_text_chunks, improved_translation_chunks):
    """
    Improve the translation iteratively based on loss percentage.

    Args:
        source_lang (str): The source language.
        target_lang (str): The target language.
        source_text_chunks (List[str]): The source text to translate.
        improved_translation_chunks (List[str]): The initial translation after 1 reflection to improve on based on loss in translation.

    Returns:
        str: The final improved translation.
    """
    current_translation_chunks = improved_translation_chunks  # Start with the initial improved translation
    iteration = 0
    max_iterations = 3
    loss_threshold = 0.1

    # Initial evaluation
    evaluation = loss_in_translation(source_lang, target_lang, source_text_chunks, current_translation_chunks)
    current_loss_percentage = calculate_loss_in_translation_percentage(evaluation)

    print("Initial Loss Percentage:", current_loss_percentage)

    while iteration < max_iterations and current_loss_percentage > loss_threshold:
        iteration += 1
        print(f"Iteration {iteration}: Current Loss is {current_loss_percentage}")

        # Evaluate the current translation
        evaluation = loss_in_translation(source_lang, target_lang, source_text_chunks, current_translation_chunks)
        if not evaluation or not isinstance(evaluation, dict):
            print("Failed to get a valid evaluation. Stopping loop.")
            break

        # Calculate the current loss percentage
        current_loss_percentage = calculate_loss_in_translation_percentage(evaluation)

        # Check if the loss is zero (i.e., perfect translation)
        if current_loss_percentage == 0:
            print("Perfect translation achieved. Stopping loop.")
            break

        # Use the evaluation reason as feedback
        reflection = evaluation['Reason']

        # Improve the translation
        improved_translation_chunks = multichunk_editor(
            source_lang, target_lang, source_text_chunks, current_translation_chunks, reflection
        )

        # Update current translation
        current_translation_chunks = improved_translation_chunks

        # Check if loss is acceptable
        if current_loss_percentage <= loss_threshold:
            print(f"Iteration {iteration}: Loss is acceptable (<= {loss_threshold}). Stopping loop.")
            break

    return current_translation_chunks

def multichunk_multiagent_translation(
    source_lang : str, target_lang : str , source_text_chunks: str,  country: str = ""
):
    """
    Translate a single chunk of text from the source language to the target language.

    This function performs a two-step translation process:
    1. Get an initial LLM translation of the source text.
    2. Get a Google Translate Translation 
    3. Vote on if the GT translation or the LLM translation is better 
    4. Take the winning translation and reflect on it to generate an improved translation
    5. Take the improved translation and calculate loss in translation and improve until loss < 20%

    Args:
        source_lang (str): The source language of the text chunks.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): The list of source text chunks to be translated.
        translation_1_chunks (List[str]): The list of initial translations for each source text chunk.
        winning_translation_chunks (List[str]): The list of the winning translation for each source text chinks. 
        reflection_chunks (List[str]): The list of reflections on the initial translations.
        country (str): Country specified for target language
    Returns:
        List[str]: The list of improved translations for each source text chunk.
    """
    
    LLM_translation_chunks = multichunk_LLM_translator(source_lang, target_lang, source_text_chunks)

    GL_translation_chunks = multichunk_Google_Translator(source_lang, target_lang, source_text_chunks)

    winning_translation_chunks = evaluation_agent(source_lang,
    target_lang,
    source_text_chunks,
    LLM_translation_chunks,
    GL_translation_chunks)

    reflection_chunks = multichunk_proofreader(
        source_lang,
        target_lang,
        source_text_chunks,
        winning_translation_chunks,
        country,
    )

    inital_improved_translation_chunks = multichunk_editor(
        source_lang,
        target_lang,
        source_text_chunks,
        winning_translation_chunks,
        reflection_chunks,
    )

    final_translation_chunks = multichunk_editor_in_chief(source_lang, target_lang, source_text_chunks, inital_improved_translation_chunks)

    return final_translation_chunks


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    """
    Calculate the chunk size based on the token count and token limit.

    Args:
        token_count (int): The total number of tokens.
        token_limit (int): The maximum number of tokens allowed per chunk.

    Returns:
        int: The calculated chunk size.

    Description:
        This function calculates the chunk size based on the given token count and token limit.
        If the token count is less than or equal to the token limit, the function returns the token count as the chunk size.
        Otherwise, it calculates the number of chunks needed to accommodate all the tokens within the token limit.
        The chunk size is determined by dividing the token limit by the number of chunks.
        If there are remaining tokens after dividing the token count by the token limit,
        the chunk size is adjusted by adding the remaining tokens divided by the number of chunks.

    Example:
        >>> calculate_chunk_size(1000, 500)
        500
        >>> calculate_chunk_size(1530, 500)
        389
        >>> calculate_chunk_size(2242, 500)
        496
    """

    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size


def translate(
    source_lang,
    target_lang,
    source_text,
    country,
    max_tokens=MAX_TOKENS_PER_CHUNK,
):
    """Translate the source_text from source_lang to target_lang."""

    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as single chunk")

        final_translation = one_chunk_translation_team(
            source_lang, target_lang, source_text, country
        )

        return final_translation

    else:
        ic("Translating text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        final_translation_chunks = multichunk_multiagent_translation(
            source_lang, target_lang, source_text_chunks, country
        )

        return "".join(final_translation_chunks)




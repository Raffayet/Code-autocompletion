import re
from enum import Enum
import sys

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu.metrics import CHRF
from rapidfuzz.distance import Levenshtein
from rouge_score import rouge_scorer

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigcode/tiny_starcoder_py"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class CompletionType(Enum):
    SINGLE_BLOCK_CONTEXT = 1
    MULTI_BLOCK_CONTEXT = 2


# Main code autocompletion logic
def code_autocomplete(prefix, suffix, completion_type: CompletionType):

    input_context = ""

    # This type analyzes only single block of code, and has its own model calling
    # This happens when user stops typing in the middle of the statement
    if completion_type == CompletionType.SINGLE_BLOCK_CONTEXT:
        input_context = prefix if len(prefix) > 0 else suffix
    # On the other hand the multi block context processing happens when users initiates new row or block of code
    elif completion_type == CompletionType.MULTI_BLOCK_CONTEXT:
        # input_context = prefix + suffix
        # First checking if there is function to load context from - more precise results
        if "def" in prefix:
            input_context = "def " + prefix.split("def")[-1]
        # If not, then loading the whole context
        else:
            input_context = prefix + suffix

    input_encoded = tokenizer(input_context, return_tensors="pt")

    if completion_type == CompletionType.SINGLE_BLOCK_CONTEXT:

        output = model.generate(
            input_encoded.input_ids,
            max_new_tokens=10,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1,
            top_p=0.95,
            top_k=10,
            num_beams=20,
            no_repeat_ngram_size=3,
            length_penalty=0.1,
            do_sample=True
        )
    else:
        output = model.generate(
            input_encoded.input_ids,
            max_new_tokens=20,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1,
            top_p=0.95,
            top_k=10,
            num_beams=20,
            no_repeat_ngram_size=3,
            length_penalty=1.5,
            do_sample=True
        )

    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

    # The middle code that was missing
    middle = generated_code.replace(prefix, "").replace(suffix, "").strip()


    # Construct the final code (prefix + generated middle + suffix)
    # middle is shown in red color
    red_middle = f"\033[91m{middle}\033[0m"
    completed_code = prefix + red_middle + suffix

    print("Final Code:\n", completed_code)
    return completed_code, middle


def split_code_at_cursor(code):
    # Cursor indicates stop place for auto completion
    code_parts = code.split("|")
    prefix = code_parts[0]
    suffix = code_parts[1]

    # ANSI escape code for red color
    red_caret = "\033[91m|\033[0m"

    print("Code before completion: \n")
    print(prefix + red_caret + suffix)

    return prefix, suffix


def calculate_bleu_score(input_code, original_code):
    # Splitting into words
    candidate = re.findall(r'\b\w+\b', input_code)
    reference = re.findall(r'\b\w+\b', original_code)

    # Max n-grams set to 1, as in the existing examples gives higher score
    max_n = 1

    weights = tuple([1.0 / max_n] * max_n)

    smoothing_function = SmoothingFunction().method7

    bleu_score = sentence_bleu(
        [reference],
        candidate,
        weights=weights,
        smoothing_function=smoothing_function
    )
    return bleu_score


def evaluate_metrics(input_code, original_code):
    # Calculate exact match as a boolean
    exact_match = input_code == original_code

    bleu_score = calculate_bleu_score(input_code, original_code)

    chrf_metric = CHRF()
    chrf_score = chrf_metric.sentence_score(input_code, [original_code]).score

    # Calculate Levenshtein Distance (Normalized)
    levenshtein_normalized = Levenshtein.distance(input_code, original_code) / max(len(input_code), len(original_code))

    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_instance.score(original_code, input_code)
    rouge_l_score = rouge_scores['rougeL'].fmeasure

    return {
        "exact_match": exact_match,
        "bleu": bleu_score,
        "chrf": chrf_score,
        "levenshtein_normalized_distance": levenshtein_normalized,
        "rouge_l_score": rouge_l_score,
    }


def read_file(path):
    with open(path) as file:
        return file.read()



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py <sample_directory>")
        print("Example: python main.py sample_codes/basic_multi_block/sample1")
        sys.exit(1)

    sample_folder = sys.argv[1]

    input_code = read_file(sample_folder + "/input.txt")
    original_code = read_file(sample_folder + "/original.txt")

    # Split code into prefix, middle (which will be predicted), and suffix
    prefix, suffix = split_code_at_cursor(input_code)
    missing_code = original_code.replace(prefix.strip(), "").replace(suffix.strip(), "").strip()

    # If the cursor reaches now row - it is probably going to type a new block of code
    # Therefore it needs the contect of the whole input code
    if len(prefix) > 0:
        if prefix[-1] == "\n" or prefix[-3:] == "   ":
            completed_code, generated_code = code_autocomplete(prefix, suffix, CompletionType.MULTI_BLOCK_CONTEXT)
            print("\nMulti context processing")
        else:
            # Otherwise, the autocompletion should just finish the line, so the new code should be short
            completed_code, generated_code = code_autocomplete(prefix, suffix, CompletionType.SINGLE_BLOCK_CONTEXT)
            print("\nSingle context processing")
    # This means that cursor is at the beginning of the file
    else:
        if suffix[-1] == "\n" or suffix[-3:] == "   ":
            completed_code, generated_code = code_autocomplete(prefix, suffix, CompletionType.MULTI_BLOCK_CONTEXT)
            print("\nMulti block processing")
        else:
        # Otherwise, the autocompletion should just finish the line, so the new code should be short
            completed_code, generated_code = code_autocomplete(prefix, suffix, CompletionType.SINGLE_BLOCK_CONTEXT)
            print("\nSingle block processing")

    if missing_code != "":
        # Evaluating metrics of the model only if there is difference in existing code
        metrics = evaluate_metrics(generated_code, missing_code)
        print(metrics)
    else:
        # When adding new code, we can't measure the similarity of the strings inside the code
        print("Only adding new content - so no metrics needed to show")
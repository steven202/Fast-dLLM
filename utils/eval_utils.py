import re
import string


def _exact_match_hf_evaluate(
    prediction: str,
    reference: str,
    regexes_to_ignore=None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
) -> bool:
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            prediction = re.sub(s, "", prediction)
            reference = re.sub(s, "", reference)

    if ignore_case:
        prediction = prediction.lower()
        reference = reference.lower()

    if ignore_punctuation:
        repl_table = str.maketrans("", "", string.punctuation)
        prediction = prediction.translate(repl_table)
        reference = reference.translate(repl_table)

    if ignore_numbers:
        repl_table = str.maketrans("", "", string.digits)
        prediction = prediction.translate(repl_table)
        reference = reference.translate(repl_table)

    return prediction == reference


def _regex_filter(text: str, regex_pattern: str, group_select: int = 0, fallback: str = "[invalid]") -> str:
    if not isinstance(text, str):
        text = ""
    regex = re.compile(regex_pattern)
    match = regex.findall(text)
    if match:
        match = match[group_select]
        if isinstance(match, tuple):
            match = [m for m in match if m]
            match = match[0] if match else fallback
        match = match.strip()
    else:
        match = fallback
    return match


def gsm8k_matches(pred: str, gold: str) -> dict:
    """
    Replicates lm_eval gsm8k filter + exact_match logic.
    Returns dict with strict/flexible and raw exact_match booleans.
    """
    regexes_to_ignore = [",", r"\$", r"(?s).*#### ", r"\.$"]

    strict_pred = _regex_filter(pred, r"#### (\-?[0-9\.\,]+)", group_select=0)
    flexible_pred = _regex_filter(
        pred,
        r"(-?[$0-9.,]{2,})|(-?[0-9]+)",
        group_select=-1,
    )

    strict_match = _exact_match_hf_evaluate(
        strict_pred,
        gold,
        regexes_to_ignore=regexes_to_ignore,
        ignore_case=True,
        ignore_punctuation=False,
        ignore_numbers=False,
    )
    flexible_match = _exact_match_hf_evaluate(
        flexible_pred,
        gold,
        regexes_to_ignore=regexes_to_ignore,
        ignore_case=True,
        ignore_punctuation=False,
        ignore_numbers=False,
    )
    raw_match = _exact_match_hf_evaluate(
        pred,
        gold,
        regexes_to_ignore=regexes_to_ignore,
        ignore_case=True,
        ignore_punctuation=False,
        ignore_numbers=False,
    )

    return {
        "strict_match": strict_match,
        "flexible_match": flexible_match,
        "raw_match": raw_match,
    }

def extract_answer_math(text):
    """Specific extractor for MATH/GSM8K"""
    # 1. Handle GSM8K style (####)
    if "####" in text:
        return text.split("####")[-1].strip().replace(',', '')
    
    # 2. Handle MATH style (\boxed{})
    # Find the last occurrence of \boxed{...}
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', text)
    if boxed_matches:
        return boxed_matches[-1]
        
    # 3. Fallback: Find last number
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if matches:
        return matches[-1]
    return None


def _remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]
        return s

    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        return s
    return s[len(left) : -1]


def _last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a_int = int(a)
        b_int = int(b)
        if string != f"{a_int}/{b_int}":
            return string
        return f"\\frac{{{a_int}}}{{{b_int}}}"
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # remove leading variable assignment like "k ="
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = _fix_a_slash_b(string)
    return string


def _is_equiv(str1: str, str2: str) -> bool:
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


def hendrycks_math_is_correct(pred: str, solution: str) -> bool:
    indices = [pos for pos, char in enumerate(pred) if char == "$"]
    if len(indices) <= 1:
        answer = pred
    else:
        answer = pred[indices[0] + 1 : indices[-1]]

    gold = _last_boxed_only_string(solution or "")
    if gold is None:
        return False
    gold = _remove_boxed(gold)
    return _is_equiv(answer, gold)

def normalize_code(code):
    """Strip whitespace, comments, and imports for comparison"""
    # Simple normalization: remove all whitespace to compare logic structure
    return re.sub(r'\s+', '', code).strip()

def is_correct_smart(pred, gold, dataset_type):
    """Polymorphic evaluator"""
    try:
        if dataset_type == "gsm8k":
            # Use lm_eval gsm8k flexible-extract exact_match
            return gsm8k_matches(pred, gold)["flexible_match"]

        if dataset_type == "math":
            return hendrycks_math_is_correct(pred, gold)
            
        elif dataset_type in ["humaneval", "mbpp"]:
            # For code, Exact Match is very harsh. 
            # We usually rely on NLL reward for code training.
            # But for logging accuracy, we try normalized string match.
            return normalize_code(pred) == normalize_code(gold)
            
    except:
        # If conversion to float fails (e.g. comparing "5" to "x=5"), return False
        return False
    return False

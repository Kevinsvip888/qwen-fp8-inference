import re
from decimal import Decimal

SYSTEM_PROMPT = "You are a helpful assistant."

USER_TEMPLATE = (
    "Solve the following math problem. Explain your reasoning briefly.\n"
    "End with a separate line in exactly this format:\n"
    "Final answer: <number>\n"
    "Use an integer or decimal number on that final line. "
    "Do not include units, fractions, expressions, or LaTeX there.\n\n"
    "Problem:\n{question}"
)

GENERATION = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "seed": 42,
    "n": 1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"

def canonical_number(text):
    text = text.strip().replace("\u2212", "-")
    if re.fullmatch(NUMBER, text) is None:
        raise ValueError(f"Invalid numeric answer: {text!r}")

    value = Decimal(text.replace(",", ""))
    if not value.is_finite():
        raise ValueError("Nonfinite answer")
    if value == 0:
        return "0"

    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result

def parse_final_answer(text):
    if not isinstance(text, str) or not text.strip():
        return None

    line = text.strip().splitlines()[-1].strip()
    line = (
        line.replace("**", "")
        .replace("`", "")
        .replace("\u2212", "-")
    )
    match = re.fullmatch(
        rf"Final answer:\s*({NUMBER})",
        line,
        flags=re.IGNORECASE,
    )
    return canonical_number(match.group(1)) if match else None

def make_request(question):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(question=question),
            },
        ],
        **GENERATION,
    }

def self_check():
    cases = [
        ("Work: 6 * 7 = 42.\nFinal answer: 42", "42"),
        ("**Final answer:** **1,200.00**", "1200"),
        ("Final answer: -0.50", "-0.5"),
        ("Final answer: `42`", "42"),
        ("Final answer: 100", "100"),
        ("Final answer: -0.00", "0"),
        ("Work gives 42.\nFinal answer: 43", "43"),
        ("Final answer: 4.\nExplanation follows.", None),
        ("Final answer: 1,20", None),
        ("Final answer: 2/3", None),
        ("The answer is 42", None),
        ("Final answer: NaN", None),
        ("Final answer: 42 dollars", None),
    ]
    for text, expected in cases:
        actual = parse_final_answer(text)
        assert actual == expected, (text, actual, expected)

    assert canonical_number(
        "123456789012345678901234567890.000"
    ) == "123456789012345678901234567890"

if __name__ == "__main__":
    self_check()
    print("PASS: numerical normalization and final-answer parser checks.")

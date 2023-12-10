import re

# 각 글자가 영어인지 한글인지 판별
def is_english_or_korean(text):
    if re.match(r"[ㄱ-ㅎㅏ-ㅣ가-힣]", text):
        return "korean"
    elif re.match(r"[a-zA-Z]", text):
        return "english"
    else:
        return "other"


def add_space_to_korean_word(text):
    """
    "삼성" -> "삼 성"
    """
    result = ""
    for char in text:
        if is_english_or_korean(char) == "korean":
            result += char + " "
        else:
            result += char
    return result.strip()

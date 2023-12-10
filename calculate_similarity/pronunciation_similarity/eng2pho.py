import phonetics
import eng_to_ipa as ipa
from hangul_romanize import Transliter
from hangul_romanize.rule import academic

from pronunciation_similarity.utils import add_space_to_korean_word


def convert_to_ipa(word):
    return ipa.convert(word)


def korean_to_english_pronunciation(korean_word):
    space_added_korean_word = add_space_to_korean_word(korean_word)
    transliter = Transliter(academic)
    romanized = transliter.translit(space_added_korean_word)
    return romanized


word = "sam sung"
korean_word = "삼 성"
english_pronunciation = repr(korean_to_english_pronunciation(korean_word))

ipa_representation = convert_to_ipa(word)
ipa_representation_kor = convert_to_ipa(english_pronunciation)

print(f"'{word}의 발음기호는 : {ipa_representation} 입니다.")

print(f"'{korean_word}' 의 영어발음은 : {english_pronunciation} 입니다.")
print(f"'{korean_word}' 의 발음기호는 : {ipa_representation_kor} 입니다.")

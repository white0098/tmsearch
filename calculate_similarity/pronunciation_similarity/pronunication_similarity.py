import phonetics

from calculate_similarity.text_similarity import TextFeatureExtractor
from calculate_similarity.pronunciation_similarity.eng2pho import (
    convert_to_ipa,
    korean_to_english_pronunciation,
)
from calculate_similarity.pronunciation_similarity.utils import add_space_to_korean_word, is_english_or_korean


class PronunciationFeatureExtractor:
    def __init__(self) -> None:
        self.text_sim_model = TextFeatureExtractor()

    def _preprocess(self, x1, x2):
        input_data = [x1, x2]
        for x in input_data:
            if is_english_or_korean(x) == "korean":
                input_data[input_data.index(x)] = convert_to_ipa(
                    korean_to_english_pronunciation(x, add_space=True)
                )

            else:
                input_data[input_data.index(x)] = convert_to_ipa(x)
        print("PRONUNCIATION: ", input_data)
        return input_data

    def calculate_similarity(self, x1, x2):
        preprocessed_data = self._preprocess(x1, x2)
        similarity = self.text_sim_model.calculate_similarity(*preprocessed_data)
        return similarity

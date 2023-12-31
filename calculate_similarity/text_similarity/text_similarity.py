import re

from calculate_similarity.pronunciation_similarity.eng2pho import convert_to_ipa, korean_to_english_pronunciation
from calculate_similarity.pronunciation_similarity.utils import is_english_or_korean

class TextFeatureExtractor:
    def _convert_to_ngram(self, text, n):
        # remove space
        text = re.sub(r"\s+", "", text)
        text = text.lower()
        text = [text[i : i + n] for i in range(len(text) - n + 1)]
        return text
    
    def _preprocess(self, x):
        if is_english_or_korean(x) == "korean":
            eng_x = korean_to_english_pronunciation(x, add_space=True)
            print(f'KO2ENG: {x} -> {eng_x}')
            return eng_x 
        else:
            return x
        
        
        

    def calculate_similarity(self, x1, x2):
        x1 = self._preprocess(x1)
        x2 = self._preprocess(x2)

        jaccard_similarity_list = []
        if len(x1) < 3 or len(x2) < 3:
            return 0.5

        for n in range(1, 4):
            x1_ngram = self._convert_to_ngram(x1, n)
            x2_ngram = self._convert_to_ngram(x2, n)
            print(x1_ngram)
            print(x2_ngram)

            # Jaccard Similarity
            union = set(x1_ngram).union(set(x2_ngram))
            intersection = set(x1_ngram).intersection(set(x2_ngram))
            jaccard_similarity = len(intersection) / len(union)
            jaccard_similarity_list.append(jaccard_similarity)

        ngram_similarity_weight = [0.1, 0.2, 0.7]
        # weighted average
        similarity = 0
        for i in range(len(jaccard_similarity_list)):
            similarity += jaccard_similarity_list[i] * ngram_similarity_weight[i]
        return similarity


if __name__ == "__main__":
    text_feature_extractor = TextFeatureExtractor()
    similarity = text_feature_extractor.calculate_similarity("Samsung", "Samdung")
    print(similarity)
    similarity = text_feature_extractor.calculate_similarity("Samsung", "SSAAMMUUUNNGGG")
    print(similarity)

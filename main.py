from calculate_similarity.image_similarity import ImageFeatureExtractor
from calculate_similarity.pronunciation_similarity import PronunciationFeatureExtractor
from calculate_similarity.text_similarity import TextFeatureExtractor

from preprocess.preprocess import preprocess_image

from config import Config

def main(tm_path_1, tm_path_2):
    config = Config()
    img_features_extractor = ImageFeatureExtractor(model_name='VGG19')
    text_features_extractor = TextFeatureExtractor()
    pronunciation_features_extractor = PronunciationFeatureExtractor()

    org_i1, masked_i1, t1 = preprocess_image(tm_path_1)
    org_i2, masked_i2, t2 = preprocess_image(tm_path_2)


    org_img_similarity = img_features_extractor.calculate_similarity(org_i1, org_i2)
    masked_img_similarity = img_features_extractor.calculate_similarity(masked_i1, masked_i2)

    img_similarity = org_img_similarity * (1 - config.mask_img_ratio) + masked_img_similarity * config.mask_img_ratio
    text_similarity = text_features_extractor.calculate_similarity(t1, t2)
    pronunciation_similarity = pronunciation_features_extractor.calculate_similarity(t1, t2)

    print('Original Image Similarity: ', org_img_similarity)
    print('Masked Image Similarity: ', masked_img_similarity)
    print('Image Similarity: ', img_similarity)
    print('Text Similarity: ', text_similarity)
    print('Pronunciation Similarity: ', pronunciation_similarity)





    
    
    


if __name__ == '__main__':
    import sys
    tm_path_1 = '/Users/june/tmsearch/data/dev/1.jpeg'
    tm_path_2 = '/Users/june/tmsearch/data/dev/1.jpeg'
    
    main(tm_path_1, tm_path_2)
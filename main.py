from calculate_similarity.image_similarity import ImageFeatureExtractor
from calculate_similarity.pronunciation_similarity import PronunciationFeatureExtractor
from calculate_similarity.text_similarity import TextFeatureExtractor

from preprocess.preprocess import preprocess_image

from config import Config

def tmsearch(tm_path_1, tm_path_2):
    config = Config()
    img_features_extractor = ImageFeatureExtractor(model_name='VGG19')
    text_features_extractor = TextFeatureExtractor()
    pronunciation_features_extractor = PronunciationFeatureExtractor()

    tm1_data = preprocess_image(tm_path_1)
    tm2_data = preprocess_image(tm_path_2)


    org_img_similarity = img_features_extractor.calculate_similarity(tm1_data['org_img'], tm2_data['org_img'])
    masked_img_similarity = img_features_extractor.calculate_similarity(tm1_data['masked_img'], tm2_data['masked_img'])
    design_text_img_similarity = img_features_extractor.calculate_similarity(tm1_data['merged_img'], tm2_data['merged_img'])

    mask_img_weight = ((1-tm1_data['bbox_area_ratio']) + (1-tm2_data['bbox_area_ratio'])) / 2 * config.text_img_split_data_weight
    design_text_img_ratio = (tm1_data['bbox_area_ratio'] + tm2_data['bbox_area_ratio']) / 2 * config.text_img_split_data_weight
    img_similarity = org_img_similarity * (1 - config.text_img_split_data_weight) + design_text_img_similarity * design_text_img_ratio + masked_img_similarity * mask_img_weight

    text_similarity = text_features_extractor.calculate_similarity(tm1_data['text'], tm2_data['text'])
    pronunciation_similarity = pronunciation_features_extractor.calculate_similarity(tm1_data['text'], tm2_data['text'])

    print('Original Image Similarity: ', org_img_similarity)
    print('Masked Image Similarity: ', masked_img_similarity)
    print('Design Text Image Similarity: ', design_text_img_similarity)
    print('Image Similarity: ', img_similarity)
    print('Text Similarity: ', text_similarity)
    print('Pronunciation Similarity: ', pronunciation_similarity)

    over_threshold = [False, False, False]

    for i, similarity in enumerate([img_similarity, text_similarity, pronunciation_similarity]):
        if similarity >= config.full_score_threshold[i]:
            over_threshold[i] = True
            print(f'Over {config.full_score_threshold[i]} threshold')
    
        

    
    
        
    
    total_similarity = img_similarity * config.similarity_ratio[0] + text_similarity * config.similarity_ratio[1] + pronunciation_similarity * config.similarity_ratio[2]
    print('Total Similarity: ', total_similarity)

    return {
        'org_img_similarity': org_img_similarity,
        'masked_img_similarity': masked_img_similarity,
        'design_text_img_similarity': design_text_img_similarity,
        'img_similarity': img_similarity,
        'text_similarity': text_similarity,
        'pronunciation_similarity': pronunciation_similarity,
        'total_similarity': total_similarity
    }




if __name__ == '__main__':
    # import os
    # same_lst = [['saboo', 'sobia'], ['madcatz', 'monster']]
    # tm_data_dir = '/Users/june/tmsearch/tm_data'
    # for s_list in same_lst:
    #     f1 = s_list[0] + '.jpg'
    #     f2 = s_list[1] + '.jpg'
    #     f1 = os.path.join(tm_data_dir, f1)
    #     f2 = os.path.join(tm_data_dir, f2)
    #     result_dict = main(f1, f2)
    #     print(f1, f2, result_dict)

    reports = open('reports2.txt', 'w')
    

    import os
    tm_data_dir = '/Users/june/tmsearch/tm_data'
    tm_data_lst = sorted(os.listdir(tm_data_dir))

    # remove file that in same_list
    # for same in same_lst:
    #     for same_data in same:
    #         if same_data+'.jpg' in tm_data_lst:
    #             print('remove ', same_data+'.jpg')
    #             tm_data_lst.remove(same_data+'.jpg')
    
    while True:
        if len(tm_data_lst) == 0:
            break
        tm1_path = tm_data_lst.pop(0)
        tm2_path = tm_data_lst.pop(0)
        tm1_path = os.path.join(tm_data_dir, tm1_path)
        tm2_path = os.path.join(tm_data_dir, tm2_path)

        result_dict = tmsearch(tm1_path, tm2_path)
        reports.write(f'{tm1_path} {tm2_path} {result_dict}\n')


    # tm_data_lst = [os.path.join(tm_data_dir, tm_data) for tm_data in tm_data_lst]

    # pop 2 same data
    
        
    

    
        


    # tm_path_1 = '/Users/june/tmsearch/data/dev/t1.jpeg'
    # tm_path_2 = '/Users/june/tmsearch/data/dev/t2.jpeg'
    
    # main(tm_path_1, tm_path_2)
class Config:
    # img size
    img_size = 224
    model = "VGG19"
    text_img_split_data_weight = 0.5

    # img text pronunciation
    full_score_threshold = [0.8, 0.75, 0.75]
    similarity_ratio = [0.2, 0.4, 0.4]

    

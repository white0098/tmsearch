from calculate_similarity.image_similarity import ImageFeatureExtractor

def main():
    img_features_extractor = ImageFeatureExtractor(model_name='VGG19')
    # tr
    img_features = img_features_extractor.forward(x)
    


if __name__ == '__main__':
    main()
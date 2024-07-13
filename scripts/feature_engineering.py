import numpy as np
import pandas as pd
import gender_guesser.detector as gender

def predict_sex(name):
    sex_predictor = gender.Detector(case_sensitive=False)
    first_name = name.str.split(' ').str.get(0)
    sex = first_name.apply(sex_predictor.get_gender)
    sex_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'andy': 0, 'mostly_male': 1, 'male': 2}
    sex_code = sex.map(sex_dict).astype(int)
    return sex_code

def extract_features(x):
    lang_list = list(enumerate(np.unique(x['lang'])))
    lang_dict = {name: i for i, name in lang_list}
    x['lang_code'] = x['lang'].map(lambda x: lang_dict[x]).astype(int)
    x['sex_code'] = predict_sex(x['name'])
    feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'sex_code', 'lang_code']
    x = x[feature_columns_to_use]
    return x

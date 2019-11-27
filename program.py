import category_encoders as ce
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from consts import NUMERIC_FEATURES, CATEGORICAL_FEATURE, NA_VALUES, CONVERTERS, FEATURES

crime_level_average = {}


def get_average(target, countries, crime_levels):
    if target.lower() in crime_level_average:
        return crime_level_average[target.lower()]

    sum = num = i = 0
    for country in countries:
        if country.lower() == target.lower():
            sum += crime_levels[i]
            num += 1
        i += 1

    crime_level_average[target.lower()] = sum // num
    return crime_level_average[target.lower()]


def one_hot(series):
    encoder = OneHotEncoder()
    return encoder.fit_transform(series.values.reshape(-1, 1)).toarray()


def impute(series, method):
    imputer = SimpleImputer(strategy=method)
    return imputer.fit_transform(series.values.reshape(-1, 1)).reshape(-1, 1)


def robust_scaler(values):
    robust = RobustScaler()
    return robust.fit_transform(values)


def ordinal_binary(series):
    ordinal = OrdinalEncoder()
    binary = ce.BinaryEncoder()

    tmp = ordinal.fit_transform(series.values.reshape(-1, 1))
    return binary.fit_transform(tmp)


def preprocess(raw_data):
    # separate each column to process each column
    feature_data = {key: raw_data[key] for key in FEATURES}
    processed_data = {key: None for key in FEATURES}

    # fill NA in categorical data with missing
    for cat_key in CATEGORICAL_FEATURE:
        feature_data[cat_key].fillna('missing', inplace=True)

    '''Year of Record'''
    # fill NAs with year from last valid example
    tmp = feature_data['Year of Record']
    tmp.fillna(method='pad', inplace=True)
    processed_data['Year of Record'] = tmp.values.reshape(-1, 1)

    '''Housing Situation'''
    tmp = feature_data['Housing Situation']
    processed_data['Housing Situation'] = one_hot(tmp)

    '''Crime Level in the City of Employement'''
    # fill NAs with the average crime level of the country
    tmp = feature_data['Crime Level in the City of Employement'].values
    crime_levels = feature_data['Crime Level in the City of Employement'].values
    countries = feature_data['Country'].values
    for i in range(len(tmp)):
        if np.isnan(tmp[i]) or tmp[i] == 0:
            tmp[i] = get_average(countries[i], countries, crime_levels)
    processed_data['Crime Level in the City of Employement'] = tmp.reshape(-1, 1)

    '''Work Experience in Current Job [years]'''
    tmp = feature_data['Work Experience in Current Job [years]']
    processed_data['Work Experience in Current Job [years]'] = impute(tmp, 'mean')

    '''Satisfation with employer'''
    tmp = feature_data['Satisfation with employer']
    processed_data['Satisfation with employer'] = one_hot(tmp)

    '''Gender'''
    tmp = feature_data['Gender']
    processed_data['Gender'] = one_hot(tmp)

    '''Age'''
    tmp = feature_data['Age']
    processed_data['Age'] = impute(tmp, 'mean')

    '''Country'''
    tmp = feature_data['Country']
    processed_data['Country'] = ordinal_binary(tmp)

    '''Size of City'''
    tmp = feature_data['Size of City']
    processed_data['Size of City'] = robust_scaler(impute(tmp, 'mean'))

    '''Profession'''
    tmp = feature_data['Profession']
    processed_data['Profession'] = ordinal_binary(tmp)

    '''University Degree'''
    tmp = feature_data['University Degree']
    processed_data['University Degree'] = one_hot(tmp)

    '''Wears Glasses'''
    tmp = feature_data['Wears Glasses']
    processed_data['Wears Glasses'] = one_hot(tmp)

    '''Hair Color'''
    tmp = feature_data['Hair Color']
    processed_data['Hair Color'] = one_hot(tmp)

    '''Body Height [cm]'''
    tmp = feature_data['Body Height [cm]']
    processed_data['Body Height [cm]'] = impute(tmp, 'mean')

    '''Yearly Income in addition to Salary (e.g. Rental Income)'''
    tmp = feature_data['Yearly Income in addition to Salary (e.g. Rental Income)']
    processed_data['Yearly Income in addition to Salary (e.g. Rental Income)'] = robust_scaler(impute(tmp, 'mean'))

    return processed_data


def main():
    numeric_dtypes = {key: np.float64 for key in NUMERIC_FEATURES}
    categorical_dtypes = {key: np.str for key in CATEGORICAL_FEATURE}
    numeric_dtypes.update(categorical_dtypes)
    dtypes = numeric_dtypes

    # extracting data set using predefined types and Na values.
    # Yearly Income in addition to Salary (e.g. Rental Income) is converted to float using converter defines in consts
    train_dataset = pd.read_csv('data/tcd-ml-1920-group-income-train.csv', dtype=dtypes, na_values=NA_VALUES,
                                converters=CONVERTERS)
    test_dataset = pd.read_csv('data/tcd-ml-1920-group-income-test.csv', dtype=dtypes, na_values=NA_VALUES,
                               converters=CONVERTERS)

    # remove duplicate entries in training set
    train_dataset = train_dataset.drop_duplicates()

    # remove extreme examples
    train_dataset = train_dataset[train_dataset['Total Yearly Income [EUR]'] >= 0]
    train_dataset = train_dataset[train_dataset['Total Yearly Income [EUR]'] <= 2500000]

    # number of examples in training and test set
    train_m = train_dataset.shape[0]
    test_m = test_dataset.shape[0]

    train_dataset = train_dataset.drop('Instance', 1)
    test_dataset = test_dataset.drop('Instance', 1)

    # isolate Y values
    Y = np.log(train_dataset.pop('Total Yearly Income [EUR]').values)
    test_dataset.pop('Total Yearly Income [EUR]')

    # combine all data to generate categorical encodings
    dataset = train_dataset.append(test_dataset)

    assert (dataset.shape[0] == train_m + test_m)

    # pre-process data for training
    processed_data = preprocess(dataset)

    tmp_dataset = [value for value in processed_data.values()]

    dataset = np.concatenate(tmp_dataset, axis=1)

    assert (dataset.shape[0] == train_m + test_m)

    # separate processed data back into training set and test set
    train_dataset = dataset[:train_m]
    test_dataset = dataset[train_m:]

    assert (train_dataset.shape[0] == train_m)
    assert (test_dataset.shape[0] == test_m)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    params = {
        'max_depth': 20,
        'learning_rate': 0.002,
        "boosting": "gbdt",
        "bagging_seed": 11,
        "metric": 'mae',
        "verbosity": -1,
    }

    X = train_dataset

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    train_data = lgb.Dataset(X_train, label=Y_train)
    test_data = lgb.Dataset(X_test, label=Y_test)

    # training using lightgbm
    model = lgb.train(params, train_data, 100000, valid_sets=[train_data, test_data], verbose_eval=1000,
                      early_stopping_rounds=500)

    test_pred = model.predict(X_test)

    err = mean_absolute_error(np.exp(Y_test), np.exp(test_pred))

    model.save_model('model.txt')

    print(err)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # predict values
    predicted = model.predict(test_dataset)
    predicted = np.exp(predicted)

    pd.DataFrame(predicted).to_csv("result.csv")


if __name__ == '__main__':
    main()

import pandas as pd
import pickle

# loading csv's
test_csv = pd.read_csv('competitionset.csv')

test_data = test_csv.copy()

# Add Claim Amount when training, remove when testing
test_data_in = test_data.loc[:, ['feature1', 'feature2', 'feature4', 'feature5', 'feature6', 'feature7',
                                      'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature14',
                                      'feature15', 'feature16', 'feature17', 'feature18']]

regr_test = test_data.copy()
drop_features = ['feature4', 'feature9', 'feature13', 'feature14', 'feature15',
                                      'feature18']
regr_test = regr_test.drop(drop_features, axis=1)

loaded_clf = pickle.load(open('xgb.sav', 'rb'))
load_predict = loaded_clf.predict(test_data_in)

originalTest = test_data.copy()
originalTest['PredictedCategory'] = load_predict

clf = pickle.load(open('rf_model.sav', 'rb'))
y_pred_val = clf.predict(regr_test)

export = pd.DataFrame(columns=['ClaimAmount'])
export['ClaimAmount'] = y_pred_val
for i in range(len(export)):
    if originalTest.iloc[i]['PredictedCategory'] == 0:
        export.iloc[i, export.columns.get_loc('ClaimAmount')] = 0
export.index.name = 'rowIndex'
export.to_csv('predictedclaimamount.csv')
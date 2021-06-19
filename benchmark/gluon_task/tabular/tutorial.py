from autogluon.tabular import TabularDataset, TabularPredictor
train_data = TabularDataset('train.csv')
subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head()
label = 'class'
print("Summary of class variable: \n", train_data[label].describe())
save_path = 'output'  # specifies folder to store trained models
predictor = TabularPredictor(label=label, path=save_path).fit(train_data)
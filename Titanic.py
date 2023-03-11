import tensorflow as tf
from tensorflow.keras.optimizers.legacy import SGD
import pandas as pd
import numpy as np
import shutil

df=pd.read_csv("titanic.csv")
print(df.columns)
df.drop(['SibSp','Parch'],axis=1,inplace=True)
print(df.head())
print(df.columns)
keys=['Survived','Pclass','Sex','Age','Fare'] #2 sex
caracteristique=keys[1:len(keys)]
etiquette=keys[0]

msk = np.random.rand(len(df)) < 0.8
traindf = df[msk]
evaldf = df[~msk]
def train_input_fn(traindf):
	return tf.compat.v1.estimator.inputs.pandas_input_fn(
		x=traindf[caracteristique],
		y=traindf[etiquette],
		batch_size=40,
		num_epochs=500,
		shuffle=True,
		queue_capacity=1000
		)
def eval_input_fn(evaldf):
	return tf.compat.v1.estimator.inputs.pandas_input_fn(
		x=evaldf[caracteristique],
		y=evaldf[etiquette],
		batch_size=40,
		shuffle=False,
		queue_capacity=1000
		)
def predict_input_fn(newdf):
	return tf.compat.v1.estimator.inputs.pandas_input_fn(
		x=newdf,
		y=None,
		batch_size=10,
		shuffle=False)
categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key='Sex',vocabulary_list=('male','female'))#Indicator columns and embedding columns never work on features directly
featuress=[tf.feature_column.numeric_column('Pclass'),
		   tf.feature_column.numeric_column('Fare'),
		   tf.feature_column.numeric_column('Age'),
		   tf.feature_column.indicator_column(categorical_column)]
outdir='titanic_trained2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

opt = SGD(learning_rate=0.01, momentum=0.9)
model = tf.estimator.DNNRegressor(hidden_units=[15,10,5], feature_columns=featuress, model_dir=outdir, activation_fn=tf.nn.sigmoid, optimizer=opt)
model.train(train_input_fn(df))

def print_rmse(model, df):
  metrics = model.evaluate(input_fn = eval_input_fn(evaldf))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))

print_rmse(model,evaldf)
tf.estimator.RunConfig(model_dir=outdir)


while True:
	s=input("Genre (male/female) :")
	a=input("Classe (1,2,3): ")
	b=input("age: (:80): ")
	c=input("tarif (0:40): ")
	newdf=pd.DataFrame({'Pclass':[int(a)],'Age':[int(b)],'Sex':[str(s)],'Fare':[int(c)]})
	prediccion=model.predict(predict_input_fn(newdf))
	print(next(prediccion))
	s=str(input("sortir? [y/n]:"))
	if s=="y":
		break;
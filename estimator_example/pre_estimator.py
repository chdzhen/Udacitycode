"""
estimator example!!!
"""
import tensorflow as tf
import numpy as np

#-- Pre-made Estimators--
# 1. Creator input function
def train_input_fn(features,labels,batch_size):
    #Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices(dict(features),labels)
    #shuffle,repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    #return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

# An input function for evaluation or prediction
def eval_input_fn(features,labels,batch_size):
    features=dict(features)
    if labels is None:
        #No labels,use only features
        inputs = features
    else:
        inputs = (features,labels)
    #Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    #Btach the examples
    assert batch_size is not None,"batch_size must not be None"
    dataset = dataset.batch(batch_size)

# Features columns describe how to use the input
my_feature_columns = []
for key in features.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#---------------------------------pre_estimator---------------------------------#
# Build 2 hidden layer DNN with 10,10 units respectively
classifier = tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns,
    hidden_units = [10,10],
    n_classes=3
) 

# Train the model 
classifier.train(
    input_fn=lambda:train_input_fn(train_x,train_y,args.batch_size),steps=args.train_steps
)
# Evaluate the model
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x,test_y,args.batch_size)
)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Generate predictions from the model
expected  = ['Setose','Versicolor','Virginica']
predict_x = {
    'SepalLength':[5.1,5.9,6.9],
    'SepalWidth':[3.3,3.0,3.1],
    'PetalLength':[1.7,4.2,5.4],
    'PetalWidth':[0.5,1.5,2.1],
}
predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(predict_x,batch_size= agrs.batch_size)
)
for pred_dict,expec in zip(predictions,expected):
    template = ('\nPrediction is "{}" ({:.1f}%),expected "{}"')
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print(template.format(iris_data.SPECIES[class_id],100*probability,expec))



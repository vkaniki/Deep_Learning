# Skin Cancer Melanoma Detection Assignment
Problem statement: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.



This assignment uses a dataset of about 2357 images of skin cancer types. The dataset contains 9 sub-directories in each train and test subdirectories. The 9 sub-directories contains the images of 9 skin cancer types respectively.

Required To change the path of image locations

data_dir_train , data_dir_test &

#### getting the exact paths of each image in train data folder

#### getting the classes of each image dataset based on the folder name as per the skin cancer name

#### Use 80% of the images for training, and 20% for validation with batch_size = 32, img_height = 180, img_width = 180
 
        
#### The higher values of negative coeeficients suggest a decrease in sale value.

#### List out all the classes of skin cancer and store them in a list. 

#### Visualize the data for all the nine classes present in the dataset
    
The image_batch is a tensor of the shape (32, 180, 180, 3). This is a batch of 32 images of shape 180x180x3 (the last dimension refers to color channels RGB). The label_batch is a tensor of the shape (32,), these are corresponding labels to the 32 images.

Dataset.cache() keeps the images in memory after they're loaded off disk during the first epoch.

Dataset.prefetch() overlaps data preprocessing and model execution while training.

#### Model I : Simple model
Create the model
Todo: Create a CNN model, which can accurately detect 9 classes present in the dataset. Use layers.experimental.preprocessing.Rescaling to normalize pixel values between (0,1). The RGB channel values are in the [0, 255] range. This is not ideal for a neural network. Here, it is good to standardize values to be in the [0, 1]

model.compile(optimizer='adam',
loss="sparse_categorical_crossentropy",
metrics=['accuracy'])

#### Run with epochs = 20

Findings from the graph and model history
Cleraly we can see that the model is overfitting the train accuracy is 87% and the validation accuracy is only 57%
One of the reasons of overfitting could be lack of sufficient images as we know that CNN model require huge amount of images to learn, hence it looks like the model is memorizing the train images and therefore leading to overfitting¶

 ### Model II with Data augmentation layer
Create the model, compile and train the model and can use Dropout layer if there is an evidence of overfitting in your findings and activation function softmax

### Write your findings after the model fit, see if there is an evidence of model overfit or underfit. Do you think there is some improvement now as compared to the previous model run?¶
There has been a considerable improvement now as compared to the previous model as we can see from the train and validation accuracy that the model overfitting has been handled, but the model performance is not so good only 94% train and 59% validation accuracy

One reason could be class imbalance

Todo: Find the distribution of classes in the training dataset.
Context: Many times real life datasets can have class imbalance, one class can have proportionately higher number of samples compared to the others. Class imbalance can have a detrimental effect on the final model quality. Hence as a sanity check it becomes important to check what is the distribution of classes in the data.

* Plotting graph to detect class imbalance *

#### Write your findings here:
Question-1:

Which class has the least number of samples?
seborrheic keratosis class has the least number of samples, that is 77

Question -2 :

Which classes dominate the data in terms proportionate number of samples?
pigmented benign keratosis, melanoma and basal cell carcinoma are the top 3 dominant classes wrt the sample count

#### Rectify the class imbalance we can use the Augmentor library to add more images to the existing samples which can help to resolve the class imbalance issue.
Context: You can use a python package known as Augmentor (https://augmentor.readthedocs.io/en/master/) to add more samples across all classes so that none of the classes have very few samples.


#### Augmentor has stored the augmented images in the output sub-directory of each of the sub-directories of skin cancer types.. Lets take a look at total count of augmented images and see the distribution of augmented data after adding new images to the original training data.

#### we have added 500 images to all the classes to maintain some class balance. We can add more images as we want to improve training process.


#### Model III with added samples using Augmentor library
Create your model (make sure to include normalization), Compile your model (Choose optimizer and loss function appropriately


#### trainig the model with epochs = 30

# Business Goal 

Finally we could handle both overfitting and class imbalance issue and the model performance has increased considerably. train accuracy 93% and validation accuracy 81.80%

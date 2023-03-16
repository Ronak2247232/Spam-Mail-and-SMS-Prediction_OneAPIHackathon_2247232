
# Spam Mail and SMS Prediction

This project aims to predict spam mail and SMS using Python. The dataset used for this project contains messages labeled as spam or ham.



## 🎨 Inspiration

The inspiration for this project comes from the increasing amount of spam messages we receive on a daily basis. These spam messages can be annoying and time-consuming to deal with, especially if they are received frequently. Therefore, it is important to have a system that can automatically filter out these spam messages and only allow genuine messages to reach our inbox.

This is where the concept of spam detection comes in. Spam detection is the process of identifying and filtering out unwanted messages, such as spam emails and SMS. Machine learning algorithms can be used to automate this process by analyzing the content of the messages and classifying them as spam or not spam.


## Installation

Install my-project with npm

To run this project, you need to have Python 3 installed on your system. Additionally, you need to install the following libraries:
```bash
pandas
numpy
scikit-learn
nltk
```

Extract the dataset.

Open the spam_detection.ipynb notebook using Jupyter Notebook or any other notebook of your choice.
Run the notebook.

## 📝 Dataset
The dataset used for this project is the SMS Spam Collection Dataset from the UCI Machine Learning Repository. This dataset contains 5,574 messages labeled as spam or ham.

You can install these libraries using pip by running the following command:

```bash
pip install pandas numpy scikit-learn nltk
pip install scikit-learn-intelex
```

## 🚀 Methodology

The project follows the following steps:

✅ Data Preprocessing: The dataset is preprocessed by removing punctuations, converting all the words to lowercase, and removing stopwords.

✅ Feature Extraction: The text is converted into a numerical representation using the TF-IDF vectorization technique.

✅ Model Building: A Support Vector Machine (SVM) classifier is trained on the preprocessed and feature-extracted dataset.

✅ Model Evaluation: The trained SVM model is evaluated on a test dataset to check its accuracy and performance


## 🧩 Tech Stack

**Client:** StreamLit Framework

**Server:** Python, Flask


## Demo

Insert gif or link to demo

![]ProjectGIF.gif)

## 🎯 Lessons Learned

✅ Data Preprocessing: You can learn how to preprocess textual data by removing punctuations, converting all the words to lowercase, and removing stopwords. Preprocessing data is a crucial step in text classification problems as it helps in reducing the complexity of the dataset and makes it easier for the machine learning algorithm to understand.

✅ Feature Extraction: You can learn about the TF-IDF vectorization technique and how it is used to convert textual data into a numerical representation. Feature extraction is an important step in text classification problems as it helps in converting text data into a format that can be easily understood by the machine learning algorithm.

✅ Model Building: You can learn how to build a Support Vector Machine (SVM) classifier and how it is used in text classification problems. SVMs are a powerful machine learning algorithm that can be used for both classification and regression problems.

✅ Model Evaluation: You can learn how to evaluate a machine learning model by using different performance metrics such as accuracy, precision, recall, and F1-score. Evaluating a model is important to check its accuracy and performance.

✅ Python Programming: You can learn how to implement machine learning algorithms using Python. Python is a popular programming language that is widely used in the field of machine learning and data science.


## Roadmap

- Additional browser support

- Add more integrations


## Authors

- [@Ronak2247232](https://www.github.com/Ronak2247232)


## Acknowledgements

 - [StreamLit IO](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [UCI Machine Learning](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
 - [OneAPI](https://github.com/oneapi-src)


## ✒️ Future Scope

The project can be extended to other types of text classification problems. Additionally, more advanced models such as neural networks can be used for better accuracy.
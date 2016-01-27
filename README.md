# Spark-Naive-Bayes-text-classification
Demonstrates how to use spark to implement a Bayes text classifier

This code example demonstrates how to use spark to 

1. create machine learning pipeline
2. train a Naive Bayes text classifier
3. evaluate and explore the learned model
4. make predictions

The example uses the data set from [http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html](http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html)

<font color='red'>The spark example does not produce the same results as published in the  paper</font>

## Overview
See the JUnit test file src/test/java/com/santacruzintegration/spark/NaiveBayesStanfordExampleTest.java
<font color='red'>The unit test simply runs the code. It does not have any asserts or other invarient tests. I.E. all tests will always pass even thou the results do not match the published results </font>

## Running the code.
```
$ mvn clean test
```



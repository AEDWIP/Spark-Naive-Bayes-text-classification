/*The MIT License (MIT)

Copyright (c) 2016 Andrew E. Davidson, Andy@SantaCruzIntegration.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
package com.santacruzintegration.spark;

import java.util.Arrays;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * @see http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
 *
 */
public class StanfordNaiveBayesTextClassificationData {
    /**
     * The number of words in our vocabulary. We'll use this to set up HashingTF 
     * so that each word hashes into a unique buck while keeping the total number of
     * buckets as small as possible to make debugging easier
     * 
     * use first prime number greater than actual size of vocabulary
     * 
     * actual dictionary is 6 words. If dictionarySize is set we only hash into 4 buckets
     */
    public static final int dictionarySize = 7; 
    
    public DataFrame createTrainingData(JavaSparkContext jsc, SQLContext ssc) {
        // make sure we only use dictionarySize words
        JavaRDD<Row> rdd = jsc.parallelize(Arrays.asList(
                // 0 is Chinese
                // 1 in notChinese
                RowFactory.create(0, 0.0, Arrays.asList("Chinese", "Beijing", "Chinese")),
                RowFactory.create(1, 0.0, Arrays.asList("Chinese", "Chinese", "Shanghai")),
                RowFactory.create(2, 0.0, Arrays.asList("Chinese", "Macao")),
                RowFactory.create(3, 1.0, Arrays.asList("Tokyo", "Japan", "Chinese"))));
               
        return createData(rdd, ssc);
    }
    
    public DataFrame createTestData(JavaSparkContext jsc, SQLContext ssc) {
        JavaRDD<Row> rdd = jsc.parallelize(Arrays.asList(
                // 0 is Chinese
                // 1 in notChinese
                // "bernoulli" requires label to be IntegerType
                RowFactory.create(4, 1.0, Arrays.asList("Chinese", "Chinese", "Chinese", "Tokyo", "Japan"))));
        return createData(rdd, ssc);
    }
    
    private DataFrame createData(JavaRDD<Row> rdd, SQLContext sqlContext) {
        StructField id = null;
        id = new StructField("id", DataTypes.IntegerType, false, Metadata.empty());

        StructField label = null;
        label = new StructField("label", DataTypes.DoubleType, false, Metadata.empty());
       
        StructField words = null;
        words = new StructField("words", DataTypes.createArrayType(DataTypes.StringType), false, Metadata.empty());

        StructType schema = new StructType(new StructField[] { id, label, words });
        DataFrame ret = sqlContext.createDataFrame(rdd, schema);
        
        return ret;
    }
    
}

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


//import static org.junit.Assert.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
// do not use old API
//import org.apache.spark.mllib.classification.NaiveBayes;
//import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * @ see http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
 * @author Andrew E. Davidson
 *
 */

// the tests are really independent testBERNOULLI prints out
// the hash ids for all the words in the corpus. Its nice for debugging
// if it runs first
@FixMethodOrder(MethodSorters.NAME_ASCENDING) 
public class NaiveBayesStanfordExampleTest extends AbstractSparkTest implements Serializable { 
    private static final long serialVersionUID = 1L;
    public static Logger logger = LoggerFactory.getLogger(NaiveBayesStanfordExampleTest.class); 
    
    public enum ModelType {BERNOULLI, MULTINOMIAL, TDINF};
    
    @Test
    public void testBERNOULLI() {
        StanfordNaiveBayesTextClassificationData data = new StanfordNaiveBayesTextClassificationData();
        DataFrame rawTrainingDF = data.createTrainingData(javaSparkContext, sqlContext);
//      rawTrainingDF.printSchema();
//      rawTrainingDF.show(false);
        
        DataFrame rawTestDF = data.createTestData(javaSparkContext, sqlContext);
        
        // pipeline
        // HashingTF -> SparseVectorToLogicalTranformer -> NaiveBayes
        testImpl(rawTrainingDF, rawTestDF, ModelType.BERNOULLI, true);
        
        // TODO Add Asserts
    }
    
    @Test
    public void testMULTINOMIAL() {
        StanfordNaiveBayesTextClassificationData data = new StanfordNaiveBayesTextClassificationData();
        DataFrame rawTrainingDF = data.createTrainingData(javaSparkContext, sqlContext);
//      rawTrainingDF.printSchema();
//      rawTrainingDF.show(false);
        
        DataFrame rawTestDF = data.createTestData(javaSparkContext, sqlContext);
   
        // pipeline
        // HashingTF -> NaiveBayes
        testImpl(rawTrainingDF, rawTestDF, ModelType.MULTINOMIAL, false); 
        
        // TODO Add Asserts
    }
    
    @Test
    public void testTDINF() {
        StanfordNaiveBayesTextClassificationData data = new StanfordNaiveBayesTextClassificationData();
        DataFrame rawTrainingDF = data.createTrainingData(javaSparkContext, sqlContext);
//      rawTrainingDF.printSchema();
//      rawTrainingDF.show(false);
        
        DataFrame rawTestDF = data.createTestData(javaSparkContext, sqlContext);
        
        // pipeline
        // HashingTF -> IDF -> NaiveBayes
        testImpl(rawTrainingDF, rawTestDF, ModelType.TDINF, false);
        
        // TODO Add Asserts
    }
    
    /**
     * 1) creates pipeline <br />
     * 2) trains/fits NaiveBayes model <br />
     * 3) makes predictions on training set. This under estimates the the
     * out of sample error how ever is useful for debugging the implementation. <br />
     * 4) explores the trained model. I.E. prints out Pi, theta, ... <br />
     * 5) makes predictions on our test set
     * 6) evaluates how well our model worked. i.e. prints confusion matrix
     * 
     * @param modelType
     * @param debug if true prints out hash ids for all words in our corpus
     */
    private void testImpl(DataFrame rawTrainingDF, DataFrame rawTestDF, ModelType modelType, boolean debug) {
        logger.warn("\n\n\n***************BEGIN {}", modelType);
        
        PipelineModel nbPipelineModel = createPipeLine(rawTrainingDF, modelType);

        // print out the hash id's for each word in our vocabulary
        // we'll need this to be able understand the model's theta, and pi
        DataFrame trainingWordsToHash = createWordToHash(rawTrainingDF);
        DataFrame wordsToHashId = calculateHashIds(trainingWordsToHash, debug);
        
        // get our trained model from the pipeline model
        NaiveBayesModel nbModel = null;
        for (Transformer transformer : nbPipelineModel.stages()) {
            if (transformer.uid().equals(NaiveBayesObjName)) {
                nbModel = (NaiveBayesModel) transformer;
                break;
            }
        }
                
        logger.warn("\nmake predicitons using training data. This will under estimate the out of sample error");
        DataFrame trainingPredictionsDF = nbPipelineModel.transform(rawTrainingDF);
        trainingPredictionsDF.show(false);
        
        exploreModel(nbModel, wordsToHashId);
        
        DataFrame predictionDF = nbPipelineModel.transform(rawTestDF);
        logger.warn("\npredictionDF.show()");
        //predictionDF.printSchema();
        predictionDF.show(false);
        
        // evaluate model
        createConfusionMatrix(predictionDF);
        
        // TODO save and load     
        
        logger.warn("\n\n\nEND {}***************", modelType);
    }


    final String HashingTFUIDName = "myHashingTF";
    final String IDFUIDName = "myIDF";
    final String NaiveBayesObjName = "myNaiveBayes";
 
    /**
     * creates pipe line and trains/fits model to rawDF
     * 
     * @param rawDF
     * @param modelType
     * @return
     */
    private PipelineModel createPipeLine(DataFrame rawDF, ModelType modelType) {
        List<PipelineStage> stages = new ArrayList<PipelineStage>(5);

        // ModelType.MULTINOMIAL is the default
        
        // All models use HashingTF
        //stage 1, calculate term frequency for each document
        HashingTF hashingTF = new HashingTF(HashingTFUIDName)
            .setInputCol("words")
            .setOutputCol("tf")
            .setNumFeatures(StanfordNaiveBayesTextClassificationData.dictionarySize);
        stages.add(hashingTF);
        String nbFeaturesCol = hashingTF.getOutputCol();
        
        if (modelType == ModelType.TDINF) {
            IDF idf = new IDF(IDFUIDName)
                    // .setMinDocFreq(1) // our vocabulary has 6 words we hash into 7, docs are very small
                    .setInputCol(hashingTF.getOutputCol())
                    .setOutputCol("idf");
            
            stages.add(idf);
            nbFeaturesCol = idf.getOutputCol();
        }
        else if (modelType == ModelType.BERNOULLI) {
            SparseVectorToLogicalTranformer svlt = new SparseVectorToLogicalTranformer();
            svlt.setInputCol(hashingTF.getOutputCol());
            svlt.setOutputCol("logicVector");
            
            stages.add(svlt);
            nbFeaturesCol = svlt.getOutputCol();
        }
        
        NaiveBayes nb = new NaiveBayes(NaiveBayesObjName)
                                .setLabelCol("label")
                                .setFeaturesCol(nbFeaturesCol)
                                .setSmoothing(1.0) // this is default, mlib NaiveBayesModel.getSmoothing() throws exception
                                ;
        
        if (modelType == ModelType.BERNOULLI) {
            nb.setModelType("bernoulli"); // TODO see if a constant or enum is defined some where
        }
        
        stages.add(nb);
        
        
        Pipeline pipeline = new Pipeline();
        PipelineStage[] ps = stages.toArray(new PipelineStage[stages.size()]);
        pipeline.setStages(ps);
        
        PipelineModel ret =  pipeline.fit(rawDF);
        return ret;
    }


    /**
     * 
     * @param wordsToHash
     * @param debug if true prints out hash ids for all words in our corpus
     * @return
     */
    private DataFrame calculateHashIds(DataFrame wordsToHash, boolean debug) {
        // create UDF and register
        SparseVectorToHashIdUDF udf = new SparseVectorToHashIdUDF();
        DataType returnType = DataTypes.IntegerType;
        final String udfName = "sparseVectorToHashIdUDF";
        sqlContext.udf().register(udfName, udf, returnType);
        String fmt = "%s(%s) as %s";
        
        // configure UDF and execute
        final String inputCol = "features"; 
        final String outputCol = "hashId";
        String stmt = String.format(fmt, udfName, inputCol, outputCol);
        //logger.warn("\nstmt: {}", stmt);
        // not sure why but this requires our unit test to be serializable.
        DataFrame ret = wordsToHash.selectExpr("*", stmt);
        
        if (debug) {
            logger.warn("\nwordsToHashId.show()");
            // ret.printSchema();
            ret.show(false);
        }
        
        return ret;
    }

    private void exploreModel(NaiveBayesModel nbModel, DataFrame wordsToHashId) {
        logger.warn("nbModel.getModelType(): {}", nbModel.getModelType());
        logger.warn("nbModel.numClasses(): {}", nbModel.numClasses());
        logger.warn("nbModel.numFeatures(): {}", nbModel.numFeatures());
        logger.warn("nbModel.getSmoothing(): {}", nbModel.getSmoothing());

        //“theta, the matrix of class probabilities for each feature 
        // (of size  for C classes and D features),
         Matrix thetaM = nbModel.theta();
         double[] theta = thetaM.toArray();

         int numRows = nbModel.numClasses();
         int numColumns = nbModel.numFeatures(); 
         double[][] dm = new double[numRows][numColumns];
         reshape(dm, theta); 
         logger.warn("\ntheta\n{}", printable(dm));
         
         double[][] expDm = new double[numRows][numColumns];
         for (int i = 0; i < expDm.length; i++) {
             for (int j = 0; j < expDm[0].length; j++) {
                 expDm[i][j] = Math.exp(dm[i][j]);
             }
         }
         
         List<String> colHdrs = getWords(wordsToHashId);
         StringBuffer colHdrBuf = new StringBuffer();
         String hdrFmt = "%-8s";
         for (String hdr : colHdrs) {
             colHdrBuf.append(String.format(hdrFmt, hdr));
         }
         logger.warn("\nexp(theta)\n{}\n{}", colHdrBuf, printable(expDm));

         //
         // pi, the C-dimensional vector of class priors.”
         // prior probability, i.e. distribution
         // of an uncertain quantity is the probability distribution 
         // that would express one's beliefs about this quantity before 
         // some evidence is taken into account.
         //
         // example
         // For example, three acres of land have the labels A, B and C. 
         // One acre has reserves of oil below its surface, while the 
         // other two do not. The probability of oil being on acre C is 
         // one third, or 0.333. A drilling test is conducted on acre B,
         // and the results indicate that no oil is present at the location. 
         // Since acres A and C are the only candidates for oil reserves, 
         // the prior probability of 0.333 becomes 0.5, as each acre has one out of two chances.
         //
         
         double[] pi = nbModel.pi().toArray();
         StringBuffer piBuff = new StringBuffer();
         String fmt = "%f %f\n";
         for(int i = 0; i < pi.length; i++) {
             String r = String.format(fmt, pi[i], Math.exp(pi[i]));
             piBuff.append(r);
         }
         logger.warn("\npi, the class priors\n-------------------\npi[i]\t| exp(pi[i])\n-------------------\n{}", piBuff);
    }

    // returns words sorted by 'hashId' columns
    private List<String> getWords(DataFrame wordsToHash) {
        List<String> ret = new ArrayList<String>(StanfordNaiveBayesTextClassificationData.dictionarySize);
        List<Row> indices = wordsToHash.select("word", "hashId")
                .sort("hashId")
                .collectAsList();
        
        for (int i = 0; i < indices.size(); i++) {
            Row r = indices.get(i);
            String word = r.getString(0);
            ret.add(word);
        }
        
        return ret;
    }

    /**
     * this works because we know input was a single word 
     * 
     * @author andrewdavidson
     *
     */
    class SparseVectorToHashIdUDF implements UDF1<Vector, Integer>, Serializable {
        private static final long serialVersionUID = 1L;

        @Override
        public Integer call(Vector v) throws Exception {
            SparseVector sv = (SparseVector) v;
            int[] idx = sv.indices();
            Integer ret = new Integer(idx[0]);
            return ret;
        }
    }

 
    /**
     * prints ConfusionMatrix matrix
     * @param df
     */
    private void createConfusionMatrix(DataFrame df) {
                
        RandomString rs = new RandomString(5);
        final String tableName = "nbTable_" + rs.nextString();
        df.registerTempTable(tableName); 
        //results.printSchema();
        //results.show();
        
        ConfusionMatrix cm = new ConfusionMatrix(sqlContext, tableName);
        cm.calculate("label", "prediction");
        logger.warn("ConfusionMatrix\n" + cm.toString());
    }    
    
 
    /**
     * 1) flattens the words column<br />
     * 2) converts type of word column to list<String> each list has a single word<br />
     * 3) sets hashingTF input column to output column of step 2.<br />
     * 4) returns df appended with hashingTF output<br />
     * 
     * @param df
     * @return
     */
    private DataFrame createWordToHash(DataFrame df) {
        // flatten the words column
        String[] colNames = df.columns(); // all the col names
        DataFrame flatDF = df.select(
                org.apache.spark.sql.functions.explode(df.col("words")).as("word"))
                .distinct();
        
//        logger.warn("\nflatDF.printSchema()");;
//        flatDF.printSchema();
//        flatDF.show(false);
        
        // figure out how each word hashes
        // HashingTF input is List<String>
        // register a UDF to convert each word into a list of a single word
        DataType returnType = DataTypes.createArrayType(DataTypes.StringType);
        final String udfName = "wordToListWordUDF";
        WordToListWordUDF udf = new WordToListWordUDF();
        sqlContext.udf().register(udfName, udf, returnType);
       
        // call udf
        final String inputCol = "word";
        final String outputCol = "hashInput";
        String fmt = "%s(%s) as %s";
        String stmt = String.format(fmt, udfName, inputCol, outputCol);
        //logger.warn("\nstmt: {}", stmt);
        // not sure why but this requires our unit test to be serializable.
        DataFrame hashableDF = flatDF.selectExpr("*", stmt);
        
//        logger.warn("\nhashableDF.printSchema()");
//        hashableDF.printSchema();
//        hashableDF.show(false );
        
        HashingTF hashingTF = new HashingTF()
                .setInputCol(outputCol)               
                .setOutputCol("features")
                .setNumFeatures(StanfordNaiveBayesTextClassificationData.dictionarySize);
    
        DataFrame wordToHash = hashingTF.transform(hashableDF);
//        logger.warn("\nwordToHash.printSchema()");
//        wordToHash.printSchema();
//        wordToHash.show(false);
        
        return wordToHash;
    }
    
    private String printable(double[][] dm) {
        final int rows = dm.length;
        final int cols = dm[0].length;
        StringBuffer matrix = new StringBuffer(100);
        
        for(int i = 0; i < rows; i++ ) {
            StringBuffer row = new StringBuffer(100);
            for (int j = 0; j < cols; j++) {
                //row.append(dm[i][j] + ",\t");
                row.append(String.format("%.4f \t", dm[i][j]));
            }
            row.append("\n");
            matrix.append(row);
        }
        
        return matrix.toString(); 
    }
 
    /**
     * assume matrix is column major ordering
     * 
     * @param a
     * @param v
     */
    void reshape(double[][] a, double[] v) {     
        for (int p= 0, i= 0; i < a.length; i++)
           for (int j= 0; j < a[i].length; j++)
              if (p == v.length) return;
              else a[i][j]= v[p++];
     }
    
    
    /**
     * convert each 'word' into a list 
     * HashingTF expects a list not individual words
     * 
     * @author andrewdavidson
     */
    class WordToListWordUDF implements UDF1<String, List<String>>, Serializable {
        private static final long serialVersionUID = 1L;

        @Override
        public List<String> call(String word) throws Exception {
            List<String> ret = new ArrayList<String>();
            ret.add(word);
            return ret;
        }
    }
    
}

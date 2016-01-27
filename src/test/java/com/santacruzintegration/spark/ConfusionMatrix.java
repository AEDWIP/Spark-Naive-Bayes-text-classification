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

import org.apache.spark.sql.SQLContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ConfusionMatrix {
    static Logger logger = LoggerFactory.getLogger(ConfusionMatrix.class);
    
    SQLContext sqlContext; 
    String tableName;
    
    double truePositives;
    double trueNegatives;
    double falsePositives;
    double falseNegatives;
    double accuracy;
    double sensitivity;
    double specificity;
    double positivePredictiveValue;
    double negativePredictiveValue;
    double count;
    
    @Override
    public String toString() {
        StringBuffer buff = new StringBuffer(150);
        buff.append("truePositives = " + truePositives + "\n");
        buff.append("trueNegatives = " + trueNegatives + "\n");
        buff.append("falsePositives = " + falsePositives + "\n");
        buff.append("falseNegatives = " + falseNegatives + "\n");
        buff.append("accuracy = " + accuracy + "\n");
        buff.append("sensitivity = " + sensitivity + "\n");
        buff.append("specificity = " + specificity + "\n");
        buff.append("positivePredictiveValue = " + positivePredictiveValue + "\n");
        buff.append("negativePredictiveValue = " + negativePredictiveValue + "\n");
        buff.append("count = " + count + "\n");
        
        return buff.toString();
    }
    /**
     * assume 0 == negative, and 1 == positive
     * @param df
     * @param labelColName
     * @param predictionColName
     */
    public ConfusionMatrix(SQLContext sqlContext, String tableName) {
        this.sqlContext = sqlContext;
        this.tableName = tableName;
    }
    
    public void calculate(String labelColName, String predictionColName) {
        // We need to run select a couple of times 
        // check to see if we need to cache table or not
        boolean wasCached = sqlContext.isCached(tableName);
        if (!wasCached) {
            sqlContext.cacheTable(tableName);
        }
        
        calcCount(labelColName, predictionColName);
        
        calcTruePositive(labelColName, predictionColName);
        calcTrueNegative(labelColName, predictionColName);
        calcFalsePositives(labelColName, predictionColName);
        calcFalseNegative(labelColName, predictionColName);
        
        accuracy = (truePositives + trueNegatives) / count * 100.0;
        sensitivity = (trueNegatives / (falsePositives + trueNegatives) * 100.0);
        specificity = trueNegatives / (falsePositives + trueNegatives) * 100.0;
        
        positivePredictiveValue = truePositives / (truePositives + falsePositives) * 100.0;
        negativePredictiveValue = trueNegatives / (falseNegatives + trueNegatives) * 100.0;
        
        if (!wasCached) {
            sqlContext.uncacheTable(tableName);
        }
    }
    
    private void calcCount(String labelColName, String predictionColName) {
        String fmt = "SELECT %s FROM %s";
        String stmt = String.format(fmt, predictionColName, tableName);
        count = sqlContext.sql(stmt).count();
    }

    private void calcFalseNegative(String labelColName, String predictionColName) {
        String fmt = "SELECT %s FROM %s where %s = 1.0 and %s = 0.0";
        String stmt = String.format(fmt, predictionColName, tableName, labelColName, predictionColName);
        logger.info("fn:select stmt \n{}", stmt);

        falseNegatives = sqlContext.sql(stmt).count();
        logger.info("falseNegatives:{}", falseNegatives);  
    }

    private void calcFalsePositives(String labelColName, String predictionColName) {
        String fmt = "SELECT %s FROM %s where %s = 0.0 and %s = 1.0";
        String stmt = String.format(fmt, predictionColName, tableName, labelColName, predictionColName);
        logger.info("fp:select stmt \n{}", stmt);

        falsePositives = sqlContext.sql(stmt).count();
        logger.info("falsePositives:{}", falsePositives);       
    }

    private void calcTrueNegative(String labelColName, String predictionColName) {
        String fmt = "SELECT %s FROM %s  where %s = 0.0 AND %s = 0.0";
        String stmt = String.format(fmt, predictionColName, tableName, labelColName, predictionColName);
        logger.info("tn:select stmt \n{}", stmt);

        trueNegatives = sqlContext.sql(stmt).count();
        logger.info("trueNegatives:{}", trueNegatives);
    }

    private void calcTruePositive(String labelColName, String predictionColName) {
        String fmt = "SELECT %s FROM %s  where %s = 1.0 AND %s = 1.0";
        String stmt = String.format(fmt, predictionColName, tableName, labelColName, predictionColName);
        logger.info("tp:select stmt \n{}", stmt);

        truePositives = sqlContext.sql(stmt).count();
        logger.info("truePositive:{} select stmt \n{}", truePositives, stmt);        
    }

    /**
     * returns number of observation that are positive and we predict positive
     * @return
     */
    public double getTruePositives() {
        return truePositives;
    }
    
    /**
     * returns the number of observation that are negative and we predict negative
     * @return
     */
    public double getTrueNegatives() {
        return trueNegatives;
    }
    
    /**
     * returns the number of observation that are negative but predicted positive
     * @return
     */
    public double getFalsePositives() {
        return falsePositives;
    }
    
    /**
     * return the number of observation that are positive and we predict negative
     * @return
     */
    public double getFalseNegatives() {
        return falseNegatives;
    }
    
    /**
     * (truePositive + trueNegative) / count * 100)
     * @return
     */
    public double getAccuracy() {
        return accuracy;
    }
    
    /**
     * Sensitivity is the percentage of correct predictions when a player was injured
     * 
     * Sensitivity = truePositive / (truePositive + falseNegative) * 100
     * @return
     */
    public double getSensitivity() {
        return sensitivity;
    }
    
    /**
     * Specificity is percentage of correct predictions when a player was not injured
     * 
     * Specificity = trueNegative / (falsePositive + trueNegative) * 100
     * @return
     */
    public double getSpecificity() {
        return specificity;
    }
    
    /**
     * Positive Predictive value is percentage of correct predictions when we predict the player was injured
     * 
     * ppv = truePositive / (truePositive + falsePositive)
     * @return
     */
    public double getPositivePredictiveValue() {
        return positivePredictiveValue;
    }
    
    /**
     * Negative Predictive value is the percentage of correct predicitons when we predict the player was not injured
     *
     * pnv = trueNegative / (falseNegative + trueNegative) * 100
     * @return
     */
    public double getNegativePredictiveValue() {
        return negativePredictiveValue;
    }
}

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

import java.io.Serializable;
import java.util.UUID;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * all output component values of vector will be will be set to 1.0
 * <p />
 *
 * Input and output are assume to be SparceVector. 
 * <p />
 * TODO: generalize to support Vectors
 * @author andrewdavidson
 *
 */
public class SparseVectorToLogicalTranformer extends Transformer implements Serializable {
    private static final long serialVersionUID = 1L;
    static Logger logger = LoggerFactory.getLogger(SparseVectorToLogicalTranformer.class);
    final String uid = SparseVectorToLogicalTranformer.class.getSimpleName() + "_" + UUID.randomUUID().toString();
   
    ConvertToLogicVectorUDF udf = null;
    static final String udfName = "ConvertToLogicVectorUDF";
    
    String inputCol;
    String outputCol;
    
    @Override
    public String uid() {
      return uid;  
    }
    
    @Override
    public Transformer copy(ParamMap arg0) {
        logger.error("AEDWIP TODO copy {}", arg0);
        return this;
    }
    
    void registerUDF(SQLContext sqlContext) {
        if (udf == null) {
            udf = new ConvertToLogicVectorUDF();
            DataType returnType = new VectorUDT();
            sqlContext.udf().register(udfName, udf, returnType);
        }
    }
    
    @Override
    public DataFrame transform(DataFrame df) {
        registerUDF(df.sqlContext());
        
        String fmt = "%s(%s) as %s";
        String stmt = String.format(fmt, udfName, inputCol, outputCol);
        logger.info("\nstmt: {}", stmt);
        DataFrame ret = df.selectExpr("*", stmt);
        
        return ret;
    }
    
    @Override
    public StructType transformSchema(StructType ret) {
        StructField outField = null; 
        DataType returnType = new VectorUDT();
        outField = new StructField(outputCol, returnType, true, Metadata.empty());

        return ret.add(outField); 
    }
    
   static class ConvertToLogicVectorUDF implements UDF1<SparseVector, SparseVector>, Serializable {
        private static final long serialVersionUID = 1L;

        @Override
        public SparseVector call(SparseVector counts) throws Exception {
            //double[] dv = counts.values();
            double[] lv = new double[counts.values().length];
            for (int i = 0; i < lv.length; i++) {
                lv[i] = 1.0;
            }
            SparseVector logicV = new SparseVector(counts.size(), counts.indices(),  lv);
            return logicV;
        }
    }

    public String getInputCol() {
        return inputCol;
    }

    public void setInputCol(String inputCol) {
        this.inputCol = inputCol;
    }

    public String getOutputCol() {
        return outputCol;
    }

    public void setOutputCol(String outputCol) {
        this.outputCol = outputCol;
    }
    
}

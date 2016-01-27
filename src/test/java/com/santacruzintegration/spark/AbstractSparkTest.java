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
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SQLContext;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @see http://mkuthan.github.io/blog/2015/03/01/spark-unit-testing/
 * @author andrewdavidson
 *
 */
public abstract class AbstractSparkTest {
    public static Logger logger = LoggerFactory.getLogger(AbstractSparkTest.class);
    static final String master = "local[8]"; // streaming needs at least 2
    static final String appName = "PWS_UNIT_TEST";

    static public SparkContext sparkContext = null;
    static public JavaSparkContext javaSparkContext = null;
    static public SQLContext sqlContext = null;
    static public Boolean isRunning = false;

    static public boolean isRunning() {
        return (isRunning != null && isRunning);
    }

    @BeforeClass
    static public  void setUpBeforeClass() throws Exception {
        setUpBeforeClassImpl();
    }

    static protected void setUpBeforeClassImpl() {
        // TODO AEDWIP ideally if a *Test.java file has more than one test
        // each test would run in with its own context
        if (!isRunning()) {
            logger.info("Starting Spark test framework appName:{} master=:{}", appName, master);
            SparkConf conf = new SparkConf().setMaster(master).setAppName(appName);

            sparkContext = new SparkContext(conf);
            javaSparkContext = new JavaSparkContext(sparkContext);
            sqlContext = new org.apache.spark.sql.SQLContext(sparkContext);
            isRunning = true;
        }
    }

    @AfterClass
    static public void tearDownAfterClass() throws Exception {
        tearDownAfterClassImpl();
    }

    static protected void tearDownAfterClassImpl() {
        // TODO AEDWIP ideally if a *Test.java file had more than one unit test
        // each would run with it's own context
        if (isRunning()) {
            logger.info(" ");
            javaSparkContext.stop();
            javaSparkContext = null;
            sqlContext = null;
            sparkContext = null;
            logger.info("Spark test framework has been shutdown appName:{} master=:{}", appName, master);
            isRunning = false;
        }
    }
}

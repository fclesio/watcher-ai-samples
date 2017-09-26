
object Cells {
  /*
  +----------+----------+-----------+-----------------+--------------------+------------+------------------+---------+--------+--------------+
carrier_id|  datepart|hour_of_day|avg_response_time|       first_attempt|last_attempt|successful_charges|no_credit|  errors|total_attempts|
  +----------+----------+-----------+-----------------+--------------------+------------+------------------+---------+--------+--------------+
         1|2017-07-31|          0|            924.0|2017-07-31 00:00:...|      2017.0|           37911.0|1790648.0|145533.0|     1974092.0|
         1|2017-07-31|          1|            835.0|2017-07-31 01:00:...|      2017.0|            1500.0|1661866.0|136852.0|     1800218.0|
         1|2017-07-31|          2|            862.0|2017-07-31 02:00:...|      2017.0|            1320.0|1835148.0|159306.0|     1995774.0|
         1|2017-07-31|          3|           1093.0|2017-07-31 03:00:...|      2017.0|           12449.0|1775057.0|170922.0|     1958428.0|
         1|2017-07-31|          4|           1032.0|2017-07-31 04:00:...|      2017.0|            1231.0|1841635.0|170584.0|     2013450.0|
  +----------+-------+----------+-----------+-----------------+--------------------+------------+------------------+---------+--------+--------------+
  */

  /* ... new cell ... */

  /*
  +------------------+
Target			       |
  +------------------+
successful_charges|
  +------------------+
           37911.0|
            1500.0|
  +------------------+
  */
  
  /*
  +------------------+
Features	         |
  +----------+----------+-----------+-----------------+--------------------+------------+---------+--------+--------------+
carrier_id|  datepart|hour_of_day|avg_response_time|       first_attempt|last_attempt|no_credit|  errors|total_attempts|
  +----------+----------+-----------+-----------------+--------------------+------------+---------+--------+--------------+
         1|2017-07-31|          0|            924.0|2017-07-31 00:00:...|      2017.0|1790648.0|145533.0|     1974092.0|
         1|2017-07-31|          1|            835.0|2017-07-31 01:00:...|      2017.0|1661866.0|136852.0|     1800218.0|
  +----------+----------+-----------+-----------------+--------------------+------------+---------+--------+--------------+
  */
  
  // A plain vanilla case of Supervised Learning

  /* ... new cell ... */

  import org.apache.spark.ml.Pipeline
  import org.apache.spark.ml.PipelineModel
  import org.apache.spark.ml.evaluation.RegressionEvaluator
  import org.apache.spark.ml.feature.VectorIndexer
  import org.apache.spark.ml.feature.VectorAssembler
  import org.apache.spark.ml.regression.DecisionTreeRegressionModel
  import org.apache.spark.ml.regression.DecisionTreeRegressor
  import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
  import org.apache.spark.ml.tuning.TrainValidationSplitModel
  
  import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, TimestampType, DoubleType, DateType}
  import org.apache.spark.sql.expressions.Window
  import org.apache.spark.sql.functions._
  import java.util.Calendar
  import java.util.Date
  import org.apache.spark.mllib.tree.model.DecisionTreeModel
  
  import resource._
  
  val ROOT = "/Users/lucianosabenca/Movile/tools/spark-notebook-0.7.0-scala-2.11.8-spark-2.1.0-hadoop-2.7.1-with-hive/notebooks/datafest-2017"
  
  val session = SparkSession
        .builder
        .appName("DecisionTreePipeline")
        .getOrCreate()
  
  // You can monitoring all this stuff in: http://localhost:4040/jobs/
  
  val customSchema = StructType(Array(
      StructField("carrier_id", IntegerType, true),
      StructField("datepart", DateType, true),
      StructField("hour_of_day", IntegerType, true),
      StructField("avg_response_time", DoubleType, true),
      StructField("first_attempt", TimestampType, true),
      StructField("last_attempt", DoubleType, true),
      StructField("successful_charges", DoubleType, true),
      StructField("no_credit", DoubleType, true),
      StructField("errors", DoubleType, true),
      StructField("total_attempts", DoubleType, true)
  ))
  
  
  val df = session.read
          .format("org.apache.spark.csv")
          .option("header", "true") //reading the headers
          .option("delimiter", ",")
          .option("mode", "DROPMALFORMED")
          .schema(customSchema)
          .csv(ROOT + "/data-sample.csv");
  
  df.show(5)

  /* ... new cell ... */

  // function to extract the week of month from a timestamp
  val udfWoM = (dt :Date) => {
     val cal = Calendar.getInstance()
     cal.setTime(dt)
    
     cal.setMinimalDaysInFirstWeek(1)
     cal.get(Calendar.WEEK_OF_MONTH).toDouble
  }
  
  // user defined function
  val weekOfMonthUDF = udf(udfWoM)

  /* ... new cell ... */

  // apply UDF to extract the week of month from date time field
  val rawData = df.withColumn("week_of_month", weekOfMonthUDF($"datepart"))
  
  rawData.show(20)

  /* ... new cell ... */

  rawData.groupBy("week_of_month").count().show()

  /* ... new cell ... */

  val avgDataFrame = rawData.groupBy("carrier_id", "hour_of_day","week_of_month").agg(avg("avg_response_time").as("avg_response_time"), 
                                                                                          avg("successful_charges").as("successful_charges"), 
                                                                                          avg("no_credit").as("no_credit"), 
                                                                                          avg("errors").as("errors"),
                                                                                          avg("total_attempts").as("total_attempts"))
  
  
  
  avgDataFrame.show(10)

  /* ... new cell ... */

  rawData.where("carrier_id = 4 AND hour_of_day = 19 and WEEk_of_month = 5").show()

  /* ... new cell ... */

  avgDataFrame.where("carrier_id = 4 AND hour_of_day = 19 and WEEk_of_month = 5").show()

  /* ... new cell ... */

  val carrierList = List(1)
  
  // filtering data by carrier_id
  val listDataFrame = carrierList.map{id =>
      avgDataFrame.filter("carrier_id = " + id)
  }
  
  
  //means from beginning until current position
  val wSpec = Window.partitionBy("week_of_month").orderBy("hour_of_day").rowsBetween(Long.MinValue, 0)
  
  val summarizedList = listDataFrame.map{dataframe => 
      val df = dataframe.withColumn("avg_response_time", avg($"avg_response_time").over(wSpec))
                 .withColumn("successful_charges", sum($"successful_charges").over(wSpec))
                 .withColumn("no_credit", sum($"no_credit").over(wSpec))
                 .withColumn("errors", sum($"errors").over(wSpec))
                 .withColumn("total_attempts", sum($"total_attempts").over(wSpec))
      df //return the dataset                                     
  }
  
  summarizedList.foreach{_.show(3)}

  /* ... new cell ... */

  val summarizedDataFrame = summarizedList.reduce(_ union _)

  /* ... new cell ... */

  summarizedDataFrame.count

  /* ... new cell ... */

  //write data to disk in parquet format
  summarizedDataFrame.write.format("parquet").mode("overwrite").save(ROOT + "/sbs-summarized-dataframe")

  /* ... new cell ... */

  val summarizedDataFrame = session.read.load(ROOT + "/sbs-summarized-dataframe").cache

  /* ... new cell ... */

  // features
  val assemblerSucccesful = new VectorAssembler()
                        .setInputCols(Array("hour_of_day", "week_of_month", "avg_response_time","no_credit", "errors", "total_attempts"))
                        .setOutputCol("features_success")
  
  
  
  // creating label in log format
  val dataWithLabels = summarizedDataFrame.withColumn("successful_charges_log", log($"successful_charges"))
  
  
  val dataWithLabelsFiltered = dataWithLabels.filter("successful_charges_log is not null")

  /* ... new cell ... */

  def buildEvaluator(label: String, predictionCol: String): RegressionEvaluator = {
    new RegressionEvaluator()
      .setLabelCol(label)
      .setPredictionCol(predictionCol)
  }
  
  def evaluateMSE(df: DataFrame, label: String, predictionCol: String): Double = {
    val evaluator = buildEvaluator(label, predictionCol)
    evaluator.setMetricName("mse")
    evaluator.evaluate(df)
  }
  
  def evaluateR2(df: DataFrame, label: String, predictionCol: String): Double = {
    val evaluator = buildEvaluator(label, predictionCol)
    evaluator.setMetricName("r2")
    evaluator.evaluate(df)
  }
  
  def evaluateRMSE(df: DataFrame, label: String, predictionCol: String): Double = {
    val evaluator = buildEvaluator(label, predictionCol)
    evaluator.setMetricName("rmse")
    evaluator.evaluate(df)
  }

  /* ... new cell ... */

  // prints result of algorithm tested
  def buildStatsMaps(carrier: Double, col: Column, label: String, df: DataFrame, predictionCol: String): Map[String, Any] = {
    val calculateAcc = (exp: Double, predicted: Double) => {
      val error = (exp - predicted) / exp
      if (error > 0.1) 0 else 1
    }
  
    val calcAccuracyUDF = udf(calculateAcc)
  
    val rmse = evaluateRMSE(df, label, "prediction_log")
    val mse = evaluateMSE(df, label, "prediction_log")
    val r2 = evaluateR2(df, label, "prediction_log")
  
    val data = df.withColumn("result_column", calcAccuracyUDF(col, df(predictionCol)))
    val total = data.count.toDouble
    // filter prediction that got right
    val correct = data.filter("result_column = 1").count.toDouble
    val accuracy = (correct / total) * 100
  
    Map("rmse" -> rmse, "mse" -> mse, "r2" -> r2, "accuracy" -> accuracy)
  }

  /* ... new cell ... */

  val trainedModels = carrierList.map { c =>
    val label  = "successful_charges_log"
    val features = "features_success"
    val predictionColumn = "successful_charges"
    val assembler = assemblerSucccesful
    val data = dataWithLabels.filter(s"carrier_id = $c")
    val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))
  
    // Train a DecisionTree model.
    val decisionTree = new DecisionTreeRegressor()
        .setLabelCol(label)
        .setFeaturesCol(features)
        .setPredictionCol("prediction_log")
  
    val pipeline = new Pipeline().setStages(Array(assemblerSucccesful, decisionTree))
  
    val paramGrid = new ParamGridBuilder()
       .addGrid(decisionTree.maxDepth, Array(6, 7, 8))
       .addGrid(decisionTree.maxBins, (15 to 32).toList)
       .build()
  
    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
       .setLabelCol(label)
       .setPredictionCol("prediction_log")
       .setMetricName("rmse")
  
  
     val trainValidationSplit = new TrainValidationSplit()
       .setEstimator(pipeline)
       .setEvaluator(evaluator)
       .setEstimatorParamMaps(paramGrid)
       .setTrainRatio(0.8)
  
     //train a model
    val model = trainValidationSplit.fit(trainingData)
  
    //make predictions
    val predictions = model.transform(testData)
    val columnValue = s"prediction_$predictionColumn"
  
    val predictionResult = predictions.withColumn(columnValue, exp($"prediction_log"))
    val statsMap = buildStatsMaps(c, predictionResult(columnValue), label, predictionResult, columnValue)
    
  
    val bestModel = model.bestModel.asInstanceOf[PipelineModel].stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println(s"maxDepth: ${bestModel.getMaxDepth}, maxBins: ${bestModel.getMaxBins}")
    (bestModel, c, statsMap)
  }

  /* ... new cell ... */

  (trainedModels(0)._1).toDebugString
    

  /* ... new cell ... */

  trainedModels.foreach{ case (m, c, statsMap) => 
                        m.write.overwrite.save(ROOT + "/trained-models-dataframe/success-c" + c)
                        println(statsMap)}

  /* ... new cell ... */


}
                  
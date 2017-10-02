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

val ROOT = "/Users/flavio.clesio/Documents/spark-notebook-production/spark-notebook-0.7.0/notebooks/spark-summit-2017"

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






// function to extract the week of month from a timestamp
val udfWoM = (dt :Date) => {
   val cal = Calendar.getInstance()
   cal.setTime(dt)
  
   cal.setMinimalDaysInFirstWeek(1)
   cal.get(Calendar.WEEK_OF_MONTH).toDouble
}

// user defined function
val weekOfMonthUDF = udf(udfWoM)






// apply UDF to extract the week of month from date time field
val rawData = df.withColumn("week_of_month", weekOfMonthUDF($"datepart"))

rawData.show(20)




rawData.groupBy("week_of_month").count().show()


val avgDataFrame = rawData.groupBy("carrier_id", "hour_of_day","week_of_month").agg(avg("avg_response_time").as("avg_response_time"), 
                                                                                        avg("successful_charges").as("successful_charges"), 
                                                                                        avg("no_credit").as("no_credit"), 
                                                                                        avg("errors").as("errors"),
                                                                                        avg("total_attempts").as("total_attempts"))



avgDataFrame.show(10)



rawData.where("carrier_id = 4 AND hour_of_day = 19 and WEEk_of_month = 5").show()


avgDataFrame.where("carrier_id = 4 AND hour_of_day = 19 and WEEk_of_month = 5").show()



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


val summarizedDataFrame = summarizedList.reduce(_ union _)


//write data to disk in parquet format
summarizedDataFrame.write.format("parquet").mode("overwrite").save(ROOT + "/sbs-summarized-dataframe")



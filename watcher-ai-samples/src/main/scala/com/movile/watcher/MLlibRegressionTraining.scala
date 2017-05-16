package com.movile.watcher

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}


/**
  * J.P. Eiti Kimura (eiti.kimura@movile.com)
  * 05/12/16.
  */
object MLlibRegressionTraining {

  //val ROOT_DIR = "<YOUR DIRECTORY HERE>/spark-notebook-resources"
  val ROOT_DIR = "/spark-notebook-resources"
  System.setProperty("hadoop.home.dir", "/")

  val conf = new SparkConf().setAppName("watcher-ai").setMaster("local[*]")
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {

    // setting apache spark log to off
    sc.setLogLevel("OFF")

    // reading pre processed dataset
    val rdd = sc.objectFile[List[Double]]("/Users/eiti/git-repository/watcher-ai-samples/spark-notebook-resources/rdd-processed")
    rdd.count();

    // building the LabelPoint, using success as Label
//    val labelSet = rdd.map { l =>
//      val label = buildLabelValue(l)
//      val features = buildFeatures(l)
//      LabeledPoint(label, Vectors.dense(features.toArray))
//    }
//
//
//    //Split data into training  and test.
//    val splits = labelSet.randomSplit(Array(0.70, 0.30), seed = 13L)
//    val training = splits(0)
//    val test = splits(1)

    //val training = labelSet
    //val test = training
    //println("Training set Label sample:" + training.take(1).mkString(""))

    //finalize the context
    sc.stop();
  }

  def buildLabelValue(list: List[Double]) : Double = {
    // index = 4 is the number of success of the hour, that is what we want to predict
    return if (list(4) != 0.0) Math.log(list(4)) else 0.0
  }

  def buildFeatures(list: List[Double]) : List[Double] = {
    // remove the index 4, which means the number of success
    return list.patch(4, Nil, 1)
  }
}

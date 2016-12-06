package com.movile.watcher

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}


/**
  * J.P. Eiti Kimura (eiti.kimura@movile.com)
  * 05/12/16.
  */
object MLlibRegressionTraining {

  //val ROOT_DIR = "<YOUR DIRECTORY HERE>/spark-notebook-resources"
  val ROOT_DIR = "/spark-notebook-resources"
  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    // initiate the spark context
    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("watcher-ai")
    val sc = new SparkContext(conf)

    // reading pre processed dataset
    val rawData = sc.textFile(getClass.getResource("/datasets-sample.csv").getPath)
      .map(line => line.split(","))
      .map(array => array.map(x => x.toDouble))

    println("records read:" + rawData.count())

    //building the LabelPoint, using success as Label
    val labelSet = rawData.map { l =>
      val label = buildLabelValue(l)
      val features = buildFeatures(l)
      LabeledPoint(label, Vectors.dense(features.toArray))
    }

    println("records pre-processed:" + labelSet.count())
    labelSet.take(10).foreach(println)

    //Split data into training  and test.
    val splits = labelSet.randomSplit(Array(0.70, 0.30), seed = 13L)
    val training = splits(0)
    val test = splits(1)

    // training the model with different kind of algorithms
    val mapDecTreeModel = buildDecTreeModelMap(training)
    val mapTraining = normTrainingSet(training)
    val mapTest = normTrainingSet(test)
    val mapSGDModel = buildSGDModelMap(mapTraining)
    val mapLassoSGDModel = buildLassoSGDModelMap(mapTraining)
    val mapRidgeRegressionSGDModel = buildRidgeRegressionSGDModelMap(mapTraining)


    // Evaluate model using unseen data (test dataset)
    val computedSet = mapTest.map { case(idx, dataset) =>
      dataset.map{point =>
        val model = mapSGDModel(idx)
        val prediction = model.predict(point.features)
        (point.label, prediction)
      }.collect
    }

    println("== Linear Model with SGD ==")
    val predictionsSGD = sc.parallelize(computedSet.reduce(_++_))
    printStats(predictionsSGD)
    val computedSetLasso = mapTest.map { case(idx, dataset) =>
      dataset.map{point =>
        val model = mapLassoSGDModel(idx)
        val prediction = model.predict(point.features)
        (point.label, prediction)
      }.collect
    }

    println("== Lasso with SGD Model ==")
    val predictionsLassoSGD = sc.parallelize(computedSetLasso.reduce(_++_))
    printStats(predictionsLassoSGD)
    val computedSetRidge = mapTest.map { case(idx, dataset) =>
      dataset.map{point =>
        val model = mapRidgeRegressionSGDModel(idx)
        val prediction = model.predict(point.features)
        (point.label, prediction)
      }.collect
    }

    println("== Ridge Regression with SGD Model ==")
    val predictionsRidgeSGD = sc.parallelize(computedSetRidge.reduce(_++_))
    printStats(predictionsRidgeSGD)
    val labelsAndPredictions = test.map { point =>
      val carrier = point.features.apply(0)
      val model = mapDecTreeModel(carrier)
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }


    println("== Decision Tree Model ==")
    printStats(labelsAndPredictions)

    //finalize the context
    sc.stop();
  }

  def buildLabelValue(list: Array[Double]) : Double = {
    // index = 4 is the number of success of the hour, that is what we want to predict
    return if (list(4) != 0.0) Math.log(list(4)) else 0.0
  }

  def buildFeatures(list: Array[Double]) : Array[Double] = {
    // remove the index 4, which means the number of success
    return list.patch(4, Nil, 1)
  }

  // normalizer function
  def normTrainingSet(rdd:RDD[LabeledPoint]) : scala.collection.immutable.Map[Double, RDD[LabeledPoint]] = {

    // StandardScaler for data normalization
    val scaler = new StandardScaler(withMean = false, withStd = true)
      .fit(rdd.map(x => x.features))

    // split data by carrier id
    val range = List(1.0, 2.0, 4.0, 5.0)

    // return (Double, Some(RDD))
    return range.map{idx =>
      val trainingSet = rdd.filter(l => l.features.apply(0) == idx)
        .map(x => LabeledPoint(x.label, scaler.transform(x.features)))
      (idx, trainingSet)
    }.toMap
  }


  // build and filter the training set map
  def buildTrainingSet(rdd:RDD[LabeledPoint]) : scala.collection.immutable.Map[Double, RDD[LabeledPoint]] = {

    // split data by carrier id
    val range = List(1.0, 2.0, 4.0, 5.0)

    // return (Double, Some(RDD))
    return range.map{idx =>
      val trainingSet = rdd.filter(l => l.features.apply(0) == idx)
      (idx, trainingSet)
    }.toMap
  }


  // build and train model by carrier id
  def buildSGDModelMap(rdd:scala.collection.immutable.Map[Double, RDD[LabeledPoint]]) : scala.collection.immutable.Map[Double, LinearRegressionModel] = {
    val range = List(1.0, 2.0, 4.0, 5.0)
    // return (Double, Some(RDD))
    return range.map{idx =>
      // Building the model
      val numIterations = 100
      val regression = new LinearRegressionWithSGD().setIntercept(true)
      regression.optimizer.setStepSize(0.1)
      regression.optimizer.setNumIterations(numIterations)

      // get dataset
      val dataset = rdd.get(idx).orNull;
      if (dataset == null) println("ERROR: data set is null for carrier:" + idx)
      (idx, regression.run(dataset))
    }.toMap
  }


  // build and train model by carrier id
  def buildRidgeRegressionSGDModelMap(rdd:scala.collection.immutable.Map[Double, RDD[LabeledPoint]]) : scala.collection.immutable.Map[Double, RidgeRegressionModel] = {
    val range = List(1.0, 2.0, 4.0, 5.0)
    // return (Double, Some(RDD))
    return range.map{idx =>
      // Building the model
      val numIterations = 100
      val regression = new RidgeRegressionWithSGD().setIntercept(true)
      regression.optimizer.setStepSize(0.1)
      regression.optimizer.setNumIterations(numIterations)

      // get dataset
      val dataset = rdd.get(idx).orNull;
      (idx, regression.run(dataset))
    }.toMap
  }


  // build and train model by carrier id
  def buildLassoSGDModelMap(rdd:scala.collection.immutable.Map[Double, RDD[LabeledPoint]]) : scala.collection.immutable.Map[Double, LassoModel] = {
    val range = List(1.0, 2.0, 4.0, 5.0)
    // return (Double, Some(RDD))
    return range.map{idx =>
      // Building the model
      val numIterations = 100
      val regression = new LassoWithSGD().setIntercept(true)
      regression.optimizer.setStepSize(0.1)
      regression.optimizer.setNumIterations(numIterations)

      // get dataset
      val dataset = rdd.get(idx).orNull;
      (idx, regression.run(dataset))
    }.toMap
  }


  // build and train model by carrier id
  def buildDecTreeModelMap(rddx:RDD[LabeledPoint]) : scala.collection.immutable.Map[Double, DecisionTreeModel] = {
    val range = List(1.0, 2.0, 4.0, 5.0)

    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "variance"
    val maxDepth = 7
    val maxBins = 32

    return range.map{idx =>
      val filteredSet = rddx.filter(l =>  l.features.apply(0) == idx)
      // building the model
      val model = DecisionTree.trainRegressor(filteredSet, categoricalFeaturesInfo, impurity, maxDepth, maxBins);
      (idx, model)
    }.toMap
  }


  // prints result of algorithm tested
  def printStats(computedRdd:RDD[(Double, Double)]) = {
    val dtTotalCorrect = computedRdd.map{ case (v, p) =>
      val error = (math.exp(v) - math.exp(p))/math.exp(v);
      if (error > 0.35) 0 else 1;
    }.sum()

    val dtAccuracy = dtTotalCorrect / computedRdd.count
    val MeanSquaredError = computedRdd.map{ case (v, p) => math.pow(v - p, 2) }.mean()
    val RootMeanSquaredError = math.sqrt(MeanSquaredError)

    println("Model Accuracy (ACC) = " + dtAccuracy)
    println("Mean Squared Error (MSE) = " + MeanSquaredError)
    println("Root Mean Squared Error (RMSE) = " + RootMeanSquaredError)
  }
}

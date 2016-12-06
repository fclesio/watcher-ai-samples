import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LassoModel
import org.apache.spark.mllib.regression.LassoWithSGD
import org.apache.spark.mllib.regression.RidgeRegressionModel
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD


val ROOT_DIR = "/Users/flavio.clesio/Desktop/spark-notebook-0.6.3-scala-2.10.5-spark-1.6.0-hadoop-2.6.0-with-hive-with-parquet/notebooks/spark-mllib-sample/spark-notebook-resources"


def buildLabelValue(list: List[Double]) : Double = {
  // index = 4 is the number of success of the hour, that is what we want to predict
  return if (list(4) != 0.0) Math.log(list(4)) else 0.0
}

def buildFeatures(list: List[Double]) : List[Double] = {
   // remove the index 4, which means the number of success
   return list.patch(4, Nil, 1)
}

// reading pre processed dataset
val rdd = sc.objectFile[List[Double]](ROOT_DIR +"/rdd-processed")

// building the LabelPoint, using success as Label
val labelSet = rdd.map{l => val label = buildLabelValue(l)
                            val features = buildFeatures(l)
                            LabeledPoint(label, Vectors.dense(features.toArray))}


//Split data into training  and test.
val splits = labelSet.randomSplit(Array(0.70, 0.30), seed = 13L)
val training = splits(0)
val test = splits(1)

//val training = labelSet
//val test = training
println("Training set Label sample:" + training.take(1).mkString(""))

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
                    var regression = new LinearRegressionWithSGD().setIntercept(true)
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
                    var regression = new RidgeRegressionWithSGD().setIntercept(true)
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
                    var regression = new LassoWithSGD().setIntercept(true)
                    regression.optimizer.setStepSize(0.1)
                    regression.optimizer.setNumIterations(numIterations)
                     
                    // get dataset
                    val dataset = rdd.get(idx).orNull; 
                    (idx, regression.run(dataset))
                  }.toMap
}


// build and train model by carrier id
def buildDecTreeModelMap(rdd:RDD[LabeledPoint]) : scala.collection.immutable.Map[Double, DecisionTreeModel] = {
  val range = List(1.0, 2.0, 4.0, 5.0)
  
  val categoricalFeaturesInfo = Map[Int, Int]()
  val impurity = "variance"
  val maxDepth = 7
  val maxBins = 32
  
  // return (Double, Some(RDD))
  return range.map{idx => 
                   val filteredSet = training.filter(l => l.features.apply(0) == idx)
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
    val MeanAbsoluteError = computedRdd.map{ case (v, p) => math.abs(v - p)/p }.mean()

    println("Model Accuracy (ACC) = " + dtAccuracy)
    println("Mean Squared Error (MSE) = " + MeanSquaredError)
    println("Root Mean Squared Error (RMSE) = " + RootMeanSquaredError)
    println("Mean Absolute Error (MAE) = " + MeanAbsoluteError)
}


val mapDecTreeModel = buildDecTreeModelMap(training)
val mapTraining = normTrainingSet(training)
val mapTest = normTrainingSet(test)
val mapSGDModel = buildSGDModelMap(mapTraining)
val mapLassoSGDModel = buildLassoSGDModelMap(mapTraining)
val mapRidgeRegressionSGDModel = buildRidgeRegressionSGDModelMap(mapTraining)


val range = List(1.0, 2.0, 4.0, 5.0)
range.map{idx =>
   println("Linear Model with SGD - carrier:" + idx.toInt + ", weights:" + mapSGDModel(idx).weights)
}

range.map{idx =>
   println("Lasso with SGD Model - carrier:" + idx.toInt + ", weights:" + mapLassoSGDModel(idx).weights)
}

range.map{idx =>
   println("Ridge Regression with SGD Model - carrier:" + idx.toInt + ", weights:" + mapRidgeRegressionSGDModel(idx).weights)
}


println("features: " + labelSet.filter(l => l.features.apply(0) == 1.0).take(1)(0))
println("normalized features: " + mapTraining(1.0).take(1)(0))

// persisting mapDecTreeModel models to disk
mapDecTreeModel.foreach{case (carrier, model) => 
                          val modelName = "c" + carrier.toInt + "model"
                          model.save(sc, ROOT_DIR + "/trained-models/" + modelName)
                       }


// Evaluate model on test dataset
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



// Evaluate model on test dataset
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


// Evaluate model on test dataset
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


// Evaluate model on test instances and compute test error
val labelsAndPredictions = test.map { point =>
  val carrier = point.features.apply(0)
  val model = mapDecTreeModel(carrier)
  val prediction = model.predict(point.features)
  (point.label, prediction)
}


println("== Decision Tree Model ==")
printStats(labelsAndPredictions)


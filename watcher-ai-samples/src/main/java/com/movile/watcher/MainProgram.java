package com.movile.watcher;

import com.github.sheigutn.pushbullet.Pushbullet;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * J.P. Eiti Kimura (eiti.kimura@movile.com)
 * 01/07/16.
 */
public class MainProgram {

    private static String ROOT = "generated-model";

    private static double ERROR_THRESHOLD = 45.0;

    public static void main(String[] args) throws Exception {
        // initializing a local spark context (embedded spark instance)
        SparkApplicationContext sparkContext = new SparkApplicationContext("sample-models");

        // load trained model
        Path path = Paths.get(ROOT + "/trained-model/c1model-succ");
        DecisionTreeModel model =  sparkContext.loadModel(path);

        // read data from a file, database or any other source
        List<double[]> rawData = readData();

        // prepare dataset
        List<LabeledPoint> dataset = createDataset(rawData);

        dataset.forEach(labeledPoint -> {
            int predicted = (int) Math.exp(model.predict(labeledPoint.features()));
            int measured = (int) Math.exp(labeledPoint.label());
            double error = ((double)(predicted - measured) / (double) measured) * 100.0;
            log("features: " + format(labeledPoint) + ", measured: " + measured + ", predicted: " + predicted + ", error: " + error);

            if (error > ERROR_THRESHOLD) {
                alert("[ALERT] high error on success " + error + ", check for problems: " + labeledPoint.features());
            }
        });

        // closing spark context
        sparkContext.close();
        log("finished :)");
    }

    /**
     * create a LabelSet dataset
     * @param rawData a list of double arrays
     * @return a List of LabelPoints
     */
    private static List<LabeledPoint> createDataset(List<double[]> rawData) {
        return rawData.stream()
                .map(array -> {
                    // get success element
                    double label = Math.log(array[4]);

                    //remove it from dataset
                    double[] features = ArrayUtils.remove(array, 4);
                    return new LabeledPoint(label, Vectors.dense(features));

                })
                .collect(Collectors.toList());
    }

    /**
     * read sample file
     * @return a double[] array
     * @throws IOException
     */
    private static List<double[]> readData() throws IOException {

        // parse csv file and readsamples as double arrays
        List<double[]> dataSet = Files.readAllLines(Paths.get(ROOT + "/samples.txt")).stream()
                .map(l -> Arrays.asList(l.split(",")))
                .map(lst -> lst.stream().mapToDouble(Double::parseDouble).toArray())
                .collect(Collectors.toList());

        return dataSet;
    }

    /**
     * load up the trained model using spark context
     * @param sparkContext the spark context
     */
    private static DecisionTreeModel loadModel(Path path, SparkApplicationContext sparkContext) {
        DecisionTreeModel model =  sparkContext.loadModel(path);
        return model;
    }

    /**
     * format a label point features
     * @param labeledPoint the label point object
     * @return a String with a log friendly features format
     */
    private static String format(LabeledPoint labeledPoint) {
        double [] array = labeledPoint.features().toArray();

        StringBuilder sb = new StringBuilder();
        sb.append("[");
        sb.append(" carrier_id: ").append((int)array[0]);
        sb.append(", hour: ").append((int)array[1]);
        sb.append(", week_of_month: ").append((int)array[2]);
        sb.append(", response_time: ").append((int)array[3]).append("ms");
        sb.append(", no_credit: ").append((int)array[4]);
        sb.append(", errors: ").append((int)array[5]);
        sb.append(", attempts: ").append((int)array[6]);
        sb.append("]");

        return sb.toString();
    }

    public static void alert(String message) {
        log(message);
        Pushbullet pushBullet = new Pushbullet("o.LU90hHNKfKbhe5IPFBbAS41GU6vQJltq");
        pushBullet.pushNote("[watcher-ai] check the platform", message);
    }

    /**
     * just print data to console
     * @param line data to be printed
     */
    private static void log(String line) {
        System.out.println(line);
    }
}

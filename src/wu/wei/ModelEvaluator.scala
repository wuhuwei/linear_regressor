package wu.wei

import BIDMat.{ Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat }
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import scala.collection.mutable.HashMap
import scala.util.control.Breaks._
import java.io._

object ModelEvaluator {
  val BLOCK_SIZE = 100000
  val BLOCK_SIZE_FILE = 1000
  val FEATURE_THRESHOLD = 100000

  def main(args: Array[String]): Unit = {
    var residual_errors: List[Double] = Nil
    var positive_precisions: List[Double] = Nil
    var positive_recalls: List[Double] = Nil
    var positive_f1s: List[Double] = Nil
    var negative_precisions: List[Double] = Nil
    var negative_recalls: List[Double] = Nil
    var negative_f1s: List[Double] = Nil
    var positive_roc_aucs: List[Double] = Nil
    var positive_lifts: List[Double] = Nil
    var negative_roc_aucs: List[Double] = Nil
    var negative_lifts: List[Double] = Nil
    var total_actual_positives: Int = 0
    var total_actual_negatives: Int = 0
    var total_positive_misclassified: Int = 0
    var total_negative_misclassified: Int = 0

    val blocks = LoadBlocks

    for (block_num <- 0 to blocks.length - 1) {
      val regressionParameters: FMat = load("trained_model_" + block_num, "regressionParameters")

      val indexToWordDict = parseDict(args(0), FEATURE_THRESHOLD)._1
      val weight_map = HashMap[String, Float]()
      for (i <- 0 to regressionParameters.nrows - 1) {
        weight_map.put(indexToWordDict.getOrElse(i, ""), regressionParameters(i, 0))
      }
      var limit = 100
      var count = 0
      breakable {
        weight_map.toList sortBy (-_._2) foreach {
          case (term, weight) =>
            println(term + " " + weight)
            count += 1
            if (count >= limit) { break }
        }
      }
      println()
      count = 0
      breakable {
        weight_map.toList sortBy (_._2) foreach {
          case (term, weight) =>
            println(term + " " + weight)
            count += 1
            if (count >= limit) { break }
        }
      }
      println()

      var (test_counts, test_counts_t, test_ratingsMat) = blocks(block_num)
      val test_predictions = test_counts * regressionParameters
      val test_residuals = test_ratingsMat - test_predictions
      val test_residualError = (test_residuals.t * test_residuals)(0, 0)
      residual_errors = test_residualError :: residual_errors

      val one_vec = ones(test_ratingsMat.ncols, test_ratingsMat.nrows)
      val test_ratings_binary = test_ratingsMat > zeros(test_ratingsMat.nrows, test_ratingsMat.ncols)
      val test_predictions_binary = test_predictions > zeros(test_predictions.nrows, test_predictions.ncols)

      val actual_negatives = ((one_vec * (test_ratings_binary == 0.0))(0, 0)).toInt
      val actual_positives = ((one_vec * test_ratings_binary)(0, 0)).toInt
      total_actual_negatives += actual_negatives
      total_actual_positives += actual_positives

      var true_positives = (one_vec * ((test_predictions_binary + test_ratings_binary) == 2.0))(0, 0).toInt
      var false_positives = (one_vec * ((test_predictions_binary + (test_ratings_binary == 0.0)) == 2.0))(0, 0).toInt
      var true_negatives = (one_vec * (((test_predictions_binary == 0) + (test_ratings_binary == 0.0)) == 2.0))(0, 0).toInt
      var false_negatives = (one_vec * (((test_predictions_binary == 0) + test_ratings_binary) == 2.0))(0, 0).toInt

      total_positive_misclassified += false_negatives
      total_negative_misclassified += false_positives

      println(true_positives + "," + false_positives + "," + true_negatives + "," + false_negatives)
      var precision = true_positives.toDouble / (true_positives + false_positives)
      var recall = true_positives.toDouble / (true_positives + false_negatives)
      var f1 = 2.0 * precision * recall / (precision + recall)
      println(precision + "," + recall + "," + f1)
      positive_precisions = precision :: positive_precisions
      positive_recalls = recall :: positive_recalls
      positive_f1s = f1 :: positive_f1s
      precision = true_negatives.toDouble / (true_negatives + false_negatives)
      recall = true_negatives.toDouble / (true_negatives + false_positives)
      f1 = 2.0 * precision * recall / (precision + recall)
      println(precision + "," + recall + "," + f1)
      negative_precisions = precision :: negative_precisions
      negative_recalls = recall :: negative_recalls
      negative_f1s = f1 :: negative_f1s

      val test_predictions_map = HashMap[Int, Float]()
      for (i <- 0 to test_predictions.nrows - 1) {
        test_predictions_map.put(i, test_predictions(i, 0))
      }

      // Get ROC curve for query for positive reviews
      var specificity = 0.0
      var last_specificity = 0.0
      recall = 0.0
      var last_recall = 0.0

      true_positives = 0
      true_negatives = actual_negatives
      false_positives = 0
      false_negatives = actual_positives

      var rocStream = new BufferedWriter(new FileWriter("roc." + block_num + ".txt"))
      var liftStream = new BufferedWriter(new FileWriter("lift." + block_num + ".txt"))

      var area_under_curve = 0.0

      rocStream.write("{" + "{" + specificity + "," + recall + "},")
      liftStream.write("{")
      var found_lift = false
      test_predictions_map.toList sortBy (-_._2) foreach {
        case (i, _) =>
          if (test_ratings_binary(i, 0) != 0) {
            false_negatives -= 1
            true_positives += 1
          } else {
            true_negatives -= 1
            false_positives += 1
          }
          specificity = 1.0 - true_negatives.toDouble / (false_positives + true_negatives)
          recall = true_positives.toDouble / (true_positives + false_negatives)
          rocStream.write("{" + specificity + "," + recall + "},")
          if (specificity > 0.0) {
            liftStream.write("{" + specificity + "," + recall / specificity + "},")
          }

          area_under_curve += (specificity - last_specificity) * (last_recall + recall) / 2.0
          if (specificity > 0.01 && !found_lift) {
            var lift = recall / specificity
            println("Positive query: Lift: " + lift)
            positive_lifts = lift :: positive_lifts
            found_lift = true
          }
          last_specificity = specificity
          last_recall = recall
      }
      println("Positive query: Area under ROC curve: " + area_under_curve)
      positive_roc_aucs = area_under_curve :: positive_roc_aucs
      rocStream.write("}")
      liftStream.write("}")
      rocStream.close()
      liftStream.close()
    }
    println("Total actual negatives: " + total_actual_negatives)
    println("Total misclassified negatives: " + total_negative_misclassified)
    println("Total actual positives: " + total_actual_positives)
    println("Total misclassified positives: " + total_positive_misclassified)

    println("Test set residual error: " + mean(residual_errors) + " (stddev " + stddev(residual_errors) + ")")
    println("query: predicted rating > 3:")
    println("  precision: " + mean(positive_precisions) + " (stddev " + stddev(positive_precisions) + ")")
    println("  recall: " + mean(positive_recalls) + " (stddev " + stddev(positive_recalls) + ")")
    println("  f1: " + mean(positive_f1s) + " (stddev " + stddev(positive_f1s) + ")")
    println("  Area under ROC curve: " + mean(positive_roc_aucs) + " (stddev " + stddev(positive_roc_aucs) + ")")
    println("  1% lift: " + mean(positive_lifts) + " (stddev " + stddev(positive_lifts) + ")")
    println("query: predicted rating < 3:")
    println("  precision: " + mean(negative_precisions) + " (stddev " + stddev(negative_precisions) + ")")
    println("  recall: " + mean(negative_recalls) + " (stddev " + stddev(negative_recalls) + ")")
    println("  f1: " + mean(negative_f1s) + " (stddev " + stddev(negative_f1s) + ")")
    println("  Area under ROC curve: " + mean(negative_roc_aucs) + " (stddev " + stddev(negative_roc_aucs) + ")")
    println("  1% lift: " + mean(negative_lifts) + " (stddev " + stddev(negative_lifts) + ")")
  }

  /* parse the most common featureThreshold number of words in the dictionary,
   * or all the words in the dictionary if featureThreshold == -1 */
  def parseDict(dictPath: String, featureThreshold: Int): (HashMap[Int, String], HashMap[String, Int]) = {
    val indexToWordDict: HashMap[Int, String] = new HashMap[Int, String]()
    val wordToIndexDict: HashMap[String, Int] = new HashMap[String, Int]()
    try {
      var br = new BufferedReader(new FileReader(dictPath))
      var sCurrentLine: String = br.readLine().trim().replaceAll("\\s+", "a")
      var numRead = 0
      while (sCurrentLine != null && sCurrentLine != "" && (featureThreshold == -1 || numRead < featureThreshold)) {
        var splitLine: Array[String] = sCurrentLine.split("\t")
        if (splitLine.size == 4) { // Don't bother handling tab
          //if (splitLine(3).startsWith("<") && splitLine(3).endsWith(">")) {
          indexToWordDict.put(Integer.parseInt(splitLine(1)), splitLine(3))
          wordToIndexDict.put(splitLine(3), Integer.parseInt(splitLine(1)))
          numRead += 1
          //}
        }
        sCurrentLine = br.readLine();

      }
    }
    (indexToWordDict, wordToIndexDict)
  }

  private def LoadBlock(block_num: Int): (BIDMat.SMat, BIDMat.SMat, BIDMat.FMat) = {
    var counts: SMat = null
    var counts_t: SMat = null
    var ratingsMat: FMat = null
    breakable {
      for (i <- block_num * BLOCK_SIZE / BLOCK_SIZE_FILE + 1 to (block_num + 1) * BLOCK_SIZE / BLOCK_SIZE_FILE) {
        var f = new File(BLOCK_SIZE_FILE + "/" + i + ".mat")
        if (!f.exists()) break
        var counts_file: SMat = load(BLOCK_SIZE_FILE + "/" + i + ".mat", "counts")
        var counts_t_file: SMat = load(BLOCK_SIZE_FILE + "/" + i + ".mat", "counts_t")
        val ratingsMat_file: FMat = load(BLOCK_SIZE_FILE + "/" + i + ".mat", "ratings")
        counts_file = counts_file(?, 0 to FEATURE_THRESHOLD - 1)
        counts_t_file = counts_t_file(0 to FEATURE_THRESHOLD - 1, ?)
        if (counts == null) { counts = counts_file } else { counts = counts on counts_file }
        if (counts_t == null) { counts_t = counts_t_file } else { counts_t = counts_t \ counts_t_file }
        if (ratingsMat == null) { ratingsMat = ratingsMat_file } else { ratingsMat = ratingsMat on ratingsMat_file }
      }
    }
    return (counts, counts_t, ratingsMat)
  }

  private def LoadBlocks: List[(BIDMat.SMat, BIDMat.SMat, BIDMat.FMat)] = {
    var block_num = 0
    var blocks = List[(SMat, SMat, FMat)]()
    breakable {
      while (true) {
        println("Loading block " + block_num)
        val block = LoadBlock(block_num)
        if (block._1 == null) { break }
        blocks = block :: blocks
        block_num += 1
      }
    }
    blocks = blocks.reverse
    blocks
  }

  // From http://parhelium.pl/blog/2010/01/30/calculator-class-add-numbers-and-calculate-average-and-standard-deviation/
  private def mean(values: List[Double]): Double = values.reduceLeft(_ + _) / values.length
  private def stddev(values: List[Double]): Double = {
    val sum: Double =
      if (values.length >= 2) {
        val mu = mean(values)
        val factor: Double = 1.0 / (values.length.toDouble - 1);
        factor * values.foldLeft(0.0) { (acc, x) => acc + math.pow(x - mu, 2) }
      } else {
        0.0
      }
    math.sqrt(sum)
  }
  
}
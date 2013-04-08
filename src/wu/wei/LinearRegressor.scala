package wu.wei

import BIDMat.{ Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat }
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._
import java.io._
import java.util.StringTokenizer
import ncsa.hdf.hdf5lib.exceptions.HDF5LibraryException

object LinearRegressor {
  val endReviewDelimiter: String = "</review>"
  val startReviewTextDelimiter: String = "<review_text>"
  val endReviewTextDelimiter: String = "</review_text>"
  val startRatingDelimiter: String = "<rating>"
  val endRatingDelimiter: String = "</rating>"

  val BLOCK_SIZE = 100000
  val BLOCK_SIZE_FILE = 1000
  val FEATURE_THRESHOLD = 100000

  def main(args: Array[String]): Unit = {
    var test_block_num = args(0).toInt
    println("Test block is " + test_block_num + " (doing 70 iterations)")
    println("Feature threshold: " + FEATURE_THRESHOLD)

    var regressionParameters: FMat = null
    var gradient_sum_squared: FMat = null

    var trained_model_file = new File("trained_model_" + test_block_num)
    if (trained_model_file.exists()) {
      regressionParameters = load("trained_model_" + test_block_num, "regressionParameters")
      gradient_sum_squared = load("trained_model_" + test_block_num, "gradient_sum_squared")
    }
    val blocks = LoadBlocks
    println("Loading test block (block " + test_block_num + ")")
    var (test_counts, test_counts_t, test_ratingsMat) = blocks(test_block_num)

    var (counts, counts_t, ratingsMat) = blocks(0)
    for (block_num <- 1 to blocks.length - 1) {
      if (block_num != test_block_num) {
        counts = counts on blocks(block_num)._1
        counts_t = counts_t \ blocks(block_num)._2
        ratingsMat = ratingsMat on blocks(block_num)._3
      }
    }

    if (!trained_model_file.exists()) {
      regressionParameters = zeros(test_counts.ncols, 1)
      gradient_sum_squared = zeros(test_counts.ncols, 1)
    }

    val stepSize = 2.0f
    var lambda = 0.001f // Regularization parameter

    for (k <- 1 to 70) yield {
      println("\nTraining, iteration k=" + k + ", " + new java.util.Date)
      println("Taking 100 gradient steps, " + new java.util.Date)
      var gradient: FMat = zeros(test_counts.ncols, 1)
      flip
      breakable {
        for (j <- 1 to 100) {
          // ADAGRAD update
          gradient ~ counts_t * ((counts * regressionParameters) - ratingsMat) + lambda *@ regressionParameters
          gradient_sum_squared += gradient *@ gradient
          var stepSizes = zeros(regressionParameters.nrows, 1)
          for (i <- 0 to regressionParameters.nrows - 1) {
            if (gradient_sum_squared(i) > 0.0f) {
              stepSizes(i) = stepSize / math.sqrt(gradient_sum_squared(i))
            }
          }
          regressionParameters -= stepSizes *@ gradient
        }
      }
      val gflops = gflop
      println("Gflops over 100 iterations: " + gflops._1)

      {
        println("magnitude of gradient: " + (gradient.t * gradient)(0, 0))
        val residuals = ratingsMat - counts * regressionParameters
        val residualError = (residuals.t * residuals)(0,0)
        println ("residual error on last training matrix: " + residualError)
        println ("objective function: " + (residualError/2 + lambda/2 * ((regressionParameters.t * regressionParameters)(0,0))))
        println("regularization value: " + (lambda * regressionParameters.t * regressionParameters)(0, 0))

        val test_predictions = test_counts * regressionParameters
        val test_residuals = test_ratingsMat - test_predictions
        val test_residualError = (test_residuals.t * test_residuals)(0, 0)
        println("test matrix residual error: " + test_residualError)

        val one_vec = ones(test_ratingsMat.nrows, test_ratingsMat.ncols)
        val test_ratings_binary = test_ratingsMat > zeros(test_ratingsMat.nrows, test_ratingsMat.ncols)
        val test_predictions_binary = test_predictions > zeros(test_predictions.nrows, test_predictions.ncols)

        val true_positives = (one_vec.t * ((test_predictions_binary + test_ratings_binary) == 2.0))(0, 0)
        val false_positives = (one_vec.t * ((test_predictions_binary + (test_ratings_binary == 0.0)) == 2.0))(0, 0)
        val true_negatives = (one_vec.t * (((test_predictions_binary == 0) + (test_ratings_binary == 0.0)) == 2.0))(0, 0)
        val false_negatives = (one_vec.t * (((test_predictions_binary == 0) + test_ratings_binary) == 2.0))(0, 0)
        val precision = true_positives / (true_positives + false_positives)
        val recall = true_positives / (true_positives + false_negatives)
        print("precision: " + precision + ", ")
        print("recall: " + recall + ", ")
        println("f1: " + 2.0 * precision * recall / (precision + recall))
      }
      saveAs("trained_model_" + test_block_num, regressionParameters, "regressionParameters", gradient_sum_squared, "gradient_sum_squared")
    }
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
}

class LittleEndianDataInputStream(i: InputStream) extends DataInputStream(i) {
  def readLongLE(): Long = java.lang.Long.reverseBytes(super.readLong())
  def readIntLE(): Int = java.lang.Integer.reverseBytes(super.readInt())
  def readCharLE(): Char = java.lang.Character.reverseBytes(super.readChar())
  def readShortLE(): Short = java.lang.Short.reverseBytes(super.readShort())
}
package wu.wei

import BIDMat.{ Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat }
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import scala.collection.mutable.HashMap
import scala.util.control.Breaks._
import java.io.BufferedReader
import java.util.StringTokenizer
import java.io.FileReader
import java.io.File
import java.io.FileInputStream
import java.io.DataInputStream
import java.io.InputStream
import java.io.EOFException
import ncsa.hdf.hdf5lib.exceptions.HDF5LibraryException

object LinearRegressor {
  val endReviewDelimiter: String = "</review>"
  val startReviewTextDelimiter: String = "<review_text>"
  val endReviewTextDelimiter: String = "</review_text>"
  val startRatingDelimiter: String = "<rating>"
  val endRatingDelimiter: String = "</rating>"
    
  val BLOCK_SIZE = 1000


  def main(args: Array[String]): Unit = {
    var i = 0

    if (false) {
    parseTokens(args(1), parseTagDict(args(0))._1)

    // Find number of last matrix file
    i = 1
    breakable { while (true) {
		var f = new File(BLOCK_SIZE + "/" + i + ".mat")
		if(!f.exists()) break
		i += 1
    } }

    // Get number of columns in last matrix file - will reflect total number of columns
    val last_counts: SMat = load(BLOCK_SIZE + "/" + (i - 1) + ".mat", "counts")
    val max_cols = last_counts.ncols
    i = 1
    
    // Update count matrix in all other matrix files to have same number of columns
    breakable { while (true) {
		var f = new File(BLOCK_SIZE + "/" + i + ".mat")
		if(!f.exists()) break
		var counts: SMat = load(BLOCK_SIZE + "/" + i + ".mat", "counts")
		if (counts.ncols != max_cols) { 
			var zeros_append = sparse(icol(0, counts.nrows - 1), icol(max_cols - counts.ncols - 1, max_cols - counts.ncols - 1), col(0, 0))
			counts = counts \ zeros_append 
			var counts_t: SMat = load(BLOCK_SIZE + "/" + i + ".mat", "counts_t")
			counts_t = counts_t on zeros_append.t
			val ratingsMat: FMat = load(BLOCK_SIZE + "/" + i + ".mat", "ratings")
	        saveAs(BLOCK_SIZE + "/" + i + ".mat", counts, "counts", counts_t, "counts_t", ratingsMat, "ratings")
		}	
		i += 1
    } }
    }
    
    i = 1
    val test_data_file = 32
	var test_counts: SMat = load(BLOCK_SIZE + "/" + test_data_file + ".mat", "counts")
	val test_ratingsFMat: FMat = load(BLOCK_SIZE + "/" + test_data_file + ".mat", "ratings")
	val test_ratingsMat: FMat = test_ratingsFMat 
    
    val stepSize = 0.000001f
    var lambda = 0.1f // Regularization parameter
    val coeff2 = stepSize * lambda
    var regressionParameters: FMat = null

    for (k <- 1 to 10000) {
        i = 1
	    breakable { while (true) {
	    	if(i != test_data_file) {
		    	var f = new File(BLOCK_SIZE + "/" + i + ".mat")
		    	if(!f.exists()) break
		    	var counts: SMat = load(BLOCK_SIZE + "/" + i + ".mat", "counts") 
		    	var counts_t: SMat = load(BLOCK_SIZE + "/" + i + ".mat", "counts_t")
		    	val ratingsMat: FMat = load(BLOCK_SIZE + "/" + i + ".mat", "ratings")
		    	if (regressionParameters == null) {
		          regressionParameters = zeros(counts.ncols, 1)
		    	}
	    	    val old_residuals = ratingsMat - counts * regressionParameters
		    	val old_residualError = (old_residuals.t * old_residuals)(0,0)
	    	    val old_regressionParameters = regressionParameters
	    	    var gradient: FMat = null
		    	breakable { for (j <- 1 to 10) {
		          val residuals = ratingsMat - counts * regressionParameters
		          gradient = -2.0f *@ (counts_t * residuals ) + lambda * regressionParameters
			      regressionParameters -= stepSize *@ gradient
		    	} }
	    	    
		    	println("File " + i + ".mat, " + new java.util.Date)
		    	println("magnitude of gradient: " + (gradient.t * gradient)(0,0))
	    	    val residuals = ratingsMat - counts * regressionParameters
		    	val residualError = (residuals.t * residuals)(0,0)
		    	println ("residual error on last training matrix: " + residualError)
		    	if (residualError > 1.1*old_residualError) {
		    	  // Made a bad move, go back
		    	  regressionParameters = old_regressionParameters 
		    	  print("Bad move, returning to previous parameter vector")
		    	}
	    	    
		    	println("regularization value: " + (lambda * regressionParameters.t * regressionParameters)(0,0))
		    	
	    	    val test_predictions = test_counts * regressionParameters
				val test_residuals = test_ratingsMat - test_predictions
				val test_residualError = (test_residuals.t * test_residuals)(0,0)
		    	println("test matrix residual error: " + test_residualError)

	    	    val one_vec = ones(test_ratingsMat.nrows, test_ratingsMat.ncols)
				val test_ratings_binary = test_ratingsMat > zeros(test_ratingsMat.nrows, test_ratingsMat.ncols) 
	    	    val test_predictions_binary = test_predictions > zeros(test_predictions.nrows, test_predictions.ncols)

	    	    val true_positives = (one_vec.t * ((test_predictions_binary + test_ratings_binary) == 2.0))(0,0)
	    	    val false_positives = (one_vec.t * ((test_predictions_binary + (test_ratings_binary == 0.0)) == 2.0))(0,0)
	    	    val true_negatives = (one_vec.t * (((test_predictions_binary == 0) + (test_ratings_binary == 0.0)) == 2.0))(0,0)
	    	    val false_negatives = (one_vec.t * (((test_predictions_binary == 0) + test_ratings_binary) == 2.0))(0,0)
	    	    val precision = true_positives/(true_positives + false_positives)
	    	    val recall = true_positives/(true_positives + false_negatives)
	    	    print("precision: " + precision + ", ")
	    	    print("recall: " + recall + ", ")
	    	    println("f1: " + 2.0*precision*recall/(precision + recall))
	    	}
	    	i += 1
	    } }
	}
    
	i += 1
  }
  
  def parseDict(dictPath: String): (HashMap[Int, String], HashMap[String, Int]) = {
    val indexToWordDict: HashMap[Int, String] = new HashMap[Int, String]()
    val wordToIndexDict: HashMap[String, Int] = new HashMap[String, Int]()
    try {
      var br = new BufferedReader(new FileReader(dictPath))
      var sCurrentLine: String = br.readLine()
      var numRead = 0
      while (sCurrentLine != null && numRead < 50) {
        var splitLine: Array[String] = sCurrentLine.split("\t")
        if (splitLine(3).startsWith("<") && splitLine(3).endsWith(">")) {
          indexToWordDict.put(Integer.parseInt(splitLine(1)), splitLine(3))
          wordToIndexDict.put(splitLine(3), Integer.parseInt(splitLine(1)))

        }
        sCurrentLine = br.readLine();
        numRead += 1
      }
    }
    (indexToWordDict, wordToIndexDict)
  }

  def parseTagDict(dictPath: String): (HashMap[Int, String], HashMap[String, Int]) = {
    val indexToWordDict: HashMap[Int, String] = new HashMap[Int, String]()
    val wordToIndexDict: HashMap[String, Int] = new HashMap[String, Int]()
    try {
      var br = new BufferedReader(new FileReader(dictPath))
      var sCurrentLine: String = br.readLine()
      var numRead = 0
      while (sCurrentLine != null && sCurrentLine != "") { // && numRead < 50) {
        var splitLine: Array[String] = sCurrentLine.split("\t")
        if (splitLine.size == 4) { // Don't bother handling tab
	        //if (splitLine(3).startsWith("<") && splitLine(3).endsWith(">")) {
              indexToWordDict.put(Integer.parseInt(splitLine(1)), splitLine(3))
	          wordToIndexDict.put(splitLine(3), Integer.parseInt(splitLine(1)))
	        //}
        }
        sCurrentLine = br.readLine();
        numRead += 1
      }
    }
    (indexToWordDict, wordToIndexDict)
  }

  /**
   * takes a binary tokens file, and parses BLOCK_SIZE of them at a time, saving them into a
   * a sparse matrix "counts" where counts(i, j) = # of occurrences of word j in review i,
   * and a rating matrix "ratings" where ratings(i, 0) = rating for review i
   */
  def parseTokens(tokensPath: String, tagDict: HashMap[Int, String]) = {
    //println(tokensPath)
    var numDocs: Int = 0

    var rowInd: List[Int] = List[Int]()
    var colInd: List[Int] = List[Int]()
    var values: List[Int] = List[Int]()
    var ratings: HashMap[Int, Int] = new HashMap[Int, Int]()

    var numReviewsProcessed: Int = 0

    try {

      val inputFile: File = new File(tokensPath);
      val fileIn: FileInputStream = new FileInputStream(inputFile)
      val dataIn: LittleEndianDataInputStream = new LittleEndianDataInputStream(fileIn)

      var ratingFlag: Boolean = false
      var countFlag: Boolean = false
      var reviewRating = -1
      var reviewWordCounts: HashMap[Int, Int] = new HashMap[Int, Int]();

      new File(BLOCK_SIZE.toString()).mkdir();

      while (true) {
        dataIn.readIntLE()
        dataIn.readIntLE()
        val tokenNum = dataIn.readIntLE()
        val tokenStr = tagDict.getOrElse(tokenNum, "")
		if (tokenStr == startReviewTextDelimiter) {
		  countFlag = true
		} else if (tokenStr == endReviewTextDelimiter) {
		  countFlag = false
		} else if (tokenStr == startRatingDelimiter) {
		  ratingFlag = true
		} else if (tokenStr == endRatingDelimiter) {
		  ratingFlag = false
		} else if (tokenStr == endReviewDelimiter) {
		  rowInd = List.fill(reviewWordCounts.size)(numReviewsProcessed % BLOCK_SIZE) ::: rowInd
		  val wordCountsList: List[(Int, Int)] = reviewWordCounts.toList
		  colInd = wordCountsList.map(x => x._1) ::: colInd
		  values = wordCountsList.map(x => x._2) ::: values
		  ratings.put(numReviewsProcessed % BLOCK_SIZE, reviewRating)
		  numReviewsProcessed += 1
		
		  reviewWordCounts = new HashMap[Int, Int]()
		  reviewRating = -1
		
		  if (numReviewsProcessed % BLOCK_SIZE == 0) {
		    saveMatrixFile(rowInd, colInd, values, ratings.toList, numReviewsProcessed)
		
		    rowInd = List[Int]()
		    colInd = List[Int]()
		    values = List[Int]()
		    ratings = new HashMap[Int, Int]()
		  }
		
		} else if (countFlag == true) {
		  reviewWordCounts.put(tokenNum, reviewWordCounts.getOrElse(tokenNum, 0) + 1)
		} else if (ratingFlag == true) {
		  if (tokenStr != "0") {
		    reviewRating = tokenStr.toInt - 3 // Make 0 the center score for better use of floating-point range
		  }
		}
      }
    } catch {
      case eof: EOFException => {
        saveMatrixFile(rowInd, colInd, values, ratings.toList, numReviewsProcessed)
      }
      case e: HDF5LibraryException => println(e.getMessage() + "\n" + "Major error: " + e.getMajorErrorNumber() + ", minor error: " + e.getMinorError(e.getMinorErrorNumber()))
      case e: Exception => println("what" + e + " " + numReviewsProcessed)
    }

  }
  
  private def saveMatrixFile(rowInd: List[Int], colInd: List[Int], values: List[Int], ratingsList: List[(Int, Int)], numReviewsProcessed: Int): Unit = {
    var counts: SMat = sparse(icol(rowInd), icol(colInd), icol(values))
    counts = sparse(ones(counts.nrows, 1)) \ counts
    val counts_t: SMat = counts.t 
    val ratingsMat: FMat = full(sparse(icol(ratingsList.map(x => x._1)), icol(List.fill(ratingsList.length)(0)), icol(ratingsList.map(x => x._2))))
    val filename = (numReviewsProcessed + BLOCK_SIZE - 1) / BLOCK_SIZE + ".mat"
    println("saving block " + filename)
    println("size: " + counts.nrows + " " + ratingsList.length)
    saveAs(BLOCK_SIZE + "/" + filename, counts, "counts", counts_t, "counts_t", ratingsMat, "ratings")
  }
}

class LittleEndianDataInputStream(i: InputStream) extends DataInputStream(i) {
  def readLongLE(): Long = java.lang.Long.reverseBytes(super.readLong())
  def readIntLE(): Int = java.lang.Integer.reverseBytes(super.readInt())
  def readCharLE(): Char = java.lang.Character.reverseBytes(super.readChar())
  def readShortLE(): Short = java.lang.Short.reverseBytes(super.readShort())
}
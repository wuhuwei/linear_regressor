package wu.wei
import BIDMat.{ Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat }
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import scala.collection.mutable.HashMap
import java.io.FileInputStream
import ncsa.hdf.hdf5lib.exceptions.HDF5LibraryException
import java.io.File
import java.io.EOFException

object DataGenerator {
  val BLOCK_SIZE: Int = 1000
  val FEATURE_THRESHOLD = 200000
  val endReviewDelimiter: String = "</review>"
  val startReviewTextDelimiter: String = "<review_text>"
  val endReviewTextDelimiter: String = "</review_text>"
  val startRatingDelimiter: String = "<rating>"
  val endRatingDelimiter: String = "</rating>"

  def main(args: Array[String]): Unit = {
    var i = 0

    if (true) {
      val dict: (HashMap[Int, String], HashMap[String, Int]) = parseDict("tokenized.mat", "smap")
      val ngramInfo: (HashMap[String, Int], HashMap[Int, String]) = selectNgramFeatures(args(1), "ngrams.mat", dict._1, 3, FEATURE_THRESHOLD, 300)
      
      val gramDictMat: CSMat = load("ngrams.mat", "gramsDict")
      val gramInfo: (HashMap[String, Int], HashMap[Int, String]) = csMatDictToMap(gramDictMat)
      parseNgramTokens(args(1), dict._1, gramInfo._1, 2, "3grams_" + BLOCK_SIZE)
    }
  }

  def parseDict(dictPath: String, matName: String): (HashMap[Int, String], HashMap[String, Int]) = {
    val indexToWordDict: HashMap[Int, String] = new HashMap[Int, String]()
    val wordToIndexDict: HashMap[String, Int] = new HashMap[String, Int]()
    var dict: CSMat = load(dictPath, matName)
    for (i <- 0 until 100) {
      indexToWordDict.put(i + 1, dict(i))
      wordToIndexDict.put(dict(i), i + 1)
    }
    (indexToWordDict, wordToIndexDict)
  }

  private def saveMatrixFile(rowInd: List[Int], colInd: List[Int], values: List[Int], ratingsList: List[(Int, Int)], numReviewsProcessed: Int, frequencyThreshold: Int, directory: String): Unit = {
    var counts: SMat = sparse(icol(rowInd), icol(colInd), icol(values))
    println(counts.ncols + " " + frequencyThreshold)
    // +1 here because dictionary indices start at 0
    val countsFull: FMat = full(counts \ sparse(zeros(counts.nrows, frequencyThreshold + 1 - counts.ncols)))
    countsFull(?, 0) = ones(counts.nrows, 1)
    counts = sparse(countsFull)
    val counts_t: SMat = counts.t
    val ratingsMat: FMat = full(sparse(icol(ratingsList.map(x => x._1)), icol(List.fill(ratingsList.length)(0)), icol(ratingsList.map(x => x._2))))
    val filename = (numReviewsProcessed + BLOCK_SIZE - 1) / BLOCK_SIZE + ".mat"
    println("saving block " + filename)
    println("size: " + counts.nrows + " " + counts.ncols + " " + ratingsList.length)
    saveAs(directory + "/" + filename, counts, "counts", counts_t, "counts_t", ratingsMat, "ratings")
  }

  def csMatDictToMap(mat: CSMat): (HashMap[String, Int], HashMap[Int, String]) = {
    val gramToIndexDict: HashMap[String, Int] = new HashMap[String, Int]()
    val indexToGramDict: HashMap[Int, String] = new HashMap[Int, String]()
    for (i <- 0 until mat.nrows) {
      gramToIndexDict.put(mat(i), i + 1)
      indexToGramDict.put(i + 1, mat(i))
    }
    (gramToIndexDict, indexToGramDict)
  }

  def parseNgramTokens(tokensPath: String, tokenDict: HashMap[Int, String], gramDict: HashMap[String, Int], numGrams: Int, directory: String) = {
    println("parseNgramTokens")
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
      var gramCounts: HashMap[Int, Int] = new HashMap[Int, Int]()
      var wordQueue: List[Int] = List[Int]();

      new File(directory).mkdir();

      while (true) {
        dataIn.readIntLE()
        dataIn.readIntLE()
        val tokenNum = dataIn.readIntLE()

        val tokenStr = tokenDict.getOrElse(tokenNum, "")
        // println(tokenNum + " " + tokenStr)
        if (tokenStr == startReviewTextDelimiter) {
          countFlag = true
        } else if (tokenStr == endReviewTextDelimiter) {
          countFlag = false
        } else if (tokenStr == startRatingDelimiter) {
          ratingFlag = true
        } else if (tokenStr == endRatingDelimiter) {
          ratingFlag = false
        } else if (tokenStr == endReviewDelimiter) {
          rowInd = List.fill(gramCounts.size)(numReviewsProcessed % BLOCK_SIZE) ::: rowInd
          val wordCountsList: List[(Int, Int)] = gramCounts.toList
          colInd = wordCountsList.map(x => x._1) ::: colInd
          values = wordCountsList.map(x => x._2) ::: values
          ratings.put(numReviewsProcessed % BLOCK_SIZE, reviewRating)
          numReviewsProcessed += 1

          gramCounts = new HashMap[Int, Int]()
          reviewRating = -1

          if (numReviewsProcessed % BLOCK_SIZE == 0) {
            saveMatrixFile(rowInd, colInd, values, ratings.toList, numReviewsProcessed, FEATURE_THRESHOLD, numGrams + "gram_")
            rowInd = List[Int]()
            colInd = List[Int]()
            values = List[Int]()
            ratings = new HashMap[Int, Int]()
          }

        } else if (countFlag == true) {
          if (wordQueue.length == numGrams) {
            for (i <- 1 to wordQueue.length) {
              val w: Seq[Int] = wordQueue.reverse.slice(0, i).toSeq
              val phrase: String = w.map(x => tokenDict.getOrElse(x, "")).reduceLeft(_ + " " + _)
              if (gramDict.contains(phrase)) {
                if (gramDict.get(phrase).get == 0) println("wtf " + phrase)

                gramCounts.put(gramDict.get(phrase).get, gramCounts.getOrElse(gramDict.get(phrase).get, 0) + 1)
              }
            }
            wordQueue = wordQueue.init
          }
          wordQueue = tokenNum :: wordQueue

        } else if (ratingFlag == true) {
          if (tokenStr != "0") {
            reviewRating = tokenStr.toInt - 3 // Make 0 the center score for better use of floating-point range
          }
        }
      }
    } catch {
      case eof: EOFException => {
        saveMatrixFile(rowInd, colInd, values, ratings.toList, numReviewsProcessed, gramDict.size, numGrams + "gram_")
      }
      case e: HDF5LibraryException => println(e.getMessage() + "\n" + "Major error: " + e.getMajorErrorNumber() + ", minor error: " + e.getMinorError(e.getMinorErrorNumber()))
      case e: Exception => println(e.getStackTrace() + " " + numReviewsProcessed)
    }
  }

  /**
   * saves a CSMat of a vector of the "featureThreshold" number of ngrams that are most frequently occurring
   * at the location specified by savePath
   */
  def selectNgramFeatures(tokensPath: String, savePath: String, wordDict: HashMap[Int, String], numGrams: Int, featureThreshold: Int, sampleRatio: Int): (HashMap[String, Int], HashMap[Int, String]) = {
    var numReviewsProcessed: Int = 0
    var ratingFlag: Boolean = false
    var countFlag: Boolean = false
    var reviewRating = -1
    var gramCounts: HashMap[Int, Int] = new HashMap[Int, Int]()
    var gramToIndexDict: HashMap[Seq[Int], Int] = new HashMap[Seq[Int], Int]()
    var indexToGramDict: HashMap[Int, String] = new HashMap[Int, String]()
    var wordQueue: List[Int] = List[Int]();

    try {
      val inputFile: File = new File(tokensPath);
      val fileIn: FileInputStream = new FileInputStream(inputFile)
      val dataIn: LittleEndianDataInputStream = new LittleEndianDataInputStream(fileIn)

      new File(BLOCK_SIZE.toString()).mkdir();

      while (true) {
        dataIn.readIntLE()
        dataIn.readIntLE()
        val tokenNum = dataIn.readIntLE()
        val tokenStr = wordDict.getOrElse(tokenNum, "")
        if (tokenStr == startReviewTextDelimiter) {
          countFlag = true
        } else if (tokenStr == endReviewTextDelimiter) {
          countFlag = false
        } else if (tokenStr == startRatingDelimiter) {
          ratingFlag = true
        } else if (tokenStr == endRatingDelimiter) {
          ratingFlag = false
        } else if (tokenStr == endReviewDelimiter) {
          numReviewsProcessed += 1
          println(numReviewsProcessed + " " + gramToIndexDict.size)
        } else if (countFlag && numReviewsProcessed % sampleRatio == 0) { // only sample every "sampleRatio"th review for ngram parsing
          if (wordQueue.length == numGrams) {
            for (i <- 1 to wordQueue.length) {
              val w: Seq[Int] = wordQueue.reverse.slice(0, i).toSeq
              val wIndex: Int = gramToIndexDict.getOrElse(w, gramToIndexDict.size)
              val phrase: String = w.map(x => wordDict.getOrElse(x, "")).reduceLeft(_ + " " + _)

              gramToIndexDict.put(w, wIndex)
              indexToGramDict.put(wIndex, phrase)
              gramCounts.put(wIndex, gramCounts.getOrElse(wIndex, 0) + 1)
            }
            wordQueue = wordQueue.init
          }
          wordQueue = tokenNum :: wordQueue
        } else if (ratingFlag) {
          if (tokenStr != "0") {
            reviewRating = tokenStr.toInt - 3 // Make 0 the center score for better use of floating-point range
          }
        }
      }
    } catch {
      case eof: EOFException => {
        for (i <- 1 to wordQueue.length) {
          val w: Seq[Int] = wordQueue.reverse.slice(0, i).toSeq
          val wIndex: Int = gramToIndexDict.getOrElse(w, gramToIndexDict.size)
          val phrase: String = w.map(x => wordDict.getOrElse(x, "")).reduceLeft(_ + " " + _)

          gramToIndexDict.put(w, wIndex)
          indexToGramDict.put(wIndex, phrase)
          gramCounts.put(wIndex, gramCounts.getOrElse(wIndex, 0) + 1)
        }
        val sortedGramCounts: List[(Int, Int)] = gramCounts.toList.sort((x, y) => (x._2 compareTo y._2) > 0)
        val kvPairs: List[(String, Int)] = sortedGramCounts.slice(0, featureThreshold).map(x => (indexToGramDict.get(x._1).get, x._1))
        val prunedGramToIndexDict: HashMap[String, Int] = new HashMap[String, Int]()
        val prunedIndexToGramDict: HashMap[Int, String] = new HashMap[Int, String]()
        for ((word, index) <- kvPairs.zipWithIndex) {
          prunedGramToIndexDict.put(word._1, index + 1)
          prunedIndexToGramDict.put(index + 1, word._1)

        }
        println((0 until featureThreshold).length + " " + (List.fill(featureThreshold)(0)).length + sortedGramCounts.slice(0, featureThreshold).map(x => x._1).length)
        val topGrams: SMat = sparse(icol((0 until featureThreshold)), icol(List.fill(featureThreshold)(0)), icol(sortedGramCounts.slice(0, featureThreshold).map(x => x._1)))
        val gramsDict: CSMat = CSMat(featureThreshold, 1, sortedGramCounts.map(x => indexToGramDict.get(x._1).get).toArray)
        saveAs(savePath, topGrams, "grams", gramsDict, "gramsDict")

        (prunedGramToIndexDict, prunedIndexToGramDict)

      }
      case e: HDF5LibraryException => println(e.getMessage() + "\n" + "Major error: " + e.getMajorErrorNumber() + ", minor error: " + e.getMinorError(e.getMinorErrorNumber()))
      case e: Exception => println("what" + e + " " + numReviewsProcessed)
    }

    val sortedGramCounts: List[(Int, Int)] = gramCounts.toList.sort((x, y) => (x._2 compareTo y._2) > 0)
    val kvPairs: List[(String, Int)] = sortedGramCounts.slice(0, featureThreshold).map(x => (indexToGramDict.get(x._1).get, x._1))
    val prunedGramToIndexDict: HashMap[String, Int] = new HashMap[String, Int]()
    val prunedIndexToGramDict: HashMap[Int, String] = new HashMap[Int, String]()
    for ((word, index) <- kvPairs.zipWithIndex) {
      prunedGramToIndexDict.put(word._1, index + 1)
      prunedIndexToGramDict.put(index + 1, word._1)

    }

    val topGrams: SMat = sparse(icol((0 until featureThreshold)), icol(List.fill(featureThreshold)(0)), icol(sortedGramCounts.slice(0, featureThreshold).map(x => x._1)))
    val gramsDict: CSMat = CSMat(featureThreshold, 1, sortedGramCounts.map(x => indexToGramDict.get(x._1).get).toArray)
    saveAs("ngrams.mat", topGrams, "grams", gramsDict, "gramsDict")

    (prunedGramToIndexDict, prunedIndexToGramDict)

  }
}
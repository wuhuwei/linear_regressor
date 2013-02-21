package wu.wei

import BIDMat.{ Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat }
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import scala.collection.mutable.HashMap
import java.io.BufferedReader
import java.util.StringTokenizer
import java.io.FileReader
import java.io.File
import java.io.FileInputStream
import java.io.DataInputStream
import java.io.InputStream
import java.io.EOFException

object LinearRegressor {
  val endReviewDelimiter: String = "</review>"
  val startReviewTextDelimiter: String = "<review_text>"
  val endReviewTextDelimiter: String = "</review_text>"
  val startRatingDelimiter: String = "<rating>"
  val endRatingDelimiter: String = "</rating>"

  def main(args: Array[String]): Unit = {
    var foo: FMat = ones(4, 4)
    saveAs("foo.mat", foo, "foo")
    parseTokens(args(1), parseTagDict(args(0))._1)
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

  /**
   * takes a binary tokens file, and parses BLOCK_SIZE of them at a time, saving them into a
   * a sparse matrix "counts" where counts(i, j) = # of occurrences of word j in review i,
   * and a rating matrix "ratings" where ratings(i, 0) = rating for review i
   */
  def parseTokens(tokensPath: String, tagDict: HashMap[Int, String]) = {
    //println(tokensPath)
    var numDocs: Int = 0
    val BLOCK_SIZE = 1000

    var rowInd: List[Int] = List[Int]()
    var colInd: List[Int] = List[Int]()
    var values: List[Int] = List[Int]()
    var ratings: HashMap[Int, Int] = new HashMap[Int, Int]()

    var currentColumn: List[Int] = List[Int]()
    var numReviewsProcessed: Int = 0

    try {

      val inputFile: File = new File(tokensPath);
      val fileIn: FileInputStream = new FileInputStream(inputFile)
      val dataIn: LittleEndianDataInputStream = new LittleEndianDataInputStream(fileIn)

      var ratingFlag: Boolean = false
      var countFlag: Boolean = false
      var reviewRating = -1
      var reviewWordCounts: HashMap[Int, Int] = new HashMap[Int, Int]();

      while (currentColumn.length <= 3) {
        currentColumn = dataIn.readIntLE() :: currentColumn
        if (currentColumn.length == 3) {
          if (tagDict.getOrElse(currentColumn(0), "") == startReviewTextDelimiter) {
            countFlag = true;
          } else if (tagDict.getOrElse(currentColumn(0), "") == endReviewTextDelimiter) {
            countFlag = false;
          } else if (tagDict.getOrElse(currentColumn(0), "") == startRatingDelimiter) {
            ratingFlag = true;
          } else if (tagDict.getOrElse(currentColumn(0), "") == endRatingDelimiter) {
            ratingFlag = false;
          } else if (tagDict.getOrElse(currentColumn(0), "") == endReviewDelimiter) {
            rowInd = List.fill(reviewWordCounts.size)(numReviewsProcessed % BLOCK_SIZE) ::: rowInd
            val wordCountsList: List[(Int, Int)] = reviewWordCounts.toList
            colInd = wordCountsList.map(x => x._1) ::: colInd
            values = wordCountsList.map(x => x._2) ::: values
            ratings.put(numReviewsProcessed % BLOCK_SIZE, reviewRating)
            numReviewsProcessed += 1

            reviewWordCounts = new HashMap[Int, Int]()
            reviewRating = -1

            if (numReviewsProcessed % BLOCK_SIZE == 0) {
              val counts: SMat = sparse(icol(rowInd), icol(colInd), icol(values))
              val ratingsList: List[(Int, Int)] = ratings.toList
              val ratingsMat: SMat = sparse(icol(ratingsList.map(x => x._1)), icol(List.fill(ratingsList.length)(0)), icol(ratingsList.map(x => x._2)))
              println("saving block " + numReviewsProcessed / BLOCK_SIZE + ".mat")
              println("size: " + counts.nrows + " " + ratingsList.length)
              saveAs(BLOCK_SIZE + "/" + numReviewsProcessed / BLOCK_SIZE + ".mat", counts, "counts", ratingsMat, "ratings")

              rowInd = List[Int]()
              colInd = List[Int]()
              values = List[Int]()
              ratings = new HashMap[Int, Int]()
            }

          } else if (countFlag == true) {
            reviewWordCounts.put(currentColumn(0), reviewWordCounts.getOrElse(currentColumn(0), 0) + 1)
          } else if (ratingFlag == true) {
            reviewRating = currentColumn(0);
          }
          currentColumn = List[Int]()
        }
      }
    } catch {
      case eof: EOFException => {
        val counts: SMat = sparse(icol(rowInd), icol(colInd), icol(values))
        val ratingsList: List[(Int, Int)] = ratings.toList
        val ratingsMat: SMat = sparse(icol(ratingsList.map(x => x._1)), icol(List.fill(ratingsList.length)(0)), icol(ratingsList.map(x => x._2)))
        println("saving block " + numReviewsProcessed / BLOCK_SIZE + ".mat")
        println("size: " + counts.nrows + " " + ratingsList.length)
        saveAs(BLOCK_SIZE + "/" + numReviewsProcessed / BLOCK_SIZE + ".mat", counts, "counts", ratingsMat, "ratings")
      }
      case e: Exception => println("what" + e + " " + numReviewsProcessed)
    }

  }
}

class LittleEndianDataInputStream(i: InputStream) extends DataInputStream(i) {
  def readLongLE(): Long = java.lang.Long.reverseBytes(super.readLong())
  def readIntLE(): Int = java.lang.Integer.reverseBytes(super.readInt())
  def readCharLE(): Char = java.lang.Character.reverseBytes(super.readChar())
  def readShortLE(): Short = java.lang.Short.reverseBytes(super.readShort())
}
/**
  * Created by huiminren on 7/10/17.
  */
import java.text.SimpleDateFormat

import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
//import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._

object RepeatBuyers extends App{
//  def main(args: Array[String]): Unit = {
//
//    if (args.length < 2) {
//      System.err.println("Usage: <input>"+
//        "<output> [<filters>]")
//      System.exit(1)
//    }
//  }

  val spark = SparkSession
    .builder
    .appName("RepeatBuyers")
    .config("spark.master", "local")
    .getOrCreate()

  LogManager.getRootLogger().setLevel(Level.ERROR)

  // For implicit conversions like converting RDDs to DataFrames
  import spark.implicits._

  // Load data
//  val path = "/Users/huiminren/Desktop/DS504/Final/data/"
  val path = "/Users/huiminren/Google Drive/DS504/Final Project/Data/data_format1"
//  val test_df = spark.read.option("header","true").csv(path+"test_format1.csv")
//  test_df.show()
//
//  val train_df = spark.read.option("header","true").csv(path+"train_format1.csv")
//  train_df.show()
//
//  val user_info_df0 = spark.read.option("header","true").csv(path+"user_info_format1.csv")
//  val ui_df1 = user_info_df0.na.fill("0",Seq("age_range"))
//  val user_info_df = ui_df1.na.fill("2",Seq("gender"))
//  user_info_df.show()

  val user_log_df0 = spark.read.option("header","true").csv(path+"/user_log_sample.csv") //user_log_sample user_log_format1
//  user_log_df.show()

// Convert "time_stamp" from String to Date
  val dateToTimeStamp = udf((date: String) => {
    val stringDate = "2014"+"/"+date.substring(0,2)+"/"+date.substring(2,4)
    val format = new SimpleDateFormat("yyyy-MM-dd")
    format.format(new SimpleDateFormat("yyy/MM/dd").parse(stringDate))
  })
  val user_log_df = user_log_df0.withColumn("time_stamp", dateToTimeStamp($"time_stamp"))



  // Calculate repeat features
  // Average span between any two actions
//  val a1 = user_log_df.groupBy("user_id").agg(max("time_stamp") as "max_date").toDF()
//  val a2 = user_log_df.groupBy("user_id").agg(min("time_stamp") as "min_date").toDF()
//  val a3 = user_log_df.groupBy("user_id").agg(count("user_id") as "count").toDF()
//
//  a1.createOrReplaceTempView("a1")
//  a2.createOrReplaceTempView("a2")
//  a3.createOrReplaceTempView("a3")
//  val a4 = spark.sql("select a1.user_id, max_date,min_date,count from a1,a2,a3 " +
//    "where a1.user_id = a2.user_id and a1.user_id=a3.user_id")
//
//  val avg_span_act_df = a4.withColumn("avg_span_actions",(datediff($"max_date",$"min_date"))/$"count")
//  avg_span_act_df.show()

  // Average span between 2 purchases
  user_log_df.createOrReplaceTempView("user_log")
//  val b1 = spark.sql("SELECT * FROM user_log where action_type = 2 order by user_id, time_stamp")
//  val b2 = b1.groupBy("user_id").agg(max("time_stamp") as "max_date").toDF()
//  val b3 = b1.groupBy("user_id").agg(min("time_stamp") as "min_date").toDF()
//  val b4 = b1.groupBy("user_id").agg(count("user_id") as "count").toDF()
//
//  b2.createOrReplaceTempView("b2")
//  b3.createOrReplaceTempView("b3")
//  b4.createOrReplaceTempView("b4")
//  val b5 = spark.sql("select b2.user_id, max_date,min_date,count from b4,b2,b3 " +
//      "where b4.user_id = b2.user_id and b4.user_id=b3.user_id")
//
//  val avg_span_purchase_df = b5.withColumn("avg_span_actions",(datediff($"max_date",$"min_date"))/$"count")
//  avg_span_purchase_df.show()

  // How many days since the last purchase
//  val c1 = spark.sql("select *,'2014-11-11' as double_11 from user_log where time_stamp<'2014-11-11' and action_type = 2")
//  val c2 = c1.groupBy("user_id").agg(max("time_stamp") as "max2_date", max("double_11") as "double_11").toDF()
//  val last_span = c2.withColumn("last_span", datediff($"double_11",$"max2_date"))
//  last_span.show()

  
}

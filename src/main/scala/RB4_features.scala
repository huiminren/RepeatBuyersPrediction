import java.text.SimpleDateFormat

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import scopt.OptionParser

/**
  * Created by huiminren on 7/27/17.
  */
object RB4 {

  case class Params(logInput:String = "",
                    infoInput:String = "",
                    labelInput: String = "",
                    outputFile: String = "",
                    outputFileName: String =""
                    )

  def FeaturesProcess(params: Params): Unit ={

    val spark = SparkSession.builder.appName("RepeatBuyers").config("spark.master", "local").getOrCreate()
    import spark.implicits._

    var user_log_sample = spark.read
      .option("header", "true")
      .csv(params.logInput)

    val newNames = Seq("user_id", "seller_id", "label")
    var label_df = spark.read
      .option("header", "true")
      .csv(params.labelInput)

    user_log_sample = user_log_sample.join(label_df, Seq("user_id", "seller_id"))

    val dateToTimeStamp = udf((date: String) => {
      val stringDate = "2014" + "/" + date.substring(0, 2) + "/" + date.substring(2, 4)
      val format = new SimpleDateFormat("yyyy-MM-dd")
      format.format(new SimpleDateFormat("yyy/MM/dd").parse(stringDate))
    })

    user_log_sample = user_log_sample.withColumn("time_stamp", dateToTimeStamp($"time_stamp"))
    user_log_sample.createOrReplaceTempView("sample_log")
    val click = spark.sql("SELECT * FROM sample_log where action_type = 0")
    val purchase = spark.sql("SELECT * FROM sample_log where action_type = 2")
    val favorite = spark.sql("SELECT * FROM sample_log where action_type = 3")

    val user_click = click.groupBy("user_id").agg(count("user_id") as "ttl_c")
    val user_purchase = purchase.groupBy("user_id").agg(count("user_id") as "ttl_p")
    val user_favor = favorite.groupBy("user_id").agg(count("user_id") as "ttl_f")

    val item_click = click.groupBy("user_id").agg(countDistinct("item_id") as "dis_c")
    val item_purchase = purchase.groupBy("user_id").agg(countDistinct("item_id") as "dis_p")
    val item_favor = favorite.groupBy("user_id").agg(countDistinct("item_id") as "dis_f")

    val merchants_click = click.groupBy("user_id").agg(countDistinct("seller_id") as "mer_c")
    val merchants_purchase = purchase.groupBy("user_id").agg(countDistinct("seller_id") as "mer_p")
    val merchants_favor = favorite.groupBy("user_id").agg(countDistinct("seller_id") as "mer_f")

    val cat_click = click.groupBy("user_id").agg(countDistinct("cat_id") as "cat_c")
    val cat_purchase = purchase.groupBy("user_id").agg(countDistinct("cat_id") as "cat_p")
    val cat_favor = favorite.groupBy("user_id").agg(countDistinct("cat_id") as "cat_f")

    val active_days = user_log_sample.groupBy("user_id").agg(countDistinct("time_stamp") as "ttl_act_d")

    val merchant_total_click = click.groupBy("seller_id").agg(count("seller_id") as "c_cnt")
    val merchant_total_purchase = purchase.groupBy("seller_id").agg(count("seller_id") as "p_cnt")
    val merchant_total_favor = favorite.groupBy("seller_id").agg(count("seller_id") as "f_cnt")
    val user_count_click_merchant = click.groupBy("seller_id").agg(countDistinct("user_id") as "c_u_cnt")
    val user_count_purchase_merchant = purchase.groupBy("seller_id").agg(countDistinct("user_id") as "p_u_cnt")
    val user_count_favor_merchant = favorite.groupBy("seller_id").agg(countDistinct("user_id") as "f_u_cnt")

    val user_merchant_click = click.groupBy("user_id", "seller_id").agg(count("user_id") as "u_m_c_cnt")
    val user_merchant_purchase = purchase.groupBy("user_id", "seller_id").agg(count("user_id") as "u_m_p_cnt")
    val user_merchant_favor = favorite.groupBy("user_id", "seller_id").agg(count("user_id") as "u_m_f_cnt")
    val user_merchant_active_days = user_log_sample.groupBy("user_id", "seller_id").agg(countDistinct("time_stamp") as "um_act_d")

    // Repeat
    // Average span between any 2 actions
    val a1 = user_log_sample.groupBy("user_id").agg(max("time_stamp") as "max_date")
    val a2 = user_log_sample.groupBy("user_id").agg(min("time_stamp") as "min_date")
    val a3 = user_log_sample.groupBy("user_id").agg(count("user_id") as "count")
    val x = a1.join(a2, Seq("user_id"))
    val a4 = x.join(a3, Seq("user_id"))
    val a5 = a4.withColumn("avg_span_actions", (datediff($"max_date", $"min_date")) / $"count")
    val a11 = a5.drop(a5.col("max_date"))
    val a22 = a11.drop(a11.col("min_date"))
    val avg_span_act_df = a22.drop(a22.col("count"))

    // Average span between 2 purchases
    user_log_sample.createOrReplaceTempView("user_log")
    val b1 = spark.sql("SELECT * FROM user_log where action_type = 2 order by user_id, time_stamp")
    val b2 = b1.groupBy("user_id").agg(max("time_stamp") as "max_date").toDF()
    val b3 = b1.groupBy("user_id").agg(min("time_stamp") as "min_date").toDF()
    val b4 = b1.groupBy("user_id").agg(count("user_id") as "count").toDF()
    val y = b2.join(b3, Seq("user_id"))
    val b5 = y.join(b4, Seq("user_id"))
    val b6 = b5.withColumn("avg_span_purchase", (datediff($"max_date", $"min_date")) / $"count")
    val b11 = b6.drop(b6.col("max_date"))
    val b22 = b11.drop(b11.col("min_date"))
    val avg_span_purchase_df = b22.drop(b22.col("count"))

    // How many days since the last purchase
    val c1 = spark.sql("select *,'2014-11-11' as double_11 from user_log " +
      "where time_stamp<'2014-11-11' and action_type = 2")
    val c2 = c1.groupBy("user_id").agg(max("time_stamp") as "max2_date", max("double_11") as "double_11").toDF()
    val c3 = c2.withColumn("last_span", datediff($"double_11", $"max2_date"))
    val c11 = c3.drop(c3.col("max2_date"))
    val last_span = c11.drop(c11.col("double_11"))

    // average span between any 2 actions for one merchant
    val d1 = user_log_sample.groupBy("user_id", "seller_id").agg(max("time_stamp") as "max_date")
    val d2 = user_log_sample.groupBy("user_id", "seller_id").agg(min("time_stamp") as "min_date")
    val d3 = user_log_sample.groupBy("user_id", "seller_id").agg(count("user_id") as "cnt_um")
    val z = d1.join(d2, Seq("user_id", "seller_id"))
    val d4 = z.join(d3, Seq("user_id", "seller_id"))
    val d5 = d4.withColumn("avg_span_act_um", (datediff($"max_date", $"min_date")) / $"cnt_um")
    val d11 = d5.drop(d5.col("max_date"))
    val d22 = d11.drop(d11.col("min_date"))
    val avg_span_act_um_df = d22.drop(d22.col("cnt_um"))

    // Repeat ratio
    user_merchant_click.createOrReplaceTempView("table1")
    user_merchant_purchase.createOrReplaceTempView("table2")
    user_merchant_favor.createOrReplaceTempView("table3")
    val total_action = user_log_sample.groupBy("user_id", "seller_id").agg(count("user_id") as "total_action_count")
    total_action.createOrReplaceTempView("action_count")
    val temp = spark.sql("SELECT action_count.user_id, action_count.seller_id, action_count.total_action_count, " +
      "table1.u_m_c_cnt FROM action_count, table1 where action_count.user_id = table1.user_id and " +
      "action_count.seller_id = table1.seller_id")
    val click_ratio0 = temp.withColumn("c_ratio", temp("u_m_c_cnt") / temp("total_action_count"))
    val temp2 = temp.select($"u_m_c_cnt" / $"total_action_count")
    val temp3 = spark.sql("SELECT action_count.user_id, action_count.seller_id, action_count.total_action_count, " +
      "table2.u_m_p_cnt FROM action_count, table2 where action_count.user_id = table2.user_id and " +
      "action_count.seller_id = table2.seller_id")
    val purchase_ratio0 = temp3.withColumn("p_ratio", temp3("u_m_p_cnt") / temp3("total_action_count"))
    val temp4 = spark.sql("SELECT action_count.user_id, action_count.seller_id, action_count.total_action_count," +
      " table3.u_m_f_cnt FROM action_count, table3 where action_count.user_id = table3.user_id and " +
      "action_count.seller_id = table3.seller_id")
    val favor_ratio0 = temp4.withColumn("f_ratio", temp4("u_m_f_cnt") / temp4("total_action_count"))
    spark.catalog.dropTempView("action_count")
    spark.catalog.dropTempView("table1")
    spark.catalog.dropTempView("table2")
    spark.catalog.dropTempView("table3")

    val click_ratio1 = click_ratio0.drop(click_ratio0.col("u_m_c_cnt"))
    val click_ratio = click_ratio1.drop(click_ratio1.col("total_action_count"))
    val purchase_ratio1 = purchase_ratio0.drop(purchase_ratio0.col("u_m_p_cnt"))
    val purchase_ratio = purchase_ratio1.drop(purchase_ratio1.col("total_action_count"))
    val favor_ratio = favor_ratio0.drop(favor_ratio0.col("u_m_f_cnt"))
    //  val favor_ratio = favor_ratio1.drop(favor_ratio1.col("total_action_count"))

    val user0 = user_click.join(user_purchase, Seq("user_id"), "outer")
      .join(user_favor, Seq("user_id"), "outer")
      .join(item_click, Seq("user_id"), "outer")
      .join(item_purchase, Seq("user_id"), "outer")
      .join(item_favor, Seq("user_id"), "outer") //user+item = join1
      .join(merchants_click, Seq("user_id"), "outer")
      .join(merchants_purchase, Seq("user_id"), "outer")
      .join(merchants_favor, Seq("user_id"), "outer")
      .join(cat_click, Seq("user_id"), "outer")
      .join(cat_purchase, Seq("user_id"), "outer")
      .join(cat_favor, Seq("user_id"), "outer") //+merchant,cat = join2
      .join(active_days, Seq("user_id"), "outer")
      .join(avg_span_act_df, Seq("user_id"), "outer")
      .join(avg_span_purchase_df, Seq("user_id"), "outer")
      .join(last_span, Seq("user_id"), "outer")

    val may = user_log_sample.filter(month($"time_stamp").equalTo(lit("05")))
    val jun = user_log_sample.filter(month($"time_stamp").equalTo(lit("06")))
    val jul = user_log_sample.filter(month($"time_stamp").equalTo(lit("07")))
    val aug = user_log_sample.filter(month($"time_stamp").equalTo(lit("08")))
    val sep = user_log_sample.filter(month($"time_stamp").equalTo(lit("09")))
    val oct = user_log_sample.filter(month($"time_stamp").equalTo(lit("10")))
    val nov = user_log_sample.filter(month($"time_stamp").equalTo(lit("11")))
    val may_c = may.where("action_type = 0").groupBy("user_id").agg(count("user_id") as "may_c_cnt")
    val jun_c = jun.where("action_type = 0").groupBy("user_id").agg(count("user_id") as "jun_c_cnt")
    val jul_c = jul.where("action_type = 0").groupBy("user_id").agg(count("user_id") as "jul_c_cnt")
    val aug_c = aug.where("action_type = 0").groupBy("user_id").agg(count("user_id") as "aug_c_cnt")
    val sep_c = sep.where("action_type = 0").groupBy("user_id").agg(count("user_id") as "sep_c_cnt")
    val oct_c = oct.where("action_type = 0").groupBy("user_id").agg(count("user_id") as "oct_c_cnt")
    val nov_c = nov.where("action_type = 0").groupBy("user_id").agg(count("user_id") as "nov_c_cnt")
    val may_p = may.where("action_type = 2").groupBy("user_id").agg(count("user_id") as "may_p_cnt")
    val jun_p = jun.where("action_type = 2").groupBy("user_id").agg(count("user_id") as "jun_p_cnt")
    val jul_p = jul.where("action_type = 2").groupBy("user_id").agg(count("user_id") as "jul_p_cnt")
    val aug_p = aug.where("action_type = 2").groupBy("user_id").agg(count("user_id") as "aug_p_cnt")
    val sep_p = sep.where("action_type = 2").groupBy("user_id").agg(count("user_id") as "sep_p_cnt")
    val oct_p = oct.where("action_type = 2").groupBy("user_id").agg(count("user_id") as "oct_p_cnt")
    val nov_p = nov.where("action_type = 2").groupBy("user_id").agg(count("user_id") as "nov_p_cnt")
    val may_f = may.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "may_f_cnt")
    val jun_f = jun.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "jun_f_cnt")
    val jul_f = jul.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "jul_f_cnt")
    val aug_f = aug.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "aug_f_cnt")
    val sep_f = sep.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "sep_f_cnt")
    val oct_f = oct.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "oct_f_cnt")
    val nov_f = nov.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "nov_f_cnt")
    val may_tt = may.groupBy("user_id").agg(count("user_id") as "may_tt")
    val jun_tt = jun.groupBy("user_id").agg(count("user_id") as "jun_tt")
    val jul_tt = jul.groupBy("user_id").agg(count("user_id") as "jul_tt")
    val aug_tt = aug.groupBy("user_id").agg(count("user_id") as "aug_tt")
    val sep_tt = sep.groupBy("user_id").agg(count("user_id") as "sep_tt")
    val oct_tt = oct.groupBy("user_id").agg(count("user_id") as "oct_tt")
    val nov_tt = nov.groupBy("user_id").agg(count("user_id") as "nov_tt")

    val monthdf = may_c.join(may_p, Seq("user_id"), "left_outer")
      .join(may_f, Seq("user_id"), "left_outer")
      .join(jun_c, Seq("user_id"), "left_outer")
      .join(jun_p, Seq("user_id"), "left_outer")
      .join(jun_f, Seq("user_id"), "left_outer")
      .join(jul_c, Seq("user_id"), "left_outer")
      .join(jul_p, Seq("user_id"), "left_outer")
      .join(jul_f, Seq("user_id"), "left_outer")
      .join(aug_c, Seq("user_id"), "left_outer")
      .join(aug_p, Seq("user_id"), "left_outer")
      .join(aug_f, Seq("user_id"), "left_outer")
      .join(sep_c, Seq("user_id"), "left_outer")
      .join(sep_p, Seq("user_id"), "left_outer")
      .join(sep_f, Seq("user_id"), "left_outer")
      .join(oct_c, Seq("user_id"), "left_outer")
      .join(oct_p, Seq("user_id"), "left_outer")
      .join(oct_f, Seq("user_id"), "left_outer")
      .join(nov_c, Seq("user_id"), "left_outer")
      .join(nov_p, Seq("user_id"), "left_outer")
      .join(nov_f, Seq("user_id"), "left_outer")
      .join(may_tt, Seq("user_id"), "left_outer")
      .join(jun_tt, Seq("user_id"), "left_outer")
      .join(jul_tt, Seq("user_id"), "left_outer")
      .join(aug_tt, Seq("user_id"), "left_outer")
      .join(sep_tt, Seq("user_id"), "left_outer")
      .join(oct_tt, Seq("user_id"), "left_outer")
      .join(nov_tt, Seq("user_id"), "left_outer")

    val user16 = user0.join(monthdf, Seq("user_id"), "outer")

    val seller0 = merchant_total_click.join(merchant_total_purchase, Seq("seller_id"), "outer")
      .join(merchant_total_favor, Seq("seller_id"), "outer")
      .join(user_count_click_merchant, Seq("seller_id"), "outer")
      .join(user_count_purchase_merchant, Seq("seller_id"), "outer")
      .join(user_count_favor_merchant, Seq("seller_id"), "outer")

    val um0 = user_merchant_click.join(user_merchant_purchase, Seq("user_id", "seller_id"), "outer")
      .join(user_merchant_favor, Seq("user_id", "seller_id"), "outer")
      .join(user_merchant_active_days, Seq("user_id", "seller_id"), "outer")
      .join(avg_span_act_um_df, Seq("user_id", "seller_id"), "outer")
      .join(click_ratio, Seq("user_id", "seller_id"), "outer")
      .join(purchase_ratio, Seq("user_id", "seller_id"), "outer")
      .join(favor_ratio, Seq("user_id", "seller_id"), "outer")

    val um_user = um0.join(user16, Seq("user_id"), "left_outer")
    val um_user_seller = um_user.join(seller0, Seq("seller_id"), "left_outer")

    var user_info_df = spark.read.option("header", "true").csv(params.infoInput)
    user_info_df = user_info_df.na.fill("0", Seq("age_range"))
    user_info_df = user_info_df.na.fill("2", Seq("gender"))

    val all_features = um_user_seller.join(user_info_df, Seq("user_id"), "left_outer")
    var features_label = label_df.join(all_features, Seq("user_id", "seller_id"))
    features_label = features_label.na.fill(0)

    // save features
    def merge(srcPath: String, dstPath: String): Unit = {
      val hadoopConfig = new Configuration()
      val hdfs = FileSystem.get(hadoopConfig)
      FileUtil.copyMerge(hdfs, new Path(srcPath), hdfs, new Path(dstPath), true, hadoopConfig, null)
      // the "true" setting deletes the source files once they are merged into the new output
    }

    val outputFileName = params.outputFile + "/temp_" + params.outputFileName
    val mergedFileName = params.outputFile + "/merged_" + params.outputFileName + ".csv"
    val mergeFindGlob = outputFileName

    features_label.write
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .mode("overwrite")
      .save(outputFileName)
    merge(mergeFindGlob, mergedFileName)
    features_label.unpersist()
  }

  def main(args: Array[String]){
    val parser = new OptionParser[Params]("AllstateClaimsSeverityRandomForestRegressor") {
      head("AllstateClaimsSeverityRandomForestRegressor", "1.0")

      opt[String]("logInput").required().valueName("<file>").action((x, c) =>
        c.copy(logInput = x)).text("Path to file for user log")

      opt[String]("infoInput").required().valueName("<file>").action((x, c) =>
        c.copy(infoInput = x)).text("Path to file for user info")

      opt[String]("labelInput").required().valueName("<file>").action((x, c) =>
        c.copy(labelInput = x)).text("Path to file/directory for training or test data")

      opt[String]("outputFile").valueName("<file>").action((x, c) =>
        c.copy(outputFile = x)).text("Path to output file")

      opt[String]("outputFileName").required().valueName("<file>").action((x, c) =>
        c.copy(outputFileName = x)).text("name of saving training or test labeled data")
    }

    parser.parse(args, Params()) match {
      case Some(params) =>
        FeaturesProcess(params)
      case None =>
        throw new IllegalArgumentException("One or more parameters are invalid or missing")
    }
  }
}

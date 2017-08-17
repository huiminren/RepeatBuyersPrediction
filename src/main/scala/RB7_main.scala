import org.apache.spark.sql.functions._
import java.text.SimpleDateFormat

import org.apache.spark.sql.functions.countDistinct
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.{Normalizer, VectorAssembler}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}


/**
  * Created by huiminren on 7/28/17.
  */

object RB7_all-code-main {
	case class Params(logInput:String = "",
                    infoInput:String = "",
                    trainInput: String = "",
                    testInput: String = "",
                    outputPath: String = ""
                    )

	case class Features_Lable(user_id: Double,seller_id: Double,u_m_c_cnt: Double,u_m_p_cnt: Double,u_m_p_i_cnt: Double,
                          u_m_r_cnt: Double,u_m_f_cnt: Double,um_act_d: Double,avg_span_act_um: Double,
                          total_action_count: Double,c_ratio: Double,p_ratio: Double,f_ratio: Double,ttl_c: Double,
                          ttl_p: Double,ttl_f: Double,dis_c: Double,dis_p: Double,dis_f: Double,mer_c: Double,mer_p: Double,
                          mer_f: Double,cat_c: Double,cat_p: Double,cat_f: Double,brand_c: Double,brand_p: Double,
                          brand_f: Double,ttl_act_d: Double,u_c_d_cnt: Double,u_p_d_cnt: Double,u_f_d_cnt: Double,
                          u_p_m_ratio: Double,avg_span_actions: Double,avg_span_purchase: Double,last_span: Double,
                          may_c_cnt: Double,may_p_cnt: Double, may_f_cnt: Double,jun_c_cnt: Double,jun_p_cnt: Double,
                          jun_f_cnt: Double,jul_c_cnt: Double,jul_p_cnt: Double,jul_f_cnt: Double, aug_c_cnt: Double,
                          aug_p_cnt: Double, aug_f_cnt: Double,sep_c_cnt: Double,sep_p_cnt: Double,sep_f_cnt: Double,
                          oct_c_cnt: Double,oct_p_cnt: Double,oct_f_cnt: Double,nov_c_cnt: Double,nov_p_cnt: Double,
                          nov_f_cnt: Double,may_tt: Double,jun_tt: Double,jul_tt: Double,aug_tt: Double,sep_tt: Double,
                          oct_tt: Double,nov_tt: Double,ttl_item: Double,ttl_cat: Double,ttl_brand: Double,
                          c_cnt: Double,p_cnt: Double,f_cnt: Double,c_u_cnt: Double,p_u_cnt: Double,f_u_cnt: Double,
                          age_range: Double,gender: Double,label: Double)

	def FeaturesProcess(datafile: String, tempfolder: String, logfile: String, infofile:String,tempOutputPath:String): DataFrame ={
		val spark = SparkSession.builder.appName("RepeatBuyers").config("spark.master", "local").getOrCreate()
    	import spark.implicits._
    	val newNames = Seq("user_id", "seller_id", "label")

    var user_log_sample = spark.read 
      .option("header", "true")
      .csv(path + logfile) 

    var label_df = spark.read.option("header", "true").csv(path + datafile) //!!!!!!!
    label_df = label_df.toDF(newNames: _*)

    user_log_sample = user_log_sample.join(label_df, Seq("user_id", "seller_id"))

    val dateToTimeStamp = udf((date: String) => {
      val stringDate = "2014" + "/" + date.substring(0, 2) + "/" + date.substring(2, 4)
      val format = new SimpleDateFormat("yyyy-MM-dd")
      format.format(new SimpleDateFormat("yyy/MM/dd").parse(stringDate))
    })

    user_log_sample = user_log_sample.withColumn("time_stamp", dateToTimeStamp($"time_stamp"))
    user_log_sample.createOrReplaceTempView("sample_log")
    val click = user_log_sample.where("action_type = 0")
    val purchase = user_log_sample.where("action_type = 2")
    val favorite = user_log_sample.where("action_type = 3")

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

    val brand_click = click.groupBy("user_id").agg(countDistinct("brand_id") as "brand_c")
    val brand_purchase = purchase.groupBy("user_id").agg(countDistinct("brand_id") as "brand_p")
    val brand_favor = favorite.groupBy("user_id").agg(countDistinct("brand_id") as "brand_f")


    val active_days = user_log_sample.groupBy("user_id").agg(countDistinct("time_stamp") as "ttl_act_d")
    val u_click_day_count = click.groupBy("user_id").agg(countDistinct("time_stamp") as "u_c_d_cnt")
    val u_purchase_day_count = purchase.groupBy("user_id").agg(countDistinct("time_stamp") as "u_p_d_cnt")
    val u_favor_day_count = favorite.groupBy("user_id").agg(countDistinct("time_stamp") as "u_f_d_cnt")

    val user_total_merchant_count = user_log_sample.groupBy("user_id").agg(countDistinct("seller_id") as "u_ttl_m_cnt")
    val user_purchase_merchant_count = purchase.groupBy("user_id", "seller_id").agg(countDistinct("seller_id") as "u_m_p_num")
    val t1 = user_total_merchant_count.join(user_purchase_merchant_count, Seq("user_id"),"outer")
    val user_purchase_m_ratio = t1.withColumn("u_p_m_ratio",t1("u_m_p_num")/t1("u_ttl_m_cnt")).select("user_id","u_p_m_ratio")

    val merchant_item_count = user_log_sample.groupBy("seller_id").agg(countDistinct("item_id") as "ttl_item")
    val merchant_cat_count = user_log_sample.groupBy("seller_id").agg(countDistinct("cat_id") as "ttl_cat")
    val merchant_brand_count = user_log_sample.groupBy("seller_id").agg(countDistinct("brand_id") as "ttl_brand")
    val merchant_total_click = click.groupBy("seller_id").agg(count("seller_id") as "c_cnt")
    val merchant_total_purchase = purchase.groupBy("seller_id").agg(count("seller_id") as "p_cnt")
    val merchant_total_favor = favorite.groupBy("seller_id").agg(count("seller_id") as "f_cnt")
    val user_count_click_merchant = click.groupBy("seller_id").agg(countDistinct("user_id") as "c_u_cnt")
    val user_count_purchase_merchant = purchase.groupBy("seller_id").agg(countDistinct("user_id") as "p_u_cnt")
    val user_count_favor_merchant = favorite.groupBy("seller_id").agg(countDistinct("user_id") as "f_u_cnt")

    val user_merchant_click = click.groupBy("user_id", "seller_id").agg(count("user_id") as "u_m_c_cnt")
    val user_merchant_purchase = purchase.groupBy("user_id", "seller_id").agg(count("user_id") as "u_m_p_cnt")
    val user_merchant_item = purchase.groupBy("user_id", "seller_id").agg(countDistinct("item_id") as "u_m_p_i_cnt")
    val um_repeat_cnt = user_merchant_purchase.withColumn("u_m_r_cnt",user_merchant_purchase("u_m_p_cnt")-1).drop("u_m_p_cnt")
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

    val click_ratio = click_ratio0.drop(click_ratio0.col("u_m_c_cnt"))
    val purchase_ratio1 = purchase_ratio0.drop(purchase_ratio0.col("u_m_p_cnt"))
    val purchase_ratio = purchase_ratio1.drop(purchase_ratio1.col("total_action_count"))
    val favor_ratio1 = favor_ratio0.drop(favor_ratio0.col("u_m_f_cnt"))
    val favor_ratio = favor_ratio1.drop(favor_ratio1.col("total_action_count"))

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
      .join(brand_click, Seq("user_id"), "outer")
      .join(brand_purchase, Seq("user_id"), "outer")
      .join(brand_favor, Seq("user_id"), "outer")
      .join(active_days, Seq("user_id"), "outer")
      .join(u_click_day_count, Seq("user_id"), "outer")
      .join(u_purchase_day_count, Seq("user_id"), "outer")
      .join(u_favor_day_count, Seq("user_id"), "outer")
      .join(user_purchase_m_ratio, Seq("user_id"), "outer")
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

    val seller0 = merchant_item_count
      .join(merchant_cat_count, Seq("seller_id"), "outer")
      .join(merchant_brand_count, Seq("seller_id"), "outer")
      .join(merchant_total_click, Seq("seller_id"), "outer")
      .join(merchant_total_purchase, Seq("seller_id"), "outer")
      .join(merchant_total_favor, Seq("seller_id"), "outer")
      .join(user_count_click_merchant, Seq("seller_id"), "outer")
      .join(user_count_purchase_merchant, Seq("seller_id"), "outer")
      .join(user_count_favor_merchant, Seq("seller_id"), "outer")

    val um0 = user_merchant_click
      .join(user_merchant_purchase, Seq("user_id", "seller_id"), "outer")
      .join( user_merchant_item, Seq("user_id", "seller_id"), "outer")
      .join(um_repeat_cnt, Seq("user_id", "seller_id"), "outer")
      .join(user_merchant_favor, Seq("user_id", "seller_id"), "outer")
      .join(user_merchant_active_days, Seq("user_id", "seller_id"), "outer")
      .join(avg_span_act_um_df, Seq("user_id", "seller_id"), "outer")
      .join(click_ratio, Seq("user_id", "seller_id"), "outer")
      .join(purchase_ratio, Seq("user_id", "seller_id"), "outer")
      .join(favor_ratio, Seq("user_id", "seller_id"), "outer")

    val um_user = um0.join(user16, Seq("user_id"), "left_outer")
    val um_user_seller = um_user.join(seller0, Seq("seller_id"), "left_outer")

    var user_info_df = spark.read.option("header", "true").csv(path + infofile)
    user_info_df = user_info_df.na.fill("0", Seq("age_range"))
    user_info_df = user_info_df.na.fill("2", Seq("gender"))

    val all_features = um_user_seller.join(user_info_df, Seq("user_id"), "left_outer")
      .join(label_df, Seq("user_id", "seller_id"))

    var features_label = all_features.na.fill(0)
    features_label = features_label.na.fill("0")
    //    features_label.show(1)

    // save features

    val outputfile = tempOutputPath
    val filename = tempfolder
    val outputFileName = outputfile + "/temp_" + filename
    val mergedFileName = outputfile + "/merged_" + filename
    val mergeFindGlob = outputFileName

    features_label.write
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .mode("overwrite")
      .save(outputFileName)
    merge(mergeFindGlob, mergedFileName)
    features_label.unpersist()

    // load data
    val features_label0 = spark.sparkContext.textFile(mergedFileName)
    val features_labelDF = features_label0.map(_.split(",")).map(attributes => Features_Lable(attributes(0).toDouble,
      attributes(1).toDouble, attributes(2).toDouble, attributes(3).toDouble, attributes(4).toDouble, attributes(5).toDouble,
      attributes(6).toDouble, attributes(7).toDouble, attributes(8).toDouble, attributes(9).toDouble, attributes(10).toDouble,
      attributes(11).toDouble, attributes(12).toDouble, attributes(13).toDouble, attributes(14).toDouble,
      attributes(15).toDouble, attributes(16).toDouble, attributes(17).toDouble, attributes(18).toDouble,
      attributes(19).toDouble, attributes(20).toDouble, attributes(21).toDouble, attributes(22).toDouble,
      attributes(23).toDouble, attributes(24).toDouble, attributes(25).toDouble, attributes(26).toDouble, attributes(27).toDouble,
      attributes(28).toDouble, attributes(29).toDouble, attributes(30).toDouble, attributes(31).toDouble,
      attributes(32).toDouble, attributes(33).toDouble, attributes(34).toDouble, attributes(35).toDouble,
      attributes(36).toDouble, attributes(37).toDouble, attributes(38).toDouble,
      attributes(39).toDouble, attributes(40).toDouble, attributes(41).toDouble, attributes(42).toDouble,
      attributes(43).toDouble, attributes(44).toDouble, attributes(45).toDouble, attributes(46).toDouble,
      attributes(47).toDouble, attributes(48).toDouble, attributes(49).toDouble,
      attributes(50).toDouble, attributes(51).toDouble, attributes(52).toDouble, attributes(53).toDouble,
      attributes(54).toDouble, attributes(55).toDouble, attributes(56).toDouble, attributes(57).toDouble,
      attributes(58).toDouble, attributes(59).toDouble, attributes(60).toDouble,
      attributes(61).toDouble, attributes(62).toDouble, attributes(63).toDouble, attributes(64).toDouble,
      attributes(65).toDouble, attributes(66).toDouble, attributes(67).toDouble, attributes(68).toDouble,
      attributes(69).toDouble, attributes(70).toDouble, attributes(71).toDouble, attributes(72).toDouble,
      attributes(73).toDouble,attributes(74).toDouble,attributes(75).toDouble)).toDF()
    features_labelDF
	}

	def merge(srcPath: String, dstPath: String): Unit = {
    val hadoopConfig = new Configuration()
    val hdfs = FileSystem.get(hadoopConfig)
    FileUtil.copyMerge(hdfs, new Path(srcPath), hdfs, new Path(dstPath), true, hadoopConfig, null)
    // the "true" setting deletes the source files once they are merged into the new output
  }

  def method(params:Params): Unit = {
  	val spark = SparkSession.builder.appName("RepeatBuyers").config("spark.master", "local").getOrCreate()
    import spark.implicits._

  	val features_labelDF = FeaturesProcess(params.trainInput,"train_temp",params.logInput,params.infoInput,params.outputPath)
  	 val ass_names = Array("user_id","seller_id","u_m_c_cnt","u_m_p_cnt","u_m_p_i_cnt","u_m_r_cnt","u_m_f_cnt","um_act_d","avg_span_act_um",
    "total_action_count","c_ratio","p_ratio","f_ratio","ttl_c","ttl_p","ttl_f","dis_c","dis_p","dis_f","mer_c","mer_p",
    "mer_f","cat_c","cat_p","cat_f","brand_c","brand_p", "brand_f","ttl_act_d","u_c_d_cnt","u_p_d_cnt","u_f_d_cnt",
    "u_p_m_ratio","avg_span_actions","avg_span_purchase","last_span","may_c_cnt","may_p_cnt",
    "may_f_cnt","jun_c_cnt","jun_p_cnt","jun_f_cnt","jul_c_cnt","jul_p_cnt","jul_f_cnt", "aug_c_cnt", "aug_p_cnt",
    "aug_f_cnt","sep_c_cnt","sep_p_cnt","sep_f_cnt","oct_c_cnt","oct_p_cnt","oct_f_cnt","nov_c_cnt","nov_p_cnt",
    "nov_f_cnt","may_tt","jun_tt","jul_tt","aug_tt","sep_tt","oct_tt","nov_tt","ttl_item","ttl_cat","ttl_brand",
    "c_cnt","p_cnt","f_cnt","c_u_cnt","p_u_cnt","f_u_cnt","age_range","gender","label")

  val assembler = new VectorAssembler().setInputCols(ass_names).setOutputCol("features_temp")
  val normalizer = new Normalizer().setInputCol("features_temp").setOutputCol("features")

  val splits = features_labelDF.randomSplit(Array(0.8, 0.2), seed = 504)
  val train = splits(0).cache()
  val test = splits(1).cache()

  //  // Run training algorithm to build the model
 //  // models
  val rf = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
  val pipeline_rf = new Pipeline().setStages(Array(assembler, normalizer, rf))
  val paramGrid_rf = new ParamGridBuilder()
    .addGrid(rf.numTrees, Seq(300, 400, 500))
    .addGrid(rf.maxDepth, Seq(5, 7, 9))
    .build()
  val cv_rf = new CrossValidator()
    .setEstimator(pipeline_rf)
    .setEvaluator(new BinaryClassificationEvaluator())
    .setEstimatorParamMaps(paramGrid_rf)
    .setNumFolds(5)

  val lr = new LogisticRegression()
    .setLabelCol("label")
  val pipeline_lr = new Pipeline().setStages(Array(assembler, normalizer, lr))
  val paramGrid_lr = new ParamGridBuilder()
    .addGrid(lr.threshold, Seq(0.3, 0.5, 0.7))
    .addGrid(lr.maxIter, Seq(7, 10, 13))
    .build()
  val cv_lr = new CrossValidator()
    .setEstimator(pipeline_lr)
    .setEvaluator(new BinaryClassificationEvaluator())
    .setEstimatorParamMaps(paramGrid_lr)
    .setNumFolds(5)

  val dt = new DecisionTreeClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
  val pipeline_dt = new Pipeline().setStages(Array(assembler, normalizer, dt))
  val paramGrid_dt = new ParamGridBuilder()
    .addGrid(dt.maxBins, Seq(40, 50, 60))
    .addGrid(dt.maxDepth, Seq(5, 7, 9))
    .build()
  val cv_dt = new CrossValidator()
    .setEstimator(pipeline_dt)
    .setEvaluator(new BinaryClassificationEvaluator())
    .setEstimatorParamMaps(paramGrid_dt)
    .setNumFolds(5)

  val gbt = new GBTClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(10)
  val pipeline_gbt = new Pipeline().setStages(Array(assembler, normalizer, gbt))
  val paramGrid_gbt = new ParamGridBuilder()
    .addGrid(gbt.maxDepth, Seq(5, 7, 9))
    .addGrid(gbt.maxIter, Seq(5, 10))
    .build()
  val cv_gbt = new CrossValidator()
    .setEstimator(pipeline_gbt)
    .setEvaluator(new BinaryClassificationEvaluator())
    .setEstimatorParamMaps(paramGrid_gbt)
    .setNumFolds(5)

  // evaluation
  val cvModel_rf = cv_rf.fit(train)
  val predictions_rf = cvModel_rf.transform(test)
  val trainPredictionsAndLabels_rf = cvModel_rf.transform(train).select("label", "prediction")
    .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
  val testPredictionsAndLabels_rf = cvModel_rf.transform(test).select("label", "prediction")
    .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
  println("Random Forest")
  evaluation(params.outputPath, predictions_rf, "result_rf.txt", trainPredictionsAndLabels_rf, testPredictionsAndLabels_rf)

  val cvModel_lr = cv_lr.fit(train)
  val predictions_lr = cvModel_lr.transform(test)
  val trainPredictionsAndLabels_lr = cvModel_rf.transform(train).select("label", "prediction")
    .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
  val testPredictionsAndLabels_lr = cvModel_rf.transform(test).select("label", "prediction")
    .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
  println("Logistic Regression")
  evaluation(params.outputPath, predictions_lr, "result_lr", trainPredictionsAndLabels_lr, testPredictionsAndLabels_lr)

  val cvModel_dt = cv_dt.fit(train)
  val predictions_dt = cvModel_dt.transform(test)
  val trainPredictionsAndLabels_dt = cvModel_rf.transform(train).select("label", "prediction")
    .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
  val testPredictionsAndLabels_dt = cvModel_rf.transform(test).select("label", "prediction")
    .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
  println("Decision tree")
  evaluation(params.outputPath, predictions_dt, "result_dt", trainPredictionsAndLabels_dt, testPredictionsAndLabels_dt)

  val cvModel_gbt = cv_gbt.fit(train)
  val predictions_gbt = cvModel_gbt.transform(test)
  val trainPredictionsAndLabels_gbt = cvModel_rf.transform(train).select("label", "prediction")
    .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
  val testPredictionsAndLabels_gbt = cvModel_rf.transform(test).select("label", "prediction")
    .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
  println("GBT")
  evaluation(params.outputPath,predictions_gbt, "result_gbt", trainPredictionsAndLabels_gbt, testPredictionsAndLabels_gbt)

  // predict test data set and save probability
  val test_df = FeaturesProcess(params.trainInput,"test_temp",params.logInput,params.infoInput,params.outputPath)

  val test_re_rf = cvModel_rf.transform(test_df)
  val tianchi_rf = test_re_rf.select("user_id", "seller_id", "prediction", "probability").rdd

  val test_re_lr = cvModel_lr.transform(test_df)
  val tianchi_lr = test_re_lr.select("user_id", "seller_id", "prediction", "probability").rdd

  val test_re_dt = cvModel_dt.transform(test_df)
  val tianchi_dt = test_re_dt.select("user_id", "seller_id", "prediction", "probability").rdd

  val test_re_gbt = cvModel_gbt.transform(test_df)
  val tianchi_gbt = test_re_gbt.select("user_id", "seller_id", "prediction", "probability").rdd

    writeMerge("test_rf", tianchi_rf, params.outputPath)
    writeMerge("test_lr", tianchi_lr, params.outputPath)
    writeMerge("test_dt", tianchi_dt, params.outputPath)
    writeMerge("test_gbt", tianchi_gbt, params.outputPath)
  }

  def evaluation(output:String,result: DataFrame, resultFile: String, trainMetrics: RDD[(Double, Double)], testMetrics: RDD[(Double, Double)]): Unit = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(result)
    val evaluatorBinary = new BinaryClassificationEvaluator().setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")
    val roc = evaluatorBinary.evaluate(result)
    val evaluator1 = new MulticlassClassificationEvaluator()
      .setLabelCol("label").setPredictionCol("prediction")
      .setMetricName("weightedPrecision")
    val p = evaluator1.evaluate(result)
    val evaluator2 = new MulticlassClassificationEvaluator()
      .setLabelCol("label").setPredictionCol("prediction")
      .setMetricName("weightedRecall")
    val s = evaluator2.evaluate(result)
    val evaluator3 = new MulticlassClassificationEvaluator()
      .setLabelCol("label").setPredictionCol("prediction")
      .setMetricName("f1")
    val f = evaluator3.evaluate(result)

    val metrics_train = new BinaryClassificationMetrics(trainMetrics)
    val metrics_test = new BinaryClassificationMetrics(testMetrics)

    // Precision by threshold
    val precision = metrics_train.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }
    // Recall by threshold
    val recall = metrics_train.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }
    // F-measure
    val f1Score = metrics_train.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }
    val beta = 0.5
    val fScore = metrics_train.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }
    val precision2 = metrics_test.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }
    // Recall by threshold
    val recall2 = metrics_test.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }
    // F-measure
    val f1Score2 = metrics_test.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }
    val fScore2 = metrics_test.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    val output = "\n=====================================================================\n" +
      s"Training data accuracy = ${accuracy}\n" +
      s"Training data roc = ${roc}\n" +
      s"Training data weightedPrecision = ${p}\n" +
      s"Training data weightedRecall = ${s}\n" +
      s"Training data F1_Score = ${f}\n" +
      s"Training data Area under ROC = ${metrics_train.areaUnderROC}\n" +
      s"Training data Area under precision-recall curve = ${metrics_train.areaUnderPR}\n" +
      "=====================================================================\n" +
      s"Test data Area under ROC = ${metrics_test.areaUnderROC}\n" +
      s"Test data Area under precision-recall curve = ${metrics_test.areaUnderPR}\n" +
      "=====================================================================\n"

    println(output)
    import java.io._
    val file = output + resultFile
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)))
    writer.write(output) // however you want to format it
    writer.close()
  }

  def writeMerge(filename: String, resutls: RDD[Row], outputfile0:String): Unit = {
      val filename0 = filename
      val outputFileName0 = outputfile0 + "/temp_" + filename0
      val mergedFileName0 = outputfile0 + "/merged_" + filename0
      val mergeFindGlob0 = outputFileName0
      resutls.saveAsTextFile(outputFileName0)
      merge(mergeFindGlob0, mergedFileName0)
      resutls.unpersist()
    }

	def main(args: Array[String]){
	val parser = new OptionParser[Params]("RepeatBuyersPrediction") {
      head("RepeatBuyersPrediction", "1.0")

      opt[String]("logInput").required().valueName("<file>").action((x, c) =>
        c.copy(logInput = x)).text("Path to file for user log")

      opt[String]("infoInput").required().valueName("<file>").action((x, c) =>
        c.copy(infoInput = x)).text("Path to file for user info")

      opt[String]("trainInput").required().valueName("<file>").action((x, c) =>
        c.copy(labelInput = x)).text("Path to file for training data")

      opt[String]("testInput").required().valueName("<file>").action((x, c) =>
        c.copy(labelInput = x)).text("Path to file for test data")

      opt[String]("outputPath").valueName("<file>").action((x, c) =>
        c.copy(outputFile = x)).text("Path to output file")
    }

    //val parser = {

    	//}

    parser.parse(args, Params()) match {
      case Some(params) =>
        method(params)
      case None =>
        throw new IllegalArgumentException("One or more parameters are invalid or missing")
    }
	}
}
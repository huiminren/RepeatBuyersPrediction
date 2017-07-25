/**
  * Created by huiminren on 7/20/17.
  */

import org.apache.spark.sql.functions._
import java.text.SimpleDateFormat

import org.apache.spark.sql.functions.countDistinct
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, GBTClassifier, LinearSVC, LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.feature.{Normalizer, VectorAssembler}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.{DataFrame, SparkSession}


object FinalRB extends App {

  val spark = SparkSession.builder.appName("RepeatBuyers").config("spark.master", "local").getOrCreate()

  LogManager.getRootLogger().setLevel(Level.ERROR)

  val path = "/Users/huiminren/Google Drive/DS504/Final Project/Data/data_format1/"
  //  val user_log_df = spark.read.option("header","true").option("inferSchema", "true").csv(path+"user_log_format1.csv")
  //  val user_log_sample = spark.read.option("header","true").option("inferSchema", "true").csv(path+"user_log_sample.csv")

  val user_log_df0 = spark.read.option("header", "true").csv(path + "user_log_sample.csv") //user_log_sample.csv,user_log_format1.csv

  import spark.implicits._

  // Convert "time_stamp" from String to Date
  val dateToTimeStamp = udf((date: String) => {
    val stringDate = "2014" + "/" + date.substring(0, 2) + "/" + date.substring(2, 4)
    val format = new SimpleDateFormat("yyyy-MM-dd")
    format.format(new SimpleDateFormat("yyy/MM/dd").parse(stringDate))
  })
  val user_log_sample = user_log_df0.withColumn("time_stamp", dateToTimeStamp($"time_stamp"))

  // Basic
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
  //  val oldest_active_day = user_log_sample.groupBy("user_id").agg(max("time_stamp") as "oldest_act_d")//？？？？？？？？

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
  //  val user_merchant_oldest_days = user_log_sample.groupBy("user_id", "seller_id").agg(max("time_stamp") as "um_oldest_d")

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
    .join(active_days, Seq("user_id"), "outer")
    .join(avg_span_act_df, Seq("user_id"), "outer")
    .join(avg_span_purchase_df, Seq("user_id"), "outer")
    .join(last_span, Seq("user_id"), "outer")


  // month

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
  val jun_p = jun.where("action_type = 0").groupBy("user_id").agg(count("user_id") as "jun_p_cnt")
  val jul_p = jul.where("action_type = 2").groupBy("user_id").agg(count("user_id") as "jul_p_cnt")
  val aug_p = aug.where("action_type = 2").groupBy("user_id").agg(count("user_id") as "aug_p_cnt")
  val sep_p = sep.where("action_type = 2").groupBy("user_id").agg(count("user_id") as "sep_p_cnt")
  val oct_p = oct.where("action_type = 2").groupBy("user_id").agg(count("user_id") as "oct_p_cnt")
  val nov_p = nov.where("action_type = 2").groupBy("user_id").agg(count("user_id") as "nov_p_cnt")
  val may_f = may.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "may_f_cnt")
  val jun_f = jun.where("action_type = 0").groupBy("user_id").agg(count("user_id") as "jun_f_cnt")
  val jul_f = jul.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "jul_f_cnt")
  val aug_f = aug.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "aug_f_cnt")
  val sep_f = sep.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "sep_f_cnt")
  val oct_f = oct.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "oct_f_cnt")
  val nov_f = nov.where("action_type = 3").groupBy("user_id").agg(count("user_id") as "nov_f_cnt")

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

  val train_df0 = spark.read.option("header", "true").csv(path + "label11.csv")//train_format1.csv
  //train_format1.csv
  val newNames = Seq("user_id", "seller_id", "label")
  val train_df = train_df0.toDF(newNames: _*)

  val user_info_df0 = spark.read.option("header", "true").csv(path + "user_info_format1.csv")
  val ui_df1 = user_info_df0.na.fill("0", Seq("age_range"))
  val user_info_df = ui_df1.na.fill("2", Seq("gender"))

  val all_features = um_user_seller.join(user_info_df, Seq("user_id"), "left_outer")
  val train_table00 = all_features.na.fill(0)
  val train_table = train_df.join(train_table00, Seq("user_id", "seller_id")) //inner join --> left join


  val train_table123 = train_table.select(Seq("user_id", "seller_id", "u_m_c_cnt", "u_m_p_cnt", "u_m_f_cnt", "um_act_d", "avg_span_act_um",
    "c_ratio", "p_ratio", "f_ratio", "ttl_c", "ttl_p", "ttl_f", "dis_c", "dis_p", "dis_f", "mer_c", "mer_p", "mer_f", "cat_c", "cat_p",
    "cat_f", "ttl_act_d", "avg_span_actions", "avg_span_purchase", "last_span", "c_cnt", "p_cnt", "f_cnt", "c_u_cnt",
    "p_u_cnt", "f_u_cnt", "age_range", "gender", "may_c_cnt", "jun_c_cnt", "jul_c_cnt", "aug_c_cnt", "sep_c_cnt", "oct_c_cnt",
    "nov_c_cnt", "may_p_cnt", "jun_p_cnt", "jul_p_cnt", "aug_p_cnt", "sep_p_cnt", "oct_p_cnt", "nov_p_cnt", "may_f_cnt", "jun_f_cnt",
    "jul_f_cnt", "aug_f_cnt", "sep_f_cnt", "oct_f_cnt", "nov_f_cnt", "label").map(
    c => col(c).cast("double")
  ): _*)

  //  val features0 = Array("user_id","seller_id","u_m_c_cnt","u_m_p_cnt","u_m_f_cnt","um_act_d","avg_span_act_um",
  //    "c_ratio","p_ratio","f_ratio","ttl_c","ttl_p","ttl_f","dis_c","dis_p","dis_f","mer_c","mer_p","mer_f","cat_c","cat_p",
  //    "cat_f","ttl_act_d","avg_span_actions","avg_span_purchase","last_span","c_cnt","p_cnt","f_cnt","c_u_cnt",
  //    "p_u_cnt","f_u_cnt","age_range","gender") //"um_oldest_d","oldest_act_d",
  //
  val assembler = new VectorAssembler().setInputCols(Array("user_id", "seller_id", "u_m_c_cnt", "u_m_p_cnt", "u_m_f_cnt",
    "um_act_d", "avg_span_act_um", "c_ratio", "p_ratio", "f_ratio", "ttl_c", "ttl_p", "ttl_f", "dis_c", "dis_p", "dis_f",
    "mer_c", "mer_p", "mer_f", "cat_c", "cat_p", "cat_f", "ttl_act_d", "avg_span_actions", "avg_span_purchase", "last_span",
    "c_cnt", "p_cnt", "f_cnt", "c_u_cnt", "p_u_cnt", "f_u_cnt", "age_range", "gender", "may_c_cnt", "jun_c_cnt", "jul_c_cnt",
    "aug_c_cnt", "sep_c_cnt", "oct_c_cnt", "nov_c_cnt", "may_p_cnt", "jun_p_cnt", "jul_p_cnt", "aug_p_cnt", "sep_p_cnt",
    "oct_p_cnt", "nov_p_cnt", "may_f_cnt", "jun_f_cnt", "jul_f_cnt", "aug_f_cnt", "sep_f_cnt", "oct_f_cnt", "nov_f_cnt"))
    .setOutputCol("features_temp")

  val normalizer = new Normalizer().setInputCol("features_temp").setOutputCol("features")

  val lr = new LogisticRegression().setMaxIter(10)
  lr.setLabelCol("label")

  val dt = new DecisionTreeClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")

  val rf = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setNumTrees(10) //????

  val gbt = new GBTClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(10)

  val lsvc = new LinearSVC()
    .setMaxIter(10)
    .setRegParam(0.1)

  val pipeline_lr = new Pipeline().setStages(Array(assembler, normalizer, lr))
  val pipeline_dt = new Pipeline().setStages(Array(assembler, normalizer, dt))
  val pipeline_rf = new Pipeline().setStages(Array(assembler, normalizer, rf))
  val pipeline_gbt = new Pipeline().setStages(Array(assembler, normalizer, gbt))
  val pipeline_lsvc = new Pipeline().setStages(Array(assembler, normalizer, lsvc))

  val splits0 = train_table123.randomSplit(Array(0.01, 0.99), seed = 504)
  val small = splits0(0).cache()

  val splits = small.randomSplit(Array(0.8, 0.2), seed = 504)
  val train = splits(0).cache()
  val test = splits(1).cache()

  val final_test0 = spark.read.option("header", "true").csv(path + "test_format1.csv")
  val final_test1 = final_test0.toDF(newNames: _*)
  val final_test2 = final_test1.join(train_table00, Seq("user_id", "seller_id"))
  //inner --> left_outer
  val final_test3 = final_test2.select(Seq("user_id", "seller_id", "u_m_c_cnt", "u_m_p_cnt", "u_m_f_cnt", "um_act_d", "avg_span_act_um",
    "c_ratio", "p_ratio", "f_ratio", "ttl_c", "ttl_p", "ttl_f", "dis_c", "dis_p", "dis_f", "mer_c", "mer_p", "mer_f", "cat_c", "cat_p",
    "cat_f", "ttl_act_d", "avg_span_actions", "avg_span_purchase", "last_span", "c_cnt", "p_cnt", "f_cnt", "c_u_cnt",
    "p_u_cnt", "f_u_cnt", "age_range", "gender", "may_c_cnt", "jun_c_cnt", "jul_c_cnt", "aug_c_cnt", "sep_c_cnt", "oct_c_cnt",
    "nov_c_cnt", "may_p_cnt", "jun_p_cnt", "jul_p_cnt", "aug_p_cnt", "sep_p_cnt", "oct_p_cnt", "nov_p_cnt", "may_f_cnt", "jun_f_cnt",
    "jul_f_cnt", "aug_f_cnt", "sep_f_cnt", "oct_f_cnt", "nov_f_cnt", "label").map(
    c => col(c).cast("double")
  ): _*)

  val final_test4 = final_test3.randomSplit(Array(0.01, 0.99), seed = 11L)
  val final_test = final_test4(0).cache()

  var model_lr = pipeline_lr.fit(train)
  var result_lr = model_lr.transform(final_test) // test
  result_lr = result_lr.select("prediction", "label") //probability

  var model_dt = pipeline_dt.fit(train)
  var result_dt = model_dt.transform(final_test) // test
  result_dt = result_dt.select("prediction", "label") //probability

  var model_rf = pipeline_rf.fit(train)
  var result_rf = model_rf.transform(final_test) // test
  result_rf = result_rf.select("prediction", "label") //probability

  var model_gbt = pipeline_gbt.fit(train)
  var result_gbt = model_gbt.transform(final_test) // test
  result_gbt = result_gbt.select("prediction", "label") //probability

  var model_lsvc = pipeline_lsvc.fit(train)
  var result_lsvc = model_lsvc.transform(final_test) // test
  result_lsvc = result_lsvc.select("prediction", "label") //probability

  def evaluation(result: DataFrame): Unit = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(result)
    println("Test Error = " + (1 - accuracy))

    val evaluatorBinary = new BinaryClassificationEvaluator().setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")
    val roc = evaluatorBinary.evaluate(result)
    println("roc = " + roc)

    val evaluator1 = new MulticlassClassificationEvaluator()
      .setLabelCol("label").setPredictionCol("prediction")
      .setMetricName("weightedPrecision")
    val p = evaluator1.evaluate(result)
    println("weightedPrecision = " + p)

    val evaluator2 = new MulticlassClassificationEvaluator()
      .setLabelCol("label").setPredictionCol("prediction")
      .setMetricName("weightedRecall")
    val s = evaluator2.evaluate(result)
    println("weightedRecall = " + s)

    val evaluator3 = new MulticlassClassificationEvaluator()
      .setLabelCol("label").setPredictionCol("prediction")
      .setMetricName("f1")
    val f = evaluator3.evaluate(result)
    println("F1_Score = " + f)
  }

  println("LR")
  val lr_eva = evaluation(result_lr)
  println("DT")
  val dt_eva = evaluation(result_dt)
  println("RF")
  val rf_eva = evaluation(result_rf)
  println("GBT")
  val gbt_eva = evaluation(result_gbt)
  println("LSVC")
  val lsvc_eva = evaluation(result_lsvc)
}

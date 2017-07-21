/**
  * Created by huiminren on 7/20/17.
  */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import java.text.SimpleDateFormat
import org.apache.spark.sql.functions.countDistinct

import org.apache.log4j.{Level, LogManager}

import org.apache.spark.sql.SparkSession

object FinalRB extends App{

  val spark = SparkSession.builder.appName("RepeatBuyers").config("spark.master", "local").getOrCreate()

  LogManager.getRootLogger().setLevel(Level.ERROR)

  val path = "/Users/huiminren/Google Drive/DS504/Final Project/Data/data_format1/"
  val user_info_df = spark.read.option("header","true").csv(path+"user_info_format1.csv")
  val user_log_df = spark.read.option("header","true").csv(path+"user_log_format1.csv")
//  val user_log_sample = spark.read.option("header","true").csv(path+"user_log_sample.csv")

  val user_log_df0 = spark.read.option("header","true").csv(path+"user_log_sample.csv")

  import spark.implicits._

  // Convert "time_stamp" from String to Date
  val dateToTimeStamp = udf((date: String) => {
    val stringDate = "2014"+"/"+date.substring(0,2)+"/"+date.substring(2,4)
    val format = new SimpleDateFormat("yyyy-MM-dd")
    format.format(new SimpleDateFormat("yyy/MM/dd").parse(stringDate))
  })
  val user_log_sample = user_log_df0.withColumn("time_stamp", dateToTimeStamp($"time_stamp"))
//  user_log_sample.show()

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
  val oldest_active_day = user_log_sample.groupBy("user_id").agg(max("time_stamp") as "oldest_act_d")


  val merchant_total_click = click.groupBy("seller_id").agg(count("seller_id") as "c_cnt")
  val merchant_total_purchase = purchase.groupBy("seller_id").agg(count("seller_id") as "p_cnt")
  val merchant_total_favor = favorite.groupBy("seller_id").agg(count("seller_id") as "f_cnt")
  val user_count_click_merchant = click.groupBy("seller_id").agg(countDistinct("user_id") as "c_u_cnt")
  val user_count_purchase_merchant = purchase.groupBy("seller_id").agg(countDistinct("user_id") as "p_u_cnt")
  val user_count_favor_merchant = favorite.groupBy("seller_id").agg(countDistinct("user_id") as "f_u_cnt")


  val user_merchant_click = click.groupBy("user_id","seller_id").agg(count("user_id") as "u_m_c_cnt")
  val user_merchant_purchase = purchase.groupBy("user_id","seller_id").agg(count("user_id") as "u_m_p_cnt")
  val user_merchant_favor = favorite.groupBy("user_id","seller_id").agg(count("user_id") as "u_m_f_cnt")
  val user_merchant_active_days = user_log_sample.groupBy("user_id","seller_id").agg(countDistinct("time_stamp") as "um_act_d")
  val user_merchant_oldest_days = user_log_sample.groupBy("user_id","seller_id").agg(max("time_stamp") as "um_oldest_d")


  user_click.createOrReplaceTempView("UserClick")
  user_purchase.createOrReplaceTempView("UserPurchase")
  user_favor.createOrReplaceTempView("UserFavorite")

  val user_total_action = spark.sql("SELECT UserClick.user_id, UserClick.ttl_c, UserPurchase.ttl_p, UserFavorite.ttl_f " +
    "FROM UserClick, UserPurchase, UserFavorite where UserClick.user_id = UserPurchase.user_id and " +
    "UserPurchase.user_id = UserFavorite.user_id")
  spark.catalog.dropTempView("UserClick")
  spark.catalog.dropTempView("UserPurchase")
  spark.catalog.dropTempView("UserFavor")

  item_click.createOrReplaceTempView("ItemClick")
  item_purchase.createOrReplaceTempView("ItemPurchase")
  item_favor.createOrReplaceTempView("ItemFavor")
  val item = spark.sql("SELECT ItemClick.user_id, ItemClick.dis_c, ItemPurchase.dis_p, ItemFavor.dis_f FROM ItemClick, " +
    "ItemPurchase, ItemFavor where ItemClick.user_id = ItemPurchase.user_id and ItemPurchase.user_id = ItemFavor.user_id")
  spark.catalog.dropTempView("ItemClick")
  spark.catalog.dropTempView("ItemPurchase")
  spark.catalog.dropTempView("ItemFavor")

  item.createOrReplaceTempView("item_info")
  user_total_action.createOrReplaceTempView("user_total")
  val join1 = spark.sql("SELECT item_info.user_id, item_info.dis_c, item_info.dis_p, item_info.dis_f, user_total.ttl_c, " +
    "user_total.ttl_p, user_total.ttl_f from item_info, user_total where item_info.user_id = user_total.user_id")
  spark.catalog.dropTempView("item_info")
  spark.catalog.dropTempView("user_total")

  merchants_click.createOrReplaceTempView("MerClick")
  merchants_purchase.createOrReplaceTempView("MerPurchase")
  merchants_favor.createOrReplaceTempView("MerFavor")
  val merchant_info = spark.sql("select MerClick.user_id, MerClick.mer_c, MerPurchase.mer_p, MerFavor.mer_f " +
    "from MerClick, MerPurchase, MerFavor where MerClick.user_id = MerPurchase.user_id and " +
    "MerPurchase.user_id = MerFavor.user_id")
  spark.catalog.dropTempView("MerPurchase")
  spark.catalog.dropTempView("MerFavor")
  spark.catalog.dropTempView("MerClick")

  cat_click.createOrReplaceTempView("catClick")
  cat_purchase.createOrReplaceTempView("catPurchase")
  cat_favor.createOrReplaceTempView("catFavor")
  val cat_info = spark.sql("select catClick.user_id, catClick.cat_c, catPurchase.cat_p, catFavor.cat_f " +
    "from catClick, catPurchase, catFavor " +
    "where catClick.user_id = catPurchase.user_id and catPurchase.user_id = catFavor.user_id")
  spark.catalog.dropTempView("catPurchase")
  spark.catalog.dropTempView("catFavor")
  spark.catalog.dropTempView("catClick")

  merchant_info.createOrReplaceTempView("merchant")
  join1.createOrReplaceTempView("join_one")
  cat_info.createOrReplaceTempView("category")
  val join_two = spark.sql("select join_one.user_id, join_one.dis_c, join_one.dis_p, join_one.dis_f, join_one.ttl_c, " +
    "join_one.ttl_p, join_one.ttl_f, merchant.mer_c, merchant.mer_p, merchant.mer_f, category.cat_c, category.cat_p, " +
    "category.cat_f " +
    "from join_one, merchant, category " +
    "where join_one.user_id = merchant.user_id and " +
    "merchant.user_id = category.user_id")
  join_two.createOrReplaceTempView("join2") //?????????????? not used


  active_days.createOrReplaceTempView("active")
  oldest_active_day.createOrReplaceTempView("oldest")

  val join_three = spark.sql("select join_one.user_id, join_one.dis_c, join_one.dis_p, join_one.dis_f, join_one.ttl_c, " +
    "join_one.ttl_p, join_one.ttl_f, merchant.mer_c, merchant.mer_p, merchant.mer_f, category.cat_c, category.cat_p, " +
    "category.cat_f, active.ttl_act_d, oldest.oldest_act_d " +
    "from join_one, merchant, category, active, oldest " +
    "where join_one.user_id = merchant.user_id and merchant.user_id = category.user_id and " +
    "category.user_id = active.user_id and active.user_id = oldest.user_id ")

  merchant_total_click.createOrReplaceTempView("total_c")
  merchant_total_purchase.createOrReplaceTempView("total_p")
  merchant_total_favor.createOrReplaceTempView("total_f")
  user_count_click_merchant.createOrReplaceTempView("count_c")
  user_count_purchase_merchant.createOrReplaceTempView("count_p")
  user_count_favor_merchant.createOrReplaceTempView("count_f")

  val seller_info = spark.sql("select total_c.seller_id, total_c.c_cnt, total_p.p_cnt, total_f.f_cnt, count_c.c_u_cnt, " +
    "count_p.p_u_cnt, count_f.f_u_cnt from total_c, total_p, total_f, count_c, count_p, count_f " +
    "where total_c.seller_id = total_p.seller_id and total_p.seller_id = total_f.seller_id and " +
    "total_f.seller_id = count_c.seller_id and count_c.seller_id = count_p.seller_id and " +
    "count_p.seller_id = count_f.seller_id ")
  spark.catalog.dropTempView("total_c")
  spark.catalog.dropTempView("total_p")
  spark.catalog.dropTempView("total_f")
  spark.catalog.dropTempView("count_c")
  spark.catalog.dropTempView("count_p")
  spark.catalog.dropTempView("count_f")

  val a1 = user_log_sample.groupBy("user_id").agg(max("time_stamp") as "max_date").toDF()
  val a2 = user_log_sample.groupBy("user_id").agg(min("time_stamp") as "min_date").toDF()
  val a3 = user_log_sample.groupBy("user_id").agg(count("user_id") as "count").toDF()
  a1.createOrReplaceTempView("a1")
  a2.createOrReplaceTempView("a2")
  a3.createOrReplaceTempView("a3")
  val a4 = spark.sql("select a1.user_id, max_date,min_date,count from a1,a2,a3 " +
    "where a1.user_id = a2.user_id and a1.user_id = a3.user_id order by a1.user_id")
  val avg_span_act_df = a4.withColumn("avg_span_actions",(datediff($"max_date",$"min_date"))/$"count")
  spark.catalog.dropTempView("a1")
  spark.catalog.dropTempView("a2")
  spark.catalog.dropTempView("a3")

  // Average span between 2 purchases
  user_log_sample.createOrReplaceTempView("user_log")
  val b1 = spark.sql("SELECT * FROM user_log where action_type = 2 order by user_id, time_stamp")
  val b2 = b1.groupBy("user_id").agg(max("time_stamp") as "max_date").toDF()
  val b3 = b1.groupBy("user_id").agg(min("time_stamp") as "min_date").toDF()
  val b4 = b1.groupBy("user_id").agg(count("user_id") as "count").toDF()

  b2.createOrReplaceTempView("b2")
  b3.createOrReplaceTempView("b3")
  b4.createOrReplaceTempView("b4")
  val b5 = spark.sql("select b2.user_id, max_date,min_date,count from b4,b2,b3 " +
    "where b4.user_id = b2.user_id and b4.user_id=b3.user_id order by b4.user_id")
  val avg_span_purchase_df = b5.withColumn("avg_span_purchase",(datediff($"max_date",$"min_date"))/$"count")
  spark.catalog.dropTempView("b2")
  spark.catalog.dropTempView("b3")
  spark.catalog.dropTempView("b4")

  // How many days since the last purchase
  val c1 = spark.sql("select *,'2014-11-11' as double_11 from user_log " +
    "where time_stamp<'2014-11-11' and action_type = 2 order by user_id")
  val c2 = c1.groupBy("user_id").agg(max("time_stamp") as "max2_date", max("double_11") as "double_11").toDF()
  val last_span = c2.withColumn("last_span", datediff($"double_11",$"max2_date"))

  /*
  Calculate user-merchant/category/brand/item repeat features
  */
  // average span between any 2 actions for one merchant
  val d1 = user_log_sample.groupBy("user_id", "seller_id").agg(max("time_stamp") as "max_date")
  val d2 = user_log_sample.groupBy("user_id", "seller_id").agg(min("time_stamp") as "min_date")
  val d3 = user_log_sample.groupBy("user_id", "seller_id").agg(count("user_id") as "cnt_um")
  d1.createOrReplaceTempView("d1")
  d2.createOrReplaceTempView("d2")
  d3.createOrReplaceTempView("d3")

  val d4 = spark.sql("select d1.user_id, d1.seller_id, max_date,min_date,cnt_um from d1,d2,d3 " +
    "where d1.user_id = d2.user_id and d1.user_id = d3.user_id and d1.seller_id = d2.seller_id and " +
    "d1.seller_id = d3.seller_id order by d1.user_id")
  val avg_span_act_um_df = d4.withColumn("avg_span_act_um",(datediff($"max_date",$"min_date"))/$"cnt_um")
  spark.catalog.dropTempView("d1")
  spark.catalog.dropTempView("d2")
  spark.catalog.dropTempView("d3")

  avg_span_act_df.createOrReplaceTempView("view1")
  avg_span_purchase_df.createOrReplaceTempView("view2")
  last_span.createOrReplaceTempView("view3")
  avg_span_act_um_df.createOrReplaceTempView("view4")
  val user = spark.sql("select join_one.user_id, join_one.dis_c, join_one.dis_p, join_one.dis_f, join_one.ttl_c, " +
    "join_one.ttl_p, join_one.ttl_f, merchant.mer_c, merchant.mer_p, merchant.mer_f, category.cat_c, " +
    "category.cat_p, category.cat_f, active.ttl_act_d, oldest.oldest_act_d, view1.avg_span_actions, " +
    "view2.avg_span_purchase, view3.last_span from join_one, merchant, category, active, oldest, view1, view2, view3 " +
    "where join_one.user_id = merchant.user_id and merchant.user_id = category.user_id and category.user_id = active.user_id " +
    "and active.user_id = oldest.user_id and oldest.user_id = view1.user_id and view1.user_id = view2.user_id " +
    "and view2.user_id = view3.user_id")
  spark.catalog.dropTempView("join_one")
  spark.catalog.dropTempView("merchant")
  spark.catalog.dropTempView("category")
  spark.catalog.dropTempView("active")
  spark.catalog.dropTempView("oldest")
  spark.catalog.dropTempView("view1")
  spark.catalog.dropTempView("view2")
  spark.catalog.dropTempView("view3")


  user_merchant_click.createOrReplaceTempView("table1")
  user_merchant_purchase.createOrReplaceTempView("table2")
  user_merchant_favor.createOrReplaceTempView("table3")
  user_merchant_active_days.createOrReplaceTempView("table4")
  user_merchant_oldest_days.createOrReplaceTempView("table5")

  val total_action = user_log_sample.groupBy("user_id","seller_id").agg(count("user_id") as "total_action_count")

  total_action.createOrReplaceTempView("action_count")
  val temp = spark.sql("SELECT action_count.user_id, action_count.seller_id, action_count.total_action_count, " +
    "table1.u_m_c_cnt FROM action_count, table1 where action_count.user_id = table1.user_id and " +
    "action_count.seller_id = table1.seller_id")
  val click_ratio = temp.withColumn("c_ratio", temp("u_m_c_cnt")/ temp("total_action_count"))
  val temp2 = temp.select($"u_m_c_cnt"/$"total_action_count")

  val temp3 = spark.sql("SELECT action_count.user_id, action_count.seller_id, action_count.total_action_count, " +
    "table2.u_m_p_cnt FROM action_count, table2 where action_count.user_id = table2.user_id and " +
    "action_count.seller_id = table2.seller_id")
  val purchase_ratio = temp3.withColumn("p_ratio", temp3("u_m_p_cnt")/ temp3("total_action_count"))

  val temp4 = spark.sql("SELECT action_count.user_id, action_count.seller_id, action_count.total_action_count," +
    " table3.u_m_f_cnt FROM action_count, table3 where action_count.user_id = table3.user_id and " +
    "action_count.seller_id = table3.seller_id")
  val favor_ratio = temp4.withColumn("f_ratio", temp4("u_m_f_cnt")/ temp4("total_action_count"))
  spark.catalog.dropTempView("action_count")

  click_ratio.createOrReplaceTempView("cr")
  purchase_ratio.createOrReplaceTempView("pr")
  favor_ratio.createOrReplaceTempView("fr")
  val join_four = spark.sql("select cr.user_id, cr.seller_id, cr.c_ratio, pr.p_ratio, fr.f_ratio from cr, pr, fr " +
    "where cr.user_id = pr.user_id and pr.user_id = fr.user_id and cr.seller_id = pr.seller_id" +
    " and pr.seller_id = fr.seller_id")
  join_four.createOrReplaceTempView("join4")
  spark.catalog.dropTempView("cr")
  spark.catalog.dropTempView("pr")
  spark.catalog.dropTempView("fr")

  val combined = spark.sql("select table1.user_id, table1.seller_id, table1.u_m_c_cnt, table2.u_m_p_cnt, " +
    "table3.u_m_f_cnt, table4.um_act_d, table5.um_oldest_d, view4.avg_span_act_um, join4.c_ratio, join4.p_ratio, " +
    "join4.f_ratio from table1, table2, table3, table4, table5, view4, join4 where table1.user_id = table2.user_id " +
    "and table1.seller_id = table2.seller_id and table2.user_id = table3.user_id and table3.user_id = table4.user_id " +
    "and table4.user_id = table5.user_id and table5.user_id = view4.user_id and view4.user_id = join4.user_id and " +
    "table2.seller_id = table3.seller_id and table3.seller_id = table4.seller_id and table4.seller_id = table5.seller_id " +
    "and table5.seller_id = view4.seller_id and view4.seller_id = join4.seller_id")
  spark.catalog.dropTempView("table1")
  spark.catalog.dropTempView("table2")
  spark.catalog.dropTempView("table3")
  spark.catalog.dropTempView("table4")
  spark.catalog.dropTempView("table5")
  spark.catalog.dropTempView("view4")
  spark.catalog.dropTempView("join4")


//  val all = spark.sql("select com.user_id, user.dis_c, user.dis_p, user.dis_f, user.ttl_c, user.ttl_p, user.ttl_f, " +
//    "user.mer_c, user.mer_p, user.mer_f, user.cat_c, user.cat_p, user.cat_f, user.ttl_act_d, user.oldest_act_d, " +
//    "user.avg_span_actions, user.avg_span_purchase, user.last_span, com.seller_id, seller_info.c_cnt, " +
//    "seller_info.p_cnt, seller_info.f_cnt, seller_info.c_u_cnt, seller_info.p_u_cnt, seller_info.f_u_cnt, " +
//    "com.seller_id, com.u_m_c_cnt, com.u_m_p_cnt, com.u_m_f_cnt, com.um_act_d, com.um_oldest_d, " +
//    "com.avg_span_act_um, com.c_ratio, com.p_ratio. com.f_ratio" +
//    "from user inner join com on user.user_id = com.user_id " +
//    "inner join seller_info on com.seller_id = seller_info.seller_id")
//  spark.catalog.dropTempView("com")
//  spark.catalog.dropTempView("user")
//  spark.catalog.dropTempView("seller_info")
//  all.show()
//  all.count()


  val all0 = combined.join(user,Seq("user_id"))
  val all1 = all0.join(seller_info,Seq("seller_id"))
  all1.printSchema()
  all1.write.format("com.databricks.spark.csv").option("header", "true").save("features")
}

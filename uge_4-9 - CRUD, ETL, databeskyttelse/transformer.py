# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 07:56:32 2025

@author: spac-30
"""

from MySQL import MySQL

## opret forbindelse til arbejdsserver
mydb=MySQL(host="localhost",
           user="root",
           password="Velkommen25")

mydb.connect()

## anvend uge8 database
mydb.use_db("uge8")

## ændre datatyper for kolonner i tabellerne
mydb.change_coltype("customers","customer_id","INT")
mydb.change_coltype("customers","zip_code","INT")

mydb.change_coltype("order_items","order_id","INT") 
mydb.change_coltype("order_items","item_id","INT") 
mydb.change_coltype("order_items","product_id","INT")
mydb.change_coltype("order_items","quantity","INT"); 
mydb.change_coltype("order_items","discount","float(10.2)")
mydb.change_coltype("order_items","list_price","float(10,2)")

mydb.change_coltype("orders","order_id","INT")
mydb.change_coltype("orders","customer_id","INT") 
mydb.change_coltype("orders","order_status","INT")
mydb.change_coltype("orders","order_date","DATE")
mydb.change_coltype("orders","required_date","DATE") 
mydb.change_coltype("orders","shipped_date","DATE")

mydb.change_coltype("staffs","manager_id","int")
mydb.change_coltype("staffs","active","int")

## ændringer i tabellerne
mydb.rename_col("stores","name","store_name")
mydb.delete_col("staffs","manager_id")

mydb.add_col("staffs","staff_name","varchar(255)")
mydb.freestyle("ALTER TABLE staffs ADD staff_name VARCHAR(255) GENERATED ALWAYS AS (CONCAT(name,' ',last_name)) STORED")


mydb.freestyle("""UPDATE orders 
               JOIN staffs ON orders.staff_name = staffs.name 
               SET orders.staff_name = staffs.staff_name""")



## tilføjer primære nøgler
mydb.add_primary_key("brands", "brand_id")
mydb.add_primary_key("categories","category_id")
mydb.add_primary_key("customers","customer_id")
mydb.add_primary_key("orders", "order_id")
mydb.add_primary_key("products","product_id")
mydb.add_primary_key("staffs","name")
mydb.add_primary_key("stores","store_name")
mydb.add_unique_key("stores","street")

## lav relationer
mydb.create_relation("orders","customers","customer_id","customer_id")
mydb.create_relation("orders_items","customers","customer_id","customer_id")
mydb.create_relation("order_items","customers","customer_id","customer_id")

mydb.create_relation("order_items","orders","order_id","order_id")
mydb.create_relation("order_items","products","product_id","product_id")
mydb.create_relation("orders","stores","store_name","store_name")

mydb.create_relation("orders","stores","store","store_name")
mydb.create_relation("orders","staffs","name","name")

mydb.create_relation("staffs","stores","store_name","store_name")
mydb.create_relation("staffs","stores","street","street")

mydb.create_relation("stocks","stores","store_name","store_name")
mydb.create_relation("products","brands","brand_id","brand_id")
mydb.create_relation("products","categories","category_id","category_id")

mydb.create_relation("stocks","products","product_id","product_id")

## afbryd forbindelse til server
mydb.disconnect()

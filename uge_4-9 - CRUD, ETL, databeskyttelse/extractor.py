# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 07:21:26 2025

@author: spac-30
"""

from MySQL import MySQL
attempt=None
try:
    mydb=MySQL(host="192.168.20.171",
                      user="curseist",
                      password="curseword",
                      port=3306)

    mydb.connect()

    mydb.backup_db("productdb","productdb.sql")

    mydb.disconnect()
except:
    attempt=False
## opret forbindelse til arbejdsserver
mydb=MySQL(host="localhost",
           user="root",
           password="Velkommen25")

mydb.connect()

## opret database, som vi arbejder på
mydb.use_db("uge8")

if attempt!=False:
    mydb.import_sql("productdb.sql")
else:
    mydb.import_sql("Data DB\prodcutdb.sql")
    
## importér tabeller fra API
mydb.import_api("https://192.168.20.171:8000/customers","customers")
mydb.import_api("https://192.168.20.171:8000/order_items","order_items")
mydb.import_api("https://192.168.20.171:8000/orders","orders")

## importér tabeller fra CSV.
mydb.import_table("Data CSV\staffs.csv")
mydb.import_table("Data CSV\stores.csv")

## afslutter forbindelse til MySQL
mydb.disconnect()

## Nu skulle alle tabellerne ligge på en lokal MySQL server, som vi kan tilgå
## og arbejde med senere. 

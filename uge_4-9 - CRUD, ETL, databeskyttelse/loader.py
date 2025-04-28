# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 07:48:09 2025

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

## eksportér uge8 database som en SQL fil
mydb.backup_db("uge8")

## afbryd forbindelse til serveren
mydb.disconnect()
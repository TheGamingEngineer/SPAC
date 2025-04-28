# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 11:12:15 2025

@author: spac-30
"""

from MySQL import MySQL

test=True


if not test:
    mydb=MySQL(host="192.168.20.171",
                      user="curseist",
                      password="curseword",
                      port=3306)
else:
    mydb=MySQL(host="localhost",
                      user="root",
                      password="Velkommen25")


mydb.connect()



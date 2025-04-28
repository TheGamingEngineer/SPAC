# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 09:09:41 2025

@author: spac-30
"""

import mysql.connector
import pandas
import sys
from tabulate import tabulate
import os
import re
import requests

## Jeg vælger at kombinere alle MySQL-funktionerne i én kommando for at gøre det mere overskueligt og lettere at håndtere forbindelsen
class MySQL:
    # definere initielt klassen med værtscomputernavn, brugernavn, adgangskode, den brugerdefinerede database og forbindelsen til MySQL
    def __init__(self,host,user,password,port=3306,db=None,connection=None):
        self.host=host ## værtscomputernavn
        self.user=user ## brugernavn
        self.password=password ## adgangskode til computeren
        self.db=db ## gemmer det anvendte database navn
        self.port=port ## opretter forbindelse via port
        self.connection=connection ## forbindelse til database
    
    # funktion til at forbinde til MySQL-serveren
    def connect(self):
        try: ## prøver at oprette forbindelse til databasen med de initielle instillinger
            self.connection=mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password)
            print("Connection Successful")
        except mysql.connector.Error as e: ## Hvis det hverken er muligt at oprette forbindelse til databasen eller at oprette databasen, rejses der en generel fejl
                print(f"Error:{e}")
                self.connection=None
    
    # funktion til at lukke forbindelsen til MySQL serveren, som også giver besked, hvis der ikke er nogen aktiv session
    def disconnect(self):
        if self.connection:
            self.connection.close()
            print("Session Closed")
        else:
            print("No Active Session")
    
    # funktion til at vælge/oprette den database, som man gerne vil bruge
    def use_db(self,db):
        if self.connection:
            ## vi tjekker om databasen findes i forvejen for at der ikke skal være fejl ved oprettelse af den. 
            ## Vi vil derfor hente en liste over eksisterende databaser på serveren. 
            db_interface=self.connection.cursor()
            db_interface.execute("SHOW DATABASES")
            db_check=[x[0] for x in db_interface.fetchall()]
            db_interface.close()
            
            ## hvis databasen findes, vil den database bare vælges.
            if db in db_check:
                print(f"Database {db} Already Exists. {db} Selected As Current Database")
            ## hvis databasen ikke findes, vil den blive oprettet før den bliver valgt. 
            else:
                self.connection.cursor().execute(f"CREATE DATABASE {db}")
                self.connection.commit()
                self.connection.cursor().close()
                print(f"Currently Using Database {db}")
            self.connection.database=db
            self.db=db
        else:
            print("No Connection Established. Start A Session First.")
    
    # funktion til at vise indhold fra tabel 'table' med mulighed for at vise alt i tabellen eller bare en kolonne via 'col'. 
    ## man kan vælge om man vil se hele tabellen eller kun første række med 'fetch'
    ## hvis man vil søge på baggrund af et filter, kan man vælge at definere core_var ("corelating variable") og core_var ("corelating value")
    def show_table(self,table,col="*",fetch="all"):
        if self.connection:
            interface=self.connection.cursor()
            
            ## hvis brugeren giver en liste af kolonnenavne, vil det blive omformet til en string, som kan anvendes i kommandoen. 
            if type(col)==list:
                col=", ".join(col)
                
            ## definer kommandoen
            command=f"SELECT {col} FROM {table}"
            ## hvis brugeren kun vil have top X række, kan brugeren sætte fetch til et heltal. 
            if isinstance(fetch,int):
                command+=f" LIMIT {fetch}"
            elif fetch.lower()=="first" or fetch.lower()=="one" or fetch.lower()=="top":
                command+=" LIMIT 1"
            ## eksekver kommando
            interface.execute(command)
            
            ## definer og vis tabel
            column_names=[title[0] for title in interface.description]
            table=interface.fetchall()
            print(tabulate(table, headers=column_names, tablefmt="fancy_grid"))
            
        else:
            print("No Connection Established. Start A Session First.")
            
            
    # funktion til at slette en tabel fra den nuværende database. 
    def delete_table(self,table):
        if self.connection:    
            self.connection.cursor().execute(f"DROP TABLE IF EXISTS {table}")
        else:
            print("No Connection Established. Start A Session First.")
    
    # funktion til at slette en database fra serveren
    def delete_db(self,db):
        if self.connection:    
            ## Vi skal lige tjekke om den givne database rent faktisk findes på serveren. 
            db_interface=self.connection.cursor()
            db_interface.execute("SHOW DATABASES")
            db_check=[x[0] for x in db_interface.fetchall()]
            if db in db_check:
                db_interface.execute(f"DROP DATABASE IF EXISTS {db}")
            else:
                print(f"ERROR: Database {db} Not Found On Server. ")
        else:
            print("No Connection Established. Start A Session First.")
    
    # funktion til at ændre værdier i en variabel i tabellen 'table' på basis af en given værdi i en variabel
    ## 'var1' og 'var2' er variabler, samt 'value1' og 'value2' er værdier til tabellen
    def update_table(self, table, var1, value1, var2, value2):
        if self.connection:  
            command=f"UPDATE {table} SET {var1} = '{value1}' WHERE {var2} = '{value2}'"
            self.connection.cursor().execute(command)
            self.connection.commit()
        else:
            print("No Connection Established. Start A Session First.")
    
    # funktion til at indsætte en tabel i databasen fra en .csv fil eller en pandas.dataframe. 
    ## her menes det, at: 
    ### 'tabel' er den tabel, som man gerne vil indsætte i databasen
    def import_table(self,tabel):
        ## hvis 'tabel' indhentes direkte fra en .csv fil, vil tabellen hentes og åbnes som en pandas.dataframe
        try:
            name=os.path.basename(tabel)
            name=name.split(".")[0]
            
            if ".csv" not in tabel:
                raise ValueError("Uploaded fil must be in .csv format.")
            data=pandas.read_csv(tabel,sep=",")
            
            ## erstatter Null med None
            data=data.where(pandas.notnull(data),None)
            
        ## der kan tages højde for andre data-input (såsom fra .xlsx filer), men for denne opgave, vil der kun blive taget højde for .csv filer og pandas.dataframe's
        except Exception as e:
            print(f"ERROR IN IMPORTING {tabel}: {e}")
            sys.exit()
        
        # opretter en tom tabel med antallet af kolonner svarende til kolonnenavnene fra 'tabel' (kodeinspiration fra www.w3schools.com) 
        columns=", ".join([f" {col} VARCHAR(255)" for col in list(data.columns)])
        self.connection.cursor().execute(f"CREATE TABLE IF NOT EXISTS {name}  ({columns})")
        self.connection.commit()
        
        # opretter de tomme celler i tabellen
        cols=", ".join(list(data.columns))
        null_content=", ".join(["%s" for x in list(data.columns)])
        
        # indsætter data i den tomme tabel 
        command=f"INSERT INTO {name} ({cols}) VALUES ({null_content})"
        content_tuples=list(data.itertuples(index=False,name=None))
        
        self.connection.cursor().executemany(command,content_tuples)
        self.connection.commit()

    ## funktion som viser hele serveren med alle de databaser, som ligger på serveren. 
    def show_server(self):
        if self.connection:  
            interface=self.connection.cursor()
            interface.execute("SHOW DATABASES")
            databases = interface.fetchall()
            for x in databases:
                print(x[0])
        else:
            print("No Connection Established. Start A Session First.")
            
    # funktion, som viser indholdet af den database, som man bruger.
    ## Den viser kun tabel navn og dimensioner. Hvis man vil se indholdet i en tabel, kan man anvende show_table() funktionen istedet. 
    def show_db(self):
        if self.connection:  
            interface=self.connection.cursor()
            ## finder lige navnet på den nuværende database
            interface.execute("SELECT DATABASE()")
            DB=interface.fetchone()
            try:
                ## også tabellerne fra databasen
                interface=self.connection.cursor()
                interface.execute("SHOW TABLES")
                tables=interface.fetchall()
                print(f"Present Database: {DB[0]} containing {len(tables)} Tables")
                for x in tables:
                    # udtrækker tabelnavn
                    name=x[0]
                    
                    # indhenter tabellen som en list of tuples   
                    interface=self.connection.cursor()
                    interface.execute(f"SELECT * FROM {name}")
                    table=interface.fetchall()
                    
                    # i tilfælde af en tom tabel
                    if len(table)==0:
                        col_count=0
                        row_count=0
                    else:
                        # tæller kolonner ved at udregne længden af første række (aka. første "tuple" i listen)
                        first_row=list(table[0])
                        col_count=len(first_row)
                        
                        # tæller rækker ved at tælle længden af listen med "tuple"'s
                        row_count=len(table)
                    
                    # printer række med tabelnawn: N Columns * M Rows
                    print(f"{name}: {col_count} Columns * {row_count} Rows")
            except Exception as e:
                print(f"ERROR IN SHOWING DATABASE: {e}")
        else:
            print("No Connection Established. Start A Session First.")
            
    # Funktion til at flytte en database til en ny database.
    def rename_db(self,old_name,new_name):
        if self.connection:  
            interface=self.connection.cursor()
            interface.execute("SHOW DATABASES")
            db_check=[x[0] for x in interface.fetchall()]
            
            # hvis den nye database findes, vil der være fejl. 
            if new_name not in db_check:
                interface.execute(f"CREATE DATABASE {new_name}")
            else:
                raise mysql.connector.Error(f"ERROR: {new_name} Already Exists On The Server.")
            interface.execute(f"SHOW TABLES FROM {old_name}")
            tables=interface.fetchall()
            ## flytter hver tabel fra database 'old_name' til database 'new_name'
            for x in tables:
                name=x[0]
                interface.execute(f"RENAME TABLE {old_name}.{name} TO {new_name}.{name}")
            
            ## hvis vi flytter indhold fra en database, som vi allerede arbejder på, flytter vi os lige over til den nye database.  
            if self.db==old_name:
                self.connection.database=new_name
                self.db=new_name
            interface.execute(f"DROP DATABASE {old_name}")
            #interface.commit()
        else:
            print("No Connection Established. Start A Session First.")
    
    # funktion til at vise et kolonne med mulige filtre.
    ## 'filters' skal være en liste over kriterier for en selektion skrevet som man vil det i SQL, f.eks "by = 'Odense'"
    ## 'andor' skal være en liste med et antal "AND"/"OR" svarende til antal kriterier -1 
    def show_col(self,table,col="*",filters=[],andor=[]):
        if self.connection:  
            interface=self.connection.cursor()
            command=f"SELECT {col} FROM {table}"
            
            ## såfremt at 'filters' og 'andor' er ikke-tomme lister, vil det, som skrives i dem blive skrevet som kommando
            if type(filters)==type(andor)==list and len(filters)!=0:
                ## er der filtre, skal det initieres med WHERE
                command+=" WHERE "
                ## Det vil ikke være utænkeligt at andor bliver kortere end filters, så det gøres lige lidt længere, så alle filtre kan komme med. 
                if len(andor)<len(filters):
                    andor+=[""]
                for x in zip(filters,andor):
                    ## inde i et filter kan der godt være "between","and", "in" og/eller "or", som f.eks. pris BETWEEN 0 AND 1000
                    ## men det skal sikres at være med stort, så det sikres her. 
                    if " between " in x[0] or " and " in x[0] or " or " in x[0] or " in " in x[0]:
                        filter_list=x[0].split(" ")
                        filter_list=[y.upper() for y in filter_list if y in ["between","and","or","in"]]
                        x[0]=" ".join(filter_list)
                    ## sikre os også at andor er med stor 
                    x[1]=x[1].upper()
                    ## opbygger kommandoen.
                    command+=f" {x[0]} {x[1]}"
            
            interface.execute(command)
            table=interface.fetchall()
            column_names = [column[0] for column in interface.description]
            
            print(tabulate(table, headers=column_names, tablefmt="fancy_grid"))
        else:
            print("No Connection Established. Start A Session First.")
    
    # Funktion, som filtere en tabel til en ny tabel
    def filter_table(self,table,new_table,col="*",filters=[],andor=[]):
        if self.connection:  
            interface=self.connection.cursor()
            ## hvis brugeren vælger en liste af variabler, skal det lige bearbejdes, så det kan læses som en SQL kommando
            if type(col)==list:
                col=", ".join(col)
            ## hvis col er angivet som en streng, skal vi sikre os at den er opsat, så den er læselig for SQL.
            elif type(col)==str and ", " not in col:
                filter_list=col.split(",")
                filter_list=", ".join(filter_list)
            command=f"CREATE TABLE {new_table} SELECT {col} FROM {table}"
            ## såfremt at 'filters' og 'andor' er ikke-tomme lister, vil det, som skrives i dem blive skrevet som kommando
            if type(filters)==type(andor)==list and len(filters)!=0 and len(andor)!=0:
                ## er der filtre, skal det initieres med WHERE
                command+=" WHERE "
                ## Det vil ikke være utænkeligt at andor bliver kortere end filters, så det gøres lige lidt længere, så alle filtre kan komme med. 
                if len(andor)<len(filters):
                    andor+=[""]
                for x in zip(filters,andor):
                    ## inde i et filter kan der godt være "between","and", "in" og/eller "or", som f.eks. pris BETWEEN 0 AND 1000
                    ## men det skal sikres at være med stort, så det sikres her. 
                    if " between " in x[0] or " and " in x[0] or " or " in x[0] or " in " in x[0]:
                        filter_list=x[0].split(" ")
                        filter_list=[y.upper() for y in filter_list if y in ["between","and","or","in"]]
                        x[0]=" ".join(filter_list)
                    ## sikre os også at andor er med stor 
                    x[1]=x[1].upper()
                    ## opbygger kommandoen.
                    command+=f" {x[0]} {x[1]}"
            
            interface.execute(command)
        else:
            print("No Connection Established. Start A Session First.")
    
    # Funktion til at slette kolonner fra en tabel i databasen.
    def delete_col(self,table,col):
        if self.connection:  
            interface=self.connection.cursor()
            if type(col)==list:
                col=", ".join(col)
            elif type(col)==str and ", " not in col:
                filter_list=col.split(",")
                col=", ".join(filter_list)
            command=f"ALTER TABLE {table} DROP COLUMN {col}"
            interface.execute(command)
        else:
            print("No Connection Established. Start A Session First.")
    
    # Funktion til at tilføje en kolonne til en tabel i databasen.
    def add_col(self,table,col,coltype):
        if self.connection:  
            interface=self.connection.cursor()
            if type(col)==list:
                col=", ".join(col)
            elif type(col)==str and ", " not in col:
                filter_list=col.split(",")
                col=", ".join(filter_list)
            command=f"ALTER TABLE {table} ADD COLUMN {col} {coltype}"
            interface.execute(command)
        else:
            print("No Connection Established. Start A Session First.")
    
    # funktion til at vise en grupperet tabel over data i en tabel i en database
    def group_by(self,table,col="*",grouping=[]):
        if self.connection: 
            interface=self.connection.cursor()
            ## strukturer kolonnerne, så de passer til kommandoen.
            if type(col)==list:
                col=", ".join(col)
            elif type(col)==str and ", " not in col:
                col=col.split(",")
                col=", ".join(col)
                ## strukturer grupperingskriterier, så de passer til kommandoen    
            if type(grouping)==list:
                grouping=", ".join(grouping)
            elif type(grouping)==str and ", " not in grouping:
                grouping=grouping.split(",")
                grouping=", ".join(grouping)
            ## opstil og eksekver kommando
            command=f"SELECT {col} FROM {table} GROUP BY {grouping}"
            interface.execute(command)
            table = interface.fetchall()
            column_names=[column[0] for column in interface.description]
            ## udskriv grupperet tabel
            print(tabulate(table, headers=column_names, tablefmt="fancy_grid"))
        else:
            print("No Connection Established. Start A Session First.")
    
    # funktion til at tilføje rækker til en tabel i databasen. 
    ## 'cols' og 'vals' skal være lister med henholdsvist kolonnenavne i tabellen og de værdier, som man vil indsætte.
    def add_row(self,table,cols=[],vals=[]):
        if self.connection: 
            interface=self.connection.cursor()
            ## typisk vil antal kolonner og celler være nogle stykker, så det vil være lettest at bearbejde dem i lister
            if type(cols)!=list or type(vals)!=list:
                raise TypeError("Variables 'cols' and 'vals' must be lists!")
                
            ## for insert kommandoen SKAL antal kolonner og celler stemme overens med tabellen
            interface.execute(f"SHOW COLUMNS FROM {table}")
            table_cols=len(interface.fetchall())
            if len(cols)!=len(vals)!=table_cols:
                ValueError(f"Variables 'cols' and 'vals' must be of equal lengths to the number of columns in table {table}")
        
            ## variablerne skal tilpasses kommandoen
            columns="("+", ".join(cols)+")"
            empty_cells="("+", ".join(['%s']*len(vals))+")"
            
            ## så har vi nok til kommandoen
            command=f"INSERT INTO {table} {columns} VALUES {empty_cells}"
            interface.execute(command,vals)
        else:
            print("No Connection Established. Start A Session First.")
            
    # funktion til at fjerne rækker i en tabel i databasen
    ## 'col_name' skal være en streng svarende til et kolonnenavn i tabellen
    ## 'col_val' skal bare være en enkelt værdi, som selekteres for i kolonnen.
    def delete_row(self,table,col_name,col_val):
        if self.connection: 
            interface=self.connection.cursor()
            col_val=(col_val,)
            command=f"DELETE FROM {table} WHERE {col_name} = %s"
            interface.execute(command,col_val)
        else:
            print("No Connection Established. Start A Session First.")
            
    
    # Funktion, som eksportere tabeller fra databasen til computeren. 
    ## 'file_path' definere stien til mappen, som filerne skal downloades til
    ## 'table_name' definere tabellen/tabellerne, som skal hentes. 
    ## hvis 'table_name' defineres som "*" (som er standard), hentes alle 
    def export_tables(self,file_path,table_name="*"):
        if self.connection: 
            interface=self.connection.cursor()
            
            ## angives 'table_name' som "*", vil alle tabeller i den nuværende database blive eksporteret
            if table_name=="*":
                interface.execute("SHOW TABLES")
                tables=interface.fetchall()
            else:
                tables=[(table_name,0)] # formatet er anderledes ved fetchall(), så for ikke at lave hver sin eksport-funktion, so laves dette som en meget kort list of tuples
            
            ## iterere over tabellerne og eksportere dem en for en. 
            for table in tables:
                table_name=table[0]
                ## henter tabelindmad
                interface.execute(f"SELECT * FROM {table_name}")
                table=interface.fetchall()
                ## henter kolonnenavne
                interface.execute(f"SELECT * FROM {table_name} LIMIT 1")
                descriptions=interface.description
                column_names=[col[0] for col in descriptions]
                ## konvertere tabellen til dataframe og skriver til en csv fil
                df = pandas.DataFrame(table, columns=column_names)
                final_path = os.path.join(file_path,f"{table_name}.csv")
                df.to_csv(final_path,index=False)
            _=interface.fetchall() ## henter ubrugt data for at sikre at vi kan bruge cursor igen.
            interface.close()
        else:
            print("No Connection Established. Start A Session First.")
    
    # funktion til at sammenflette to tabeller med en fælles nøgle.
    ## Jeg havde tænkt på at lave det som en multi-flet-funktion, hvor man kunne flette mere end 2 tabeller.
    ## men det var svært at tage højde for de forskellige nøgler. 
    def join_tables(self, left_table, right_table, new_table, common_key, direction="INNER"):
        if self.connection:
            interface=self.connection.cursor()
            
            ## få alle kolonnerne fra venstre til højre
            interface.execute(f"SHOW COLUMNS FROM {left_table}")
            left_columns=[name[0] for name in interface.fetchall()]
            interface.execute(f"SHOW COLUMNs FROM {right_table}")
            right_columns=[name[0] for name in interface.fetchall()]
            
            ## Vi er nød til at omdøbe variablerne, hvis der er duplikater. 
            columns_renamed=[f"{right_table}.{col} AS {right_table}_{col}" if col in left_columns and col!=common_key else f"{right_table}.{col}" for col in right_columns if col != common_key]
            select_columns=", ".join([f"{left_table}.{col}" for col in left_columns] + columns_renamed)
            
            ## opsætter de primære kommandoer for join-query
            base_command=f"CREATE TABLE {new_table} AS SELECT {select_columns} FROM {left_table} "
            
            ## her tages der højde for at brugeren kan skrive 'direction' med småt
            direction=direction.upper()
            
            ## basis retningskommandoen defineres her. 
            direction_command=f"{direction} JOIN {right_table} ON {left_table}.{common_key} = {right_table}.{common_key}"
            
            ## her skelnes der imellem almindelige join-queries og full join-query
            command=base_command + direction_command
            
            ## eksevker kommandoen
            interface.execute(command)
            interface.close()
        else:
            print("No Connection Established. Start A Session First.")
    
    ## funktion, som tillader en at omdøbe en kolonne. 
    def rename_col(self,table,old_colname,new_colname):
        if self.connection:
            interface=self.connection.cursor()
            command=f"ALTER TABLE {table} CHANGE {old_colname} {new_colname} VARCHAR(255)"
            interface.execute(command)
            interface.close()
        else:
            print("No Connection Established. Start A Session First.")
    
    ## funktion til at slette alle ikke-nødvendige databaser fra serveren. 
    def clean_server(self):
        if self.connection:
            interface=self.connection.cursor()
            ## finder alle databaserne på serveren
            interface.execute("SHOW DATABASES")
            content=interface.fetchall()
            ## liste over standard databaser fra en MySQL server, som vi ikke vil have slettet
            system_db = ['information_schema', 'performance_schema', 'mysql']
            
            ## så går vi individuelt igennem hver database for at slette dem. 
            for x in content:
                name=x[0]
                ## sletter kun databaser, som ikke er system databaser. 
                if name not in system_db:
                    interface.execute(f"DROP DATABASE {name}")
        else:
            print("No Connection Established. Start A Session First.")
    
    ## funktion til at gemme database fra serveren til computeren. 
    def backup_db(self,db,file_path=""):
        if self.connection:
            ## sættter isolation for at sikre imod deadlock
            #self.connection.start_transaction(isolation_level='READ COMMITTED')
        
            
            ## hvis file_path ikke er angivet, eksporteres databasen til rodmappen. 
            if not file_path or file_path=="":
                file_path = os.path.join(os.getcwd(),f"{db}.sql")
                database_name=db
            ## hvis der kun er givet filnavn, skal vi sikre os at det rent faktisk er en sql fil. 
            elif "sql" not in file_path.split(".")[-1]:
                database_name=os.path.basename(file_path)
                file_path+=".sql"
            else:
                database_name=os.path.basename(file_path).rstrip(".sql")
            
            try:
                ## Få  fat i alle tabeller i databasen
                interface = self.connection.cursor()
                interface.execute("SHOW TABLES")
                tables=[row[0] for row in interface.fetchall()]
                
                ## få fat i relationer
                relation_command=f"""SELECT
                                    kcu.table_name,
                                    kcu.column_name,
                                    kcu.referenced_table_name,
                                    kcu.referenced_column_name,
                                    kcu.constraint_name
                                    FROM
                                    information_schema.referential_constraints rc
                                    JOIN information_schema.key_column_usage kcu
                                    ON rc.constraint_name = kcu.constraint_name
                                    AND rc.constraint_schema = kcu.table_schema
                                    WHERE
                                    rc.constraint_schema = '{db}';
                                """
                                
                interface.execute(relation_command)
                relations=interface.fetchall()
                
                # Hent PRIMARY og UNIQUE keys
                constraints = {}
                for table in tables:
                    interface.execute(f"SHOW INDEX FROM `{table}`")
                    keys = interface.fetchall()
                    for row in keys:
                        key_name = row[2]
                        column = row[4]
                        non_unique = row[1]
                        if table not in constraints:
                            constraints[table] = {"PRIMARY": [], "UNIQUE": []}
                        if key_name == 'PRIMARY':
                            constraints[table]["PRIMARY"].append(column)
                        elif non_unique == 0:
                            constraints[table]["UNIQUE"].append(column)
                
                # hent brugere, roller og privilegier
                interface.execute("SELECT user, host FROM mysql.user")
                users = interface.fetchall()
                grants = []

                for user, host in users:
                    try:
                        interface.execute(f"SHOW GRANTS FOR '{user}'@'{host}'")
                        grants_for_user = interface.fetchall()
                        for grant in grants_for_user:
                            grants.append(grant[0] + ";")  # grant[0] is the actual SQL statement
                    except Exception as f:
                        print(f"Could not get grants for {user}@{host}: {f}")
                
                # hent views
                interface.execute(f"SELECT table_name, view_definition FROM information_schema.views WHERE table_schema = '{db}'")
                views=interface.fetchall()
                interface.close()

                
                with open(file_path,"w") as output_file:
                    output_file.write(f"CREATE DATABASE IF NOT EXISTS {database_name};\n")
                    output_file.write(f"USE {database_name};\n\n")
                    for table in tables:
                        ## opretter tabellen i SQL-filen med sikkerhed om at der ikke sker fejl, hvis en tabel allerede findes
                        name=table
                        output_file.write(f"DROP TABLE IF EXISTS `{name}`;\n")
                        output_file.write(f"CREATE TABLE `{name}` (\n")
                        
                        ## opretter kolonnebeskrivelser
                        interface=self.connection.cursor()
                        interface.execute(f"DESCRIBE `{name}`")
                        col = interface.fetchall()
                        interface.close()
                        
                        ## få kolonnenavne til filen
                        columns_sql=[f"`{column[0]}` {column[1]}" for column in col]
                        #output_file.write(",\n".join(columns_sql) + "\n);\n\n")
                        
                        if constraints.get(table):
                            if constraints[table]["PRIMARY"]:
                                pk = ", ".join(f"`{x}`" for x in constraints[table]["PRIMARY"])
                                columns_sql.append(f"PRIMARY KEY ({pk})")
                            if constraints[table]["UNIQUE"]:
                                uq = ", ".join(f"`{x}`" for x in constraints[table]["UNIQUE"])
                                columns_sql.append(f"UNIQUE ({uq})")
                        output_file.write(",\n".join(columns_sql) + "\n);\n\n")
                        
                        interface=self.connection.cursor()
                        interface.execute(f"SELECT * FROM {name}")
                        content = interface.fetchall()
                        interface.close()
                       
                        content_values=[]
                        for row in content:    
                            ## erstatter enkelt apostrof med dobbelt, da SQL ellers vil læse fejl. 
                            row=[None if x is None else x.replace("'","''") if isinstance(x,str) else x for x in row]
                            values="("+", ".join(["NULL" if x is None else f"'{str(x)}'" for x in row])+")"
                            content_values+=[values]
                        content_values=",".join(content_values)
                        output_file.write(f"INSERT INTO `{name}` VALUES {content_values};\n")
                        output_file.write("\n")
                        
                    ## hvis relation_command giver resultater, er der relationer i databasen, som også skal skrives ind
                    if len(relations)>0:
                        output_file.write("-- Foreign Keys\n")
                        for relation in relations:
                            ## sådan som relation_command er sat op, vil formatet for output være:
                            ## [(child_table, child_col, parent_table, parent_col, relation_key)]
                            output_file.write(f"""ALTER TABLE {relation[0]}
                                                  ADD CONSTRAINT {relation[4]}
                                                  FOREIGN KEY ({relation[1]})
                                                  REFERENCES {relation[2]} ({relation[3]})
                                                  ON DELETE CASCADE; \n \n
                                              """)
                    
                    ## hvis der er roller, brugere og privilegier, vil det også blive skrevet ind. 
                    if len(grants)>0:
                        output_file.write("-- roles, users and privileges \n")
                        for grant in grants:
                            output_file.write(grant+"\n")
                        output_file.write("\n")
                        
                    ## hvis der er views, vil det også blive skrevet ind
                    if len(views)>0:
                        output_file.write("-- views\n")
                        for view_name,view_def in views:
                            output_file.write(f"CREATE OR REPLACE VIEW '{view_name}' AS {view_def};\n\n")
                    
                    self.connection.commit()
            except Exception as e:
                print(f"ERROR IN EXPORTING DATABASE {db}: {e}")
        else:
            print("No Connection Established. Start A Session First.")
            
    ## funktion til at undersøge data type og danne let overblik over data information
    ### tabel skal bare være en streng med tabelnavnet og colname skal ligeledes være en streng med kolonnenavne. 
    ### colname burde også kunne tage Wildcard variabler (defineret med tilstedeværelsen af % og/eller _)
    def describe(self,tabel,colname="*"):
        if self.connection:
            interface=self.connection.cursor()
            
            ## definer kommandoen altefter om man ønsker at beskrive en hel tabel, en enkelt variabel eller flere via et wildcard 
            command=f"DESC {tabel}"
            if colname!="*":
                command+=f" {colname}"
            
            ## eksevker kommando og hent resultater
            interface.execute(command)
            content=interface.fetchall()
            
            ## henter kolonnenavnene og sætter det pænt op
            col_description=interface.description
            col_names=[item[0] for item in col_description]
            
            ## sætter tabellen pænt op og udskriver den
            print(tabulate(content,headers=col_names,tablefmt="fancy_grid"))
        else:
            print("No Connection Established. Start A Session First.")
    
    def import_api(self, url, tablename, headers=None, params=None):
        if self.connection:
            interface=self.connection.cursor()
            ## hvis tabellen allerede findes på databasen, vil en kopi blive oprettet i lign. format som windows.
            interface.execute("SHOW TABLES")
            tables=interface.fetchall()
            
            if tablename in tables:
                temp_name=""
                n=0
                while tablename!=temp_name:
                    n+=1
                    temp_name = tablename + f"({n})"
                    if temp_name not in tables:
                        tablename=temp_name
            try:
                ## 'header' og 'params' varaiablerne skal være enten dicts eller None
                if headers!=None and params!=None:
                    if type(headers)!=dict or type(params)!=dict:
                        raise TypeError("parameter 'header' and 'params' must either be of type 'dict' or None.")
                
                
                ## her prøver vi at oprette forbindelse. umiddelbart forventer vi øjeblikkelig forbindelse,
                ### men det er let at få den til at vente. 
                response=requests.get(url, headers=headers, params=params)
                if response.status_code!=200:
                    raise ConnectionError("URL did not respond in time.")
                
                ## konvertere respons til json
                data=response.json()
                
                ## output kommer som en list of dicts, som er konverteret til string, så det skal bearbejdes.
                data=data.strip("[]") # fjerner kasse parenteserne 
                data=data.strip("{") # fjerner de perifære tuborg parenteser 
                data=data.rstrip("}") # -//-
                data=data.split("},{") # på nuværende tidspunkt er hver linje adskilt med "},{"
                
                ## nu vil det være bedst at læse denne liste igemmen, og separere på hvad der er nøgle og indhold
                ### og indsætte resten af data i en ny list of dicts. 
                clean_data=[]
                for line in data:
                    Line=line.split(",") # hver celle i en række er komma-separeret
                    row={}
                    for cell in Line:
                        key,Cell=cell.split(":") # hver celle per linje er separeret med kolon med nøglen til venstre
                        row[key]=Cell # nu kan hver celle indsættes i den nye række
                    clean_data+=[row] # som kan indsættes i den nye tabel
                
                # som let kan konverteres til en dataframe
                data_df = pandas.DataFrame(clean_data)
                ## Dog er alle tomme værdier defineret nu som en "NULL" streng, så det skal erstattes. 
                data_df=data_df.where(data_df!="NULL",None)
                
                # opretter en tom tabel med antallet af kolonner svarende til kolonnenavnene fra 'tabel' (kodeinspiration fra www.w3schools.com) 
                columns=", ".join([f"{col} VARCHAR(255)" for col in list(data_df.columns)])
                columns=columns.replace('"','')
                interface.execute(f"CREATE TABLE IF NOT EXISTS {tablename} ({columns})")
                self.connection.commit()
                
                # opretter de tomme celler i tabellen
                cols=columns.replace('VARCHAR(255)','')
                null_content=", ".join(["%s" for x in list(data_df.columns)])

                # indsætter data i den tomme tabel
                command=f"INSERT INTO {tablename} ({cols}) VALUES ({null_content})"
                content_tuples=list(data_df.itertuples(index=False,name=None))
                interface.executemany(command,content_tuples)
                self.connection.commit()
             
            except Exception as e:
                print(f"ERROR IN IMPORTING DATA FROM API: {e}")
        else:
            print("No Connection Established. Start A Session First.")
    
    ## funktionalitet til at ændre datatype for en kolonne
    ### 'newtype' skal være læseligt i MySQL
    def change_coltype(self,tabel,colname,newtype):
        if self.connection:
            interface=self.connection.cursor()
            
            newtype=newtype.upper()
            
            command=f"ALTER TABLE {tabel} MODIFY COLUMN {colname} {newtype}"
            try:
                if "DATE" in newtype or "TIME" in newtype:
                    ## hvis variablen skal være en datovariabel, skal vi sikre os at den kan læses som en datovariabel af MySQL
                    date_formats = [(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%s"),
                                    (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
                                    (r"\d{2}/\d{2}/\d{4}", "%d/%m/%Y"),
                                    (r"\d{2}-\d{2}-\d{4}", "%d-%m-%Y"),
                                    (r"\d{4}/\d{2}/\d{2}", "%Y/%m/%d"),
                                    (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%s"),
                                    (r"\d{2}:\d{2}:\d{2}", "%H:%M:%s")]
                    
                    ## Vi laver et hurtigt udtræk på de 5 første ikke-tomme celler i variablen og ser hvilket format de er i.
                    value_extract_command=f"SELECT {colname} FROM {tabel} WHERE {colname} IS NOT NULL LIMIT 5"
                    interface.execute(value_extract_command)
                    raw_samples=interface.fetchall()
                    value_samples=[cell[0].replace("\"","") for cell in raw_samples]
                    
                    ## så tjekker vi formattet
                    final_format=""
                    for value in value_samples:
                        for pattern,Format in date_formats:
                            if re.match(pattern,value):
                                if final_format=="":
                                    final_format=Format
                                #elif final_format!=Format:
                                #    raise AttributeError(f"Column {colname} in Tabel {tabel} Contains Values of Multiple Different Format")
                    if final_format=="":
                        raise ValueError(f"No Date Format Detected for Column {colname} in Tabel {tabel}.")
                    
                    ## Vi vil også sikre os at funktionen ikke fejler, hvis der er tomme variabler. Såsom hvis en dato ikke er givet for en afsluttet datovariabel
                    ### sætter en CASE for at håndtere tomme værdier
                    mod_command=f"""UPDATE `{tabel}` 
                                        SET `{colname}` = CASE
                                            WHEN UPPER(TRIM(REPLACE(SUBSTRING_INDEX(`{colname}`, '+', 1), '"', ''))) IN ('', 'NA', 'NULL') THEN NULL
                                            ELSE STR_TO_DATE(REPLACE(SUBSTRING_INDEX(`{colname}`, '+', 1), '"', ''), '{final_format}')
                                        END
                                        WHERE `{colname}` IS NOT NULL AND `{colname}` != ''"""
                ## der kan også være problemer, hvis der er NA i tal-variabler
                elif any([True if i in newtype else False for i in ["INT","DECIMAL","NUM","FLOAT","DOUBLE"]]):
                    mod_command=f"""
                                UPDATE `{tabel}`
                                SET `{colname}` = CASE
                                    WHEN `{colname}` IN ('','NA','NULL') THEN NULL
                                    WHEN `{colname}` REGEXP '^-?[0-9]+$' THEN `{colname}`
                                    ELSE NULL
                                END
                                WHERE `{colname}` IS NOT NULL AND `{colname}` !=''
                                """
                
                ## eksekvere denne opdatering inden vi ændre datatype
                interface.execute(mod_command)
                self.connection.commit()
                
                ## eksevker kommando for at ændre datatype. 
                interface.execute(command)
                self.connection.commit()
            except Exception as e:
                print(f"ERROR IN CHANGING {colname} TO {newtype}: {e}")
        else:
            print("No Connection Established. Start A Session First.")
    
    ## funktion til at give et statistisk overblik over en tabel
    ### gemmer ikke tabellen i noget miljø og printer den kun.
    def stat_table(self,table):
        if self.connection:
            interface=self.connection.cursor()
            
            ## henter nødvendig data fra tabellen
            interface.execute(f"SELECT * FROM {table}")
            data=interface.fetchall()
            colnames=[i[0] for i in interface.description]
            #### henter også lige datatype:
            interface.execute(f"DESCRIBE {table}")
            info=interface.fetchall()
            coltype_index={}
            for i in info:
                coltype_index[i[0]]=i[1]
            
            ## vi sætter data op som en midlertidig dataframe, så vi kan lave statistik over det. 
            df=pandas.DataFrame(data,columns=colnames)
            
            ## og opretter en tjekliste over de værdier, som vi kan lave generel statistik på. 
            stat_types=["TINYINT","SMALLINT","MEDIUMINT","INT","INTEGER","BIGINT","FLOAT","DOUBLE","DOUBLE PRECISION","DECIMAL","DEC"]
            date_types=["DATE","TIMESTAMP","DATETIME","TIME"]
            all_types=stat_types+date_types
            
            stat_table_list=[]
            for col in df.columns:
                missing=df[col].isna().sum()/df[col].shape[0] * 100
                if any([True if i in coltype_index[col].upper() else False for i in all_types]):
                    
                    ## hvis variablen er en tidsvariabel, skal den lige konverteres først  
                    if any([True if i in coltype_index[col].upper() else False for i in date_types]):
                        df[col]=pandas.to_datetime(df[col])
                    
                    stats= {"Column name":col,
                            "Data Type":coltype_index[col],
                            "Mean":df[col].mean(),
                            "Median":df[col].median(),
                            "Standard Deviation":df[col].std(),
                            "Minimum":df[col].min(),
                            "Maximum":df[col].max(),
                            "# Uniques":df[col].nunique(),
                            "% Missing":missing,
                            "# Missing":df[col].isna().sum(),
                            "# rows":df[col].shape[0]}   
                else:
                    stats= {"Column name":col,
                            "Data Type":coltype_index[col],
                            "Mean":None,
                            "Median":None,
                            "Standard Deviation":None,
                            "Minimum":None,
                            "Maximum":None,
                            "# Uniques":df[col].nunique(),
                            "% Missing":missing,
                            "# Missing":df[col].isna().sum(),
                            "# rows":df[col].shape[0]}
                stat_table_list+=[stats]
            
            stat_df = pandas.DataFrame(stat_table_list)
            print(tabulate(stat_df, headers='keys', tablefmt="fancy_grid"))
        else:
            print("No Connection Established. Start A Session First.")
    
    ## importer SQL-fil
    ### OBS: med mindre at filen indeholder kode for at oprette og flytte til en ny database
    ### vil al data blive hentet til den nuværende database
    def import_sql(self,file_path):
        if self.connection:
            interface=self.connection.cursor()
            
            try:
                ## åbner filen som en streng
                with open(file_path,"r") as file:
                    sql=file.read()
                ## en SQL fil er normalt en række SQL kommandoer som er semikolon separeret.
                ### så vi separere hver kommando i en liste med split på semikolon 
                queries=sql.split(";")
                for query in queries:
                    query=query.strip()
                    if query:
                        interface.execute(query)
                        self.connection.commit()
                    
            except Exception as e:
                print(f"ERROR IN SQL IMPORT: {e}")
        else:
            print("No Connection Established. Start A Session First.")
    
    ## funktion til at flytte tabel imellem to databaser. 
    def move_table(self, table, old_db, new_db):
        if self.connection:
            interface=self.connection.cursor()
            
            ## for at spare plads, sletter vi den gamle tabel
            move_command= f"CREATE TABLE {new_db}.{table} AS SELECT * FROM {old_db}.{table}"
            clean_command= f"DROP TABLE {old_db}.{table}"
            
            ## for at få koden til at fylde mindre, iterere vi over disse to kommandoer. 
            command_list=[move_command, clean_command]
            for i in command_list:
                interface.execute(i)
                interface.commit()
        else:
            print("No Connection Established. Start A Session First.")    
    
    ## funktion til at skabe relationer imellem tabeller i en database. 
    def create_relation(self, child_table, parent_table, child_col, parent_col, relation_key="",on_delete="CASCADE"):
        if self.connection:
            interface=self.connection.cursor()
            
            if relation_key=="":
                relation_key=f"fk_{child_table}_{child_col}"
            
            ## parringskommando
            command= f"""ALTER TABLE {child_table}
                        ADD CONSTRAINT {relation_key}
                        FOREIGN KEY ({child_col})
                        REFERENCES {parent_table} ({parent_col})
                        ON DELETE {on_delete}"""
            
            ## ekekvering af kommandto 
            interface.execute(command)
            self.connection.commit()
        else:
            print("No Connection Established. Start A Session First.")
    
    ## funktion til at eksekvere SQL kommandoer rent, som skrevet i en SQL prompt. 
    ## "command" skal være en streng.
    def freestyle(self,command):
        if self.connection:
            interface=self.connection.cursor()
            try:
                if type(command)!=str:
                    raise ValueError("Command Must Be A String!")
                    
                interface.execute(command)
                
                # Hvis det er en SELECT eller lign., så hent og print resultater
                if interface.with_rows:
                    results = interface.fetchall()
                    colnames = [desc[0] for desc in interface.description]
                    print(tabulate(results, headers=colnames, tablefmt="fancy_grid"))
                    self.connection.commit()
                else:
                    self.connection.commit()
                
            except Exception as e:
                print(f"FREESTYLE ERROR: {e}")
        else:
            print("No Connection Established. Start A Session First.")
    
    ## funktion til at tilføje primary key til en tabel
    def add_primary_key(self,table,col):
        if self.connection:
            try:
                ## man kan enten være en eller flere kolonner, som i det sidste tilfælde skal være en liste
                if type(col) not in [str,list]:
                    raise ValueError("col variabel must either be a string or list")
                if type(col)==list:
                    col=", ".join(col)
                interface=self.connection.cursor()
                interface.execute(f"ALTER TABLE {table} ADD PRIMARY KEY ({col});")
            except Exception as e:
                print(f"ERROR IN SETTING PRIMARY KEY: {e}")
        else:
            print("No Connection Established. Start A Session First.")
        
    ## funktion til at tilføje unique key til en tabel
    def add_unique_key(self,table,col):
        if self.connection:
            try:
                ## man kan enten være en eller flere kolonner, som i det sidste tilfælde skal være en liste
                if type(col) not in [str,list]:
                    raise ValueError("col variabel must either be a string or list")
                if type(col)==list:
                    col=", ".join(col)
                interface=self.connection.cursor()
                interface.execute(f"ALTER TABLE {table} ADD UNIQUE ({col});")
            except Exception as e:
                print(f"ERROR IN SETTING UNIQUE KEY: {e}")
        else:
            print("No Connection Established. Start A Session First.")
    
    ## funktion til at håndtere roller
    def role(self,action,name="",user=""):
        if self.connection:
            try:
                if action.upper() not in ["CREATE","DELETE","GRANT","SHOW","GRANTED"]:
                    raise ValueError("Parameter 'action' can only be create, delete, grant, granted, or show.")
                    
                interface=self.connection.cursor()
                
                ## kan enten skabe eller slette roller
                if action.upper()!="SHOW":
                    if action.upper()=="CREATE":
                        command=f"CREATE ROLE {name}"
                    elif action.upper()=="DELETE":
                        command=f"DROP ROLE {name}"
                    elif action.upper()=="GRANT":
                        command=f"GRANT {name} TO {user}"
                    interface.execute(command)
                else:
                    if action.upper()=="GRANT":
                        command="SELECT DISTINCT TO_USER AS role_name, TO_HOST FROM mysql.role_edges"
                    else:
                        command="SELECT FROM_USER AS user, FROM_HOST AS user_host, TO_USER AS role, TO_HOST AS role_host FROM mysql.role_edges ORDER BY user"
                    results = interface.fetchall()
                    colnames = [desc[0] for desc in interface.description]
                    print(tabulate(results, headers=colnames, tablefmt="fancy_grid"))
                    self.connection.commit()
                interface.close()
            except Exception as e:
                print(f"ROLE ERROR: {e}")
        else:
            print("No Connection Established. Start A Session First.")
    
    ## funktion til at håndtere roller
    def user(self,name,action,password="",role="",access_from="'%'"):
        if self.connection:
            try:
                if action.upper() not in ["CREATE","DELETE","SHOW"]:
                    raise ValueError("Parameter 'action' can only be create, delete, or show.")
                    
                interface=self.connection.cursor()
                
                if action.upper() in ["CREATE","DELETE"]:
                    if action.upper()=="CREATE":
                        command=f"CREATE USER '{name}'@'{access_from}'"
                        if password!="":
                            command+=f" IDENTIFIED BY '{password}'"
                    elif action.upper()=="DELETE":
                        command=f"DROP USER {name}"
                    interface.execute(command)
                elif action.upper()=="SHOW":
                    command="SELECT user, host FROM mysql.user"
                    results = interface.fetchall()
                    colnames = [desc[0] for desc in interface.description]
                    print(tabulate(results, headers=colnames, tablefmt="fancy_grid"))
                    self.connection.commit()
                interface.close()
            except Exception as e:
                print(f"USER ERROR: {e}")
        else:
            print("No Connection Established. Start A Session First.")
    
    ## funktion til at håndtere privilegier
    def privileges(self, name, action, db, privileges=[], table="*"):
        if self.connection:
            try:
                if type(privileges)==str:
                    privileges=[privileges]
                
                privileges=[x.upper() for x in privileges]
                action=action.upper()
                
                if privileges!=[] and not set(privileges).issubset(["ALL","ALTER","CREATE","DELETE","DROP","GRANT","INDEX","INSERT", "REFERENCES","REVOKE","SELECT","SHOW", "TRIGGER","UPDATE"]):
                    raise ValueError("'privileges' parameter can only consist of either all, alter, create, delete, drop, grant, index, insert, references, revoke, select, show, trigger, and/or update")
                
                if action not in ["GRANT","REVOKE","SHOW"]:
                    raise ValueError("'action' parameter can only be one of either grant, revoke or show")
               
                if "ALL" not in privileges:
                    privileges=", ".join(privileges)
                else:
                    privileges="ALL PRIVILEGES"
              
                interface=self.connection.cursor()
                if action in ["GRANT","REVOKE"]:
                    command=f"{action} {privileges} ON {db}"
                    if table!="":
                        command+=f".{table}"
                    if action=="GRANT":
                        command+=f" TO {name}"
                    else:
                        command+=f" FROM {name}"
                    interface.execute(command)
                    interface.execute("FLUSH PRIVILEGES")
                else:
                    command=f"SHOW GRANTS FOR {name}"
                    results = interface.fetchall()
                    for row in results:
                        print(row[0])
                    self.connection.commit()
                    interface.close()
            except Exception as e:
                print(f"PRIVILEGE ERROR: {e}")
        else:
            print("No Connection Established. Start A Session First.")
    
    ## funktion til at håndtere views
    def view(self,name,action,table,col="*",condition=""):
        if self.connection:
            try:
                if type(col)==list:
                    col=", ".join(col)
                if action.upper() not in ["CREATE","DROP","REPLACE","SHOW"]:
                    raise ValueError("'action' parameter must be either create, drop, replace, or show.")
                
                interface=self.connection.cursor()
                if action.upper()!="SHOW":
                    if action.upper() in ["CREATE","REPLACE"]:
                        if action.upper()=="CREATE":
                            command=f"CREATE SQL SECURITY INVOKER VIEW {name} "
                        else:
                            command=f"CREATE OR REPLACE SQL SECURITY INVOKER VIEW {name} "
                        command+=f"AS SELECT {col} FROM {table}"
                        if condition!="":
                            command+=f" {condition}"
                            
                    elif action.upper()=="DROP":
                        command=f"DROP VIEW IF EXISTS {name} "
                    interface.execute(command)
                else:
                    command=f"SELECT * FROM {name}"
                    results = interface.fetchall()
                    colnames = [desc[0] for desc in interface.description]
                    print(tabulate(results, headers=colnames, tablefmt="fancy_grid"))
                    self.connection.commit()
                interface.close()
                
            except Exception as e:
                print(f"VIEW ERROR: {e}")
        else:
            print("No Connection Established. Start A Session First.")
                
                
        
        
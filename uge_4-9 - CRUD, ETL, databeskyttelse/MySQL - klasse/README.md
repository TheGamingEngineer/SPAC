###################################
###### MySQL - python klasse ######
###################################

MySQL indeholder en klasse kaldet MySQL med funktioner for at forbinde til og operere på en MySQL server via en python konsol. 
Dog skal MySQL initieringsvariablen initieres og en forbindelse oprettes via denne variabel, før at man kan arbejde i MySQL. 
Der er funktioner til at håndtere basale SQL operationer og for selv at skrive sine SQL kommandoer. 


##########################
###### REQUIREMENTS ######
##########################
mysql==0.0.3
mysql-connector==2.2.9
mysql-connector-python==9.2.0
pandas==2.2.3
requests==2.32.3
tabulate==0.9.0


Nedenfor er en række kommandoer, som er tilgængelig i klassen. 

############################################
###### MySQL - kommandoer cheat sheet ######
############################################


#########################
### SERVER FUNKTIONER ###
#########################

## initiering
MySQL=MySQL(host,user,password)

## opret forbindelse
MySQL.connect()

## afslut forbindelse
MySQL.disconnect()

## Vis indhold af server
MySQL.show_server()

## rens serveren for al tilføjet data og databaser
MySQL.clean_server()
### OBS: Den spørger ikke om du er sikker på om du vil gøre det. Så vær forsigtig med at bruge denne funktion.

###########################
### DATABASE FUNKTIONER ###
###########################

## anvend/opret database
MySQL.use_db(databse)

## slet database (OBS: kræver at man er på en anden database)
MySQL.delete_db(database)

## vis indhold af den anvendte databse
MySQL.show_db()

## omdøb database
MySQL.rename_db(oldname,newname)

## eksportér database, som en SQL-fil 
MySQL.backup_db(database, file_path="")
### hvis "file_path" ikke defineres, gemmes databasen i samme mappe, som eksekveringsskriptet. 

## importer SQl database
MySQL.import_sql(file)

########################
### TABEL FUNKTIONER ###
########################

## print tabel
MySQL.show_table(tablename,col="*",fetch="all")
### hvis man kun vil have et vist antal rækker, skal "fetch" sættes til et heltal. 

## slet tabel
MySQL.delete_table(tablename)

## ændre værdier i tabel
MySQL.update_table(tablename, variable_1, value_1, variable_2, value_2)

## opsummér data fra en tabel med datatyper på nuværende database
MySQL.describe(tabelname, colname="*")
### hvis "colname" ikke defineres, vil den lave et overblik over hele tabellen. 
### "colname" skal dog defineres som en streng med navnet på en kolonne, hvis den skal defineres.

## upload tabeller fra .csv-filer til database
MySQL.import_table(file_path)
 
## filtre ny tabel og overfør det til en ny tabel
MySQL.filter_table(tabelname, col="*", filters=[], andor=[])
### "filters" skal være liste af kriterier som skrevet i SQL, som f.eks. "by = Odense"
### "andor" skal være en liste med et antal "AND"/"OR" svarende til "filters" længde minus 1

## vis en grupperet tabel over data fra en tabel på nuværende database
MySQL.group_by(tablename, col="*", grouping=[])
### "grouping" er grupperingskriterier, som skrevet i MySQL

## eksporter en tabel fra serveren til ens lokale computer
MySQL.export_tables(file_path,table_name="*")
### "file_path" indikere stien, hvortil tabellerne skal gemmes
### "table_name" indikere enten alle eller en enkelt tabel, som skal eksporteres

## sammenflet tabeller via en fælles nøgle
MySQL.join_tables(left_tablename, right_tablename, new_tablename, common_key, direction="INNER")
### "common_key" er variablen, som der sammenflettes efter i BEGGE TABELLER ! Så common_key SKAL være tilstede i begge tabeller
### "direction" kan tage "left","right" og "inner". Den kan ikke tage "full"

## importer data fra API
MySQL.import_api(URL,table_name, headers=None, params=None)
### 'headers' og 'params' skal enten være dicts eller None

## vis statistik tabel over en tabel
MySQL.stat_tabel(tabel)
### OBS: Der differentieres udelukkende på hvad der er tal-variabler og hvad der ikke er på baggrund af hvad de er registreret som på MySQL serveren. Alle ikke-tal variabler vil ikke få udregnet gennemsnit, median, std, minimum eller maksimum. 

##########################
### KOLONNE FUNKTIONER ###
##########################

## vis en kolonne med mulige filtre
MySQL.show_col(tablename,col="*",filters=[],andor=[])
### "filters" skal være liste af kriterier som skrevet i SQL, som f.eks. "by = Odense"
### "andor" skal være en liste med et antal "AND"/"OR" svarende til "filters" længde minus 1

## slet en kolonne fra en tabel
MySQL.delete.col(tabelname, colname)

## tilføj kolonne til en tabel
MySQL.add_col(tablename, colname, coltype)
### "coltype" skal være en datatype, som MySQL kan læse

## omdøb kolonne i tabel i den nuværende database
MySQL.rename_col(tablename, old_colname, new_colname)

## ændre datatype for en kolonne i en tabel på den nuværende database
MySQL.change_coltype(tabelname, colname, newtype)
### "newtype" skal defineres som en type, som man kan skrive direkte ind i en MySQL kommando. 

########################
### RÆKKE FUNKTIONER ###
########################

## tilføj en række til en tabel i den nuværende database
MySQL.add_row(tabelname, col=[], vals=[])
### "cols" skal være en liste med kolonnenaven, som man vil udfylde med data
### "vals" skal være en liste med værdier, som skal indsættes ved de tilsvarende kolonnenavne. 

## slet rækker fra en tabel i den nuværende database
MySQL.delete_row(tabelname, col_name, col_val)
### col_name er kolonnenavn, som er indikator
### col_val er værdien, som kolonnen indeholder og med hvilke rækkerne skal slettes

#################
### FREESTYLE ###
#################

## eksekver en hver SQL-kommando, som man kunne tænke sig.
MySQL.freestyle(command)
### "command" skal være en streng med den SQL kommando, som man gerne vil eksekvere. 
### kommandoen skal dog skrives præcist, som man vil skrive det i SQL. 

############
### KEYS ###
############

## funktion til at tilføje primary key til en tabel
MySQL.add_primary_key(table,col)
### "table": tabelnavnet (som en streng) på den tabel på den nuværende database, som man vil tilføje den primære nøgle til. 
### "col": kolonnenavn som streng eller kolonnenavne som liste af strenge. 
        
## funktion til at tilføje unique key til en tabel
MySQL.add_unique_key(table,col)
### "table": tabelnavnet (som en streng) på den tabel på den nuværende database, som man vil tilføje den primære nøgle til. 
### "col": kolonnenavn som streng eller kolonnenavne som liste af strenge. 

################################################
### SIKKERHEDS- OG ADMINISTRERINGSKOMMANDOER ###
################################################

## funktion til at håndtere roller
MySQL.role(action,name="",user="")
### "action": rolle handlingen, som skal eksekveres. kan være en af følgende: "CREATE","DELETE","GRANT","SHOW","GRANTED"
### "name": rollenavnet for den givne rolle, som en streng. Nødvendigt for alle rolle handlinger, undtaget "GRANTED", men er normalt tom.
### "user": en given bruger på MySQL-serveren som en streng. Relevant for "GRANT" og "SHOW", men er normalt tom.

## funktion til at håndtere brugere
MySQL.user(name,action,password="",role="",access_from="'%'")
### "name": brugernavn på den givne bruger, som en streng. Nødvendigt for alle bruger handlinger. 
### "action": bruger handlingen, som skal eksekveres og som en streng. kan være en af følgende: "CREATE","DELETE","SHOW"
### "password": adgangskode, som tildeles den enkelte bruger for at logge ind. er normalt tom, men skal være en streng. 
### "role": rollen, som den givne bruger tildeles ved oprettelse. er normalt tom, så bruger oprettes uden rolle. skrives som en streng.
### "access_from": adgangspunkt, som brugeren får adgang fra. Almindeligvist sat til "'%'", men skal skrives som en streng
    
## funktion til at håndtere privilegier
MySQL.privileges(name, action, db="", privileges=[], table="*")
### "name": bruger-/rolle navn, som privilegierne håndteres for. skal angives som en tekst streng. 
### "action": privilegie handlingen, som skal eksekveres, givet som en tekst streng. kan være en af følgende: "GRANT","REVOKE","SHOW"
### "db": databasenavnet, som bruger/rolle får privilegier til. skal skrives som en tekst streng. Nødvendigt for "GRANT" og "REVOKE" prvilegie handlinger.
### "privileges": privilegier, som håndteres. kan være enten en tekst streng eller en liste af tekst strenge. kan være en af følgende: "ALL","ALTER","CREATE","DELETE","DROP","GRANT","INDEX","INSERT", "REFERENCES","REVOKE","SELECT","SHOW", "TRIGGER","UPDATE"
### "table": tabelnavn, som privilegier skal håndteres for for den enkelte rolle/bruger. skal angives som en tekst streng, men almindeligvist er sat til "*", så handling håndteres for alle tabeller på databasen
    
## funktion til at håndtere views
MySQL.view(name,action,table,col="*",condition="")
### "name": view navnet, som vises ved kaldet. skal skrives som en tekst streng. 
### "action": view handlinger, som skal eksekveres. skal skrives som en tekst streng. kan være en af følgende: "CREATE","DROP","REPLACE","SHOW"
### "table": tabelnavnet, hvorfra kolonnerne til viewet skal hentes. skal skrives som en tekst streng.
### "col": kolonnenavne, som skal vises i det givne view. skal skrives enten som en tekst streng eller en liste af tekst strenge. almindeligvist sat til "*", så alle kolonner ses. 
### "condition": ekstra forbehold, som skal inkluderes i view'et. skal skrives som en tekst streng, og som man vil skrive conditions i SQL. 

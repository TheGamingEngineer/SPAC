from Bio import Entrez, SeqIO
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from urllib.error import HTTPError

Entrez.email="onewingedweeman@gmail.com"
Entrez.tool = "promoter_fetcher_script"
Entrez.api_key = "700c18ded41ed0b7f3bac0cd53c69fa12609"


rige="eukaryot"


if rige=="prokaryot":
    organismer = ["Escherichia coli", 
                  "salmonella enterica subsp. enterica",
                  "mycobacterium tuberculosis",
                  "streptococcus pneumoniae",
                  "pseudomonas putida",
                  "staphylococcus aureus",
                  "neisseria meningitidis",
                  "salmonella enterica",
                  "bacillus subtilis",
                  "Helicobacter pylori",
                  "Klebsiella aerogenes",
                  "Haemophilus influenzae",
                  "Streptococcus gordonii",
                  "Neisseria gonorrhoeae",
                  "corynebacterium ulcerans",
                  "trypanosoma cruzi",
                  "trypanosoma brucei",
                  "leptospira interrogens",
                  "borrelia burgdorferi",
                  "borrelia recurrentis",
                  "treponema pallidum",
                  "mycobacterium leprae",
                  "corynebacterium diphtheriae",
                  "campylobacter coli",
                  "pseudomonas aeruginosa",
                  "vibrio vulnificus",
                  "vibrio parahaemolycus",
                  "yersinia enterocolitica",
                  "klebsiella pneumoniae",
                  "shigella",
                  "clostridium difficile",
                  "clostridium botulinum",
                  "clostridium tetani",
                  "clostridium perfringens",
                  "bacillus cereus",
                  "listeria monocytogenes",
                  "streptococcus agalactiae",
                  "streptococcus pyogenes"]
    
elif rige=="eukaryot":
    organismer=["Homo sapiens",
                "Vulpus vulpus",
                "Canis lupus familiaris",
                "Mus musculus",
                "Rattus norvegicus",
                "Ursus arctos"]

elif rige=="fungi":
    organismer=["penicillium",
                "aspergillus"]

elif rige=="archaea":
    organismer=[]
    
elif rige=="virus":
    organismer=["ebola",
                "corona",
                "herpes",
                "influenza",
                "filo",
                "noro",
                "hepatitis",
                "astro",
                "sapo",
                "arbo",
                "morbilli",
                "polio",
                "papilloma",
                "variola"]

def robust_esearch(term, db="nucleotide", retries=3, delay=3):
    """Robust wrapper til Entrez.esearch med retry-logik."""
    for attempt in range(retries):
        try:
            handle = Entrez.esearch(db=db, term=term, usehistory="y")
            results = Entrez.read(handle)
            handle.close()
            return results
        except RuntimeError as e:
            print(f"RuntimeError (forsøg {attempt+1}/{retries}): {e}")
        except HTTPError as e:
            print(f"HTTPError (forsøg {attempt+1}/{retries}): {e}")
        time.sleep(delay + attempt * 2)  # Øget forsinkelse per forsøg
    print("‼️ Giver op på søgning:", term)
    return None



output_file=f"promoters_{rige}.csv"


batch_size=100

data=pd.DataFrame({"organism":[],"sequence":[],"Description":[],"promoter":[]})

endelige_organismer=[]
for organisme in organismer:
    time.sleep(1.0  + random.uniform(0, 1.0))
    print(f"samler promotere for {organisme}")
    søgeord=f"promoter[Title] AND {organisme}[Organism]"
    resultater = robust_esearch(søgeord)
    
    count = int(resultater["Count"])
    webenv = resultater["WebEnv"]
    query_key = resultater["QueryKey"]
        
    for start in range(0,count,batch_size):
        end=min(count, start+batch_size)
        
        handle= Entrez.efetch(
            db="nucleotide",
            rettype="fasta",
            retmode="text",
            retstart=start,
            retmax=batch_size,
            webenv=webenv,
            query_key=query_key
            )
        
        records = SeqIO.parse(handle,"fasta")
            
        for record in records:
            sande_navn = record.description.split("[")[-1].replace("]","") if "[" in record.description else organisme
            data.loc[len(data)]=[sande_navn, str(record.seq), record.description.replace(",","|"), 1]
            if organisme not in endelige_organismer:
                endelige_organismer.append(organisme)
        handle.close()
        time.sleep(1.0  + random.uniform(0, 1.0))
    
    print(f"samler ikke-promotere for {organisme}")
    søgeord = f"{organisme}[Organism] AND NOT promoter[All Fields]"
    resultater = robust_esearch(søgeord)
    
    cds_count = int(resultater["Count"])
    cds_webenv = resultater["WebEnv"]
    cds_query_key = resultater["QueryKey"]
    
    for start in range(0,cds_count,batch_size):
        end=min(count, start+batch_size)
        
        handle= Entrez.efetch(
            db="nucleotide",
            rettype="fasta",
            retmode="text",
            retstart=start,
            retmax=batch_size,
            webenv=cds_webenv,
            query_key=cds_query_key
            )
        
        records = SeqIO.parse(handle, "fasta")
        
        for record in records:
            if len(str(record.seq)) > 200:
                subseq = str(record.seq)[:200]
                description = record.description.replace(",","|")
                sande_navn = record.description.split("[")[-1].replace("]","") if "[" in record.description else organisme
                data.loc[len(data)]=[sande_navn, subseq, description, 0]
        handle.close()
        time.sleep(1.0  + random.uniform(0, 1.0))
                


counts = data["organism"].value_counts()
#data = data[data["organism"].isin(counts[counts >= 10].index)]

train, temp = train_test_split(data, test_size=0.3, stratify=data["organism"], random_state=38)

test, validation = train_test_split(temp, test_size= 0.5, stratify=temp["organism"], random_state=38)       


train.to_csv(f"training_{rige}.csv",index=False)
test.to_csv(f"test_{rige}.csv",index=False)
validation.to_csv(f"validation_{rige}.csv",index=False)


print(f"færdig! inkludere {len(endelige_organismer)}/{len(organismer)} af de ønskede organismer")
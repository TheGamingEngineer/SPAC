from Bio import Entrez, SeqIO
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from urllib.error import HTTPError

############## Valgfrie indstillinger ##############
curated=None
#curated=" AND srcdb_refseq[PROP]"
#rige="prokaryot"
rige="eukaryot"
#rige="fungi"
#rige="archaea"
#rige="virus"
limit=5000
pooled = False
############## Indstillinger **DO NOT TOUCH** ##############
Entrez.email="onewingedweeman@gmail.com"
Entrez.tool = "promoter_fetcher_script"
Entrez.api_key = "700c18ded41ed0b7f3bac0cd53c69fa12609"


if rige=="prokaryot":
    organismer = ["Escherichia coli",
                  "bacillus subtilis",
                  "Helicobacter pylori",
                  "pseudomonas aeruginosa"]
    
elif rige=="eukaryot":
    organismer=["Homo sapiens",
                "Canis lupus familiaris",
                "Mus musculus",
                "Rattus norvegicus"]

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


batch_size=500

data=pd.DataFrame({"organism":[],"sequence":[],"Description":[],"promoter":[]})

endelige_organismer=[]
max_promoter_længde={}
max_non_længde={}
min_promoter_længde={}
min_non_længde={}

for organisme in organismer:
    max_promoter_længde[organisme]=0
    max_non_længde[organisme]=0
    min_promoter_længde[organisme]=10**6
    min_non_længde[organisme]=10**6
    
    time.sleep(1.0  + random.uniform(0, 1.0))
    print(f"samler promotere for {organisme}")
    søgeord=f"promoter[Title] AND {organisme}[Organism]"
    if curated:
        søgeord+=curated
    resultater = robust_esearch(søgeord)
    
    count = min(limit,int(resultater["Count"]))
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
            #sande_navn = record.description.split("[")[-1].replace("]","") if "[" in record.description else organisme
            data.loc[len(data)]=[organisme, str(record.seq), record.description.replace(",","|"), 1]
            
            if len(str(record.seq)) > max_promoter_længde[organisme]:
                max_promoter_længde[organisme] = len(str(record.seq))
            elif len(str(record.seq)) < min_promoter_længde[organisme]:
                min_promoter_længde[organisme] = len(str(record.seq))
            
            if organisme not in endelige_organismer:
                endelige_organismer.append(organisme)

        handle.close()
        time.sleep(1.0  + random.uniform(0, 1.0))
    
    print(f"samler ikke-promotere for {organisme}")
    #søgeord = f"CDS[Feature Key] AND {organisme}[Organism] AND NOT promoter[All Fields]"
    søgeord=f"CDS[Feature Key] AND {organisme}[Organism] AND NOT promoter[All Fields]"
    if curated:
        søgeord+=curated
    resultater = robust_esearch(søgeord)
    
    new_count = int(resultater["Count"]) if int(resultater["Count"])<count else count
    
    cds_count = min(limit,new_count)
    cds_webenv = resultater["WebEnv"]
    cds_query_key = resultater["QueryKey"]
    
    for start in range(0,cds_count,batch_size):
        end=min(cds_count, start+batch_size)
        
        handle= Entrez.efetch(
            db="nucleotide",
            rettype="fasta",
            retmode="text",
            retstart=start,
            retmax=batch_size,
            webenv=cds_webenv,
            query_key=cds_query_key
            )
        
        records = SeqIO.parse(handle,"fasta")
        n=0
        for record in records:
            #sande_navn = record.description.split("[")[-1].replace("]","") if "[" in record.description else organisme
            data.loc[len(data)]=[organisme, str(record.seq), record.description.replace(",","|"), 0]
            
            
            if len(str(record.seq)) > max_non_længde[organisme]:
                max_non_længde[organisme] = len(str(record.seq))
            elif len(str(record.seq)) < min_non_længde[organisme]:
                min_non_længde[organisme] = len(str(record.seq))
            
            if organisme not in endelige_organismer:
                endelige_organismer.append(organisme)
            
            n+=1
            if n==cds_count:
                break
            
        handle.close()
        time.sleep(1.0  + random.uniform(0, 1.0))

        handle.close()
        time.sleep(random.uniform(0, 1.0))
                


counts = data["organism"].value_counts()
data = data[data["organism"].isin(counts[counts >= 10].index)]


if not pooled:
    train, temp = train_test_split(data, test_size=0.3, stratify=data["organism"], random_state=38)
    test, validation = train_test_split(temp, test_size= 0.5, stratify=temp["organism"], random_state=38)   
    overview={}
    for i in data["organism"].unique():
        Tr = train[train["organism"]==i]
        Te = test[test["organism"]==i]
        V = validation[validation["organism"]==i]
        
        Tr_P = len(Tr[Tr["promoter"]==1])
        Tr_N = len(Tr[Tr["promoter"]==0])
        
        Te_P = len(Te[Te["promoter"]==1])
        Te_N = len(Te[Te["promoter"]==0])
        
        V_P = len(V[V["promoter"]==1])
        V_N = len(V[V["promoter"]==0])
        
        overview[i]=[Tr_P + Te_P + V_P, 
                     Tr_P,
                     Te_P,
                     V_P,
                     Tr_N + Te_N + V_N, 
                     Tr_N,
                     Te_N,
                     V_N,
                     Tr_P + Te_P + V_P + Tr_N + Te_N + V_N]
            
        overview_data = pd.DataFrame(overview,index=["PROMOTERS",
                                                     "training",
                                                     "testing",
                                                     "validation",
                                                     "NON-PROMOTERS",
                                                     "training",
                                                     "testing",
                                                     "validation",
                                                     "TOTAL"])


    train.to_csv(f"training_{rige}.csv",index=False)
    test.to_csv(f"test_{rige}.csv",index=False)
    validation.to_csv(f"validation_{rige}.csv",index=False)
    overview_data.to_csv(f"overview_{rige}_unpooled.csv",index=True)
else:
    overview={}
    for i in data["organism"].unique():
        org_data = data[data["organism"]==i]
        
        org_P = len(org_data[org_data["promoter"]==1])
        org_N = len(org_data[org_data["promoter"]==0])
        
        overview[i]=[org_P + org_N, 
                     org_P,
                     org_N]
        
        overview_data = pd.DataFrame(overview,index=["TOTAL",
                                                     "PROMOTERS",
                                                     "NON-PROMOTERS"])
        
        
        
    data.to_csv(f"data_{rige}.csv",index=False)
    overview_data.to_csv(f"overview_{rige}_pooled.csv",index=True)



print(f"færdig! inkludere {len(endelige_organismer)}/{len(organismer)} af de ønskede organismer")
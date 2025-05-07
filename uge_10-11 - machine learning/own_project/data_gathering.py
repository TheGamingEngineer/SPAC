from Bio import Entrez, SeqIO
import time

Entrez.mail="alexander.andersen@specialisterne.com"

rige="prokaryot"
promoter_or_genome="promoter"

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
    organismer=["homo sapiens",
                "vulpus vulpus",
                "Canis lupus familiaris",
                "mus musculus",
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


output_file=f"{promoter_or_genome}s_{rige}.csv"

if promoter_or_genome=="genome":
    promoter_or_genome="complete genome"


antal_sekvenser=20


batch_size=100

with open(output_file, "w", newline="") as file:
    file.write("organism,sequence,Description\n")

endelige_organismer=[]
for organisme in organismer:
    print(f"samler for {organisme}")
    søgeord=f"{promoter_or_genome}[Title] AND {organisme}[Organism]"
    søgning = Entrez.esearch(db="nucleotide", term=søgeord, usehistory="y")
    resultater = Entrez.read(søgning)
    søgning.close()
    
    count = int(resultater["Count"])
    webenv = resultater["WebEnv"]
    query_key = resultater["QueryKey"]
        
    
    with open(output_file, "a+") as file:
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
                description=record.description.replace(",","|")
                samlet = f"{organisme},{str(record.seq)},{description}"
                line = f'"{samlet}"\n'
                file.write(line)
                if organisme not in endelige_organismer:
                    endelige_organismer.append(organisme)
            handle.close()
            
            time.sleep(0.4)

print(f"færdig! inkludere {len(endelige_organismer)}/{len(organismer)} af de ønskede organismer")
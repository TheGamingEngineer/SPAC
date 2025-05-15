#########################
###### DENNE MAPPE ######
#########################

Denne mappe er brugt som arbejdsmappe for et af mine projekter kaldet "promoter_learner" ('placeholder' navn). 

Via pytroch i python har jeg udviklet en deep learning model ved brug af et 2-lags RNN til at forudsige om givne DNA-sekvenser er promotere og hvilke af 4 eukaryoter, som de tilhører. 

Projektet er opdelt i tre skripts: 
* data_gathering.py
|-> samler promoter- og ikke-promoter sekvenser, samt organisme navne, fra NCBI og tildeler dem en binær variabel som indikator for om de er promoterer eller ej.
|-> hårdkodede instillinger for om man vil have euraryoter, prokaryoter, svampe, viruser eller archaea, men lister er ufuldendte og skal hårdkodes. 
|-> ligeledes er der hårdkodede instillinger for om koder skal tages fra refseq (kurerede eller ej), grænsen for antal hits per organisme eller om data skal separeres eller ej (pooled)
|-> som grundlag er den instillet til at samle sekvenser fra en liste eukaryoter fra hele NCBI-databasen med en grænse på 5000 hits per organisme og at data skal separeres. 
|-> den gemmer data i .jsonl format, samtidigt med at den laver en oversigtstabel i .csv format, som viser antal promotere og ikke-promotere per organisme. 
|-> hvis data separation vælges (pooled=False), vil den opdele data tilfældigt, men ligeligt, i tre datasæt: training_*, test_* og validation_*, samt at oversigtstabellen også viser sekvensfordelingen blandt disse datasæt
|-> ved dataseparation opdeles data ifølge dette forhold: training_*: 75% af data; test_*:15% af data; validation_*:15%

* main.py
|-> laver og træner modeller indtil antallet af epoker er nået. 
|-> anvender et 2-lags RNN med et GRU-lag, med inputstørrelse på 4, skjult størrelse på 64 og sammenlagt 6 neuroner, af hvilke 5 af dem bruges til organisme forudsigelse.
|-> træner modellen for både forudsigele for promoter og for organisme baseret på sekvensen. 
|-> laver en 4-ledet one-hot encoding på både de kendte nukleotider og på de usikre nukleotider med lavere sandsynlighder på de usikre nukleotider.
|-> promoter tabsfunktionen er en BCEWithLogitsLoss (anvender både Sigmoid og BCE), imens organisme tabsfunktionen er krydsentropitab
|-> skriptet lavet to filer: 
||-> promoter_learner.pt: dette er modellen eller vægtene altefter ens instillinger
||-> promoter_models_<dato>.xlsx:  denne fil indeholder træningstab, testtab og accuracy for promoter forudsigelse, organisme forudsigelse og total forudsigelse for hver epoke. 


* model_interpreter.py
|-> anvender både valideringsdatasættet fra data_gathering.py, vægtene promoter_learner.pt fra main.py og funktionerne fra main.py
|-> anvender valideringsdatasættet til at teste modellen imod.
|-> resultaterne af hver forudsigelse bliver samlet i to matricer: en for promoter forudsigelse og en for organisme forudsigelse.
|-> disse matricer bruges så til at lave konfusionsmatricer med basis i heatmaps. 


########################
##### REQUIREMENTS #####
########################
biopython==1.85
json5==0.10.0
matplotlib==3.10.1
matplotlib-inline==0.1.7
pandas==2.2.3
python-dateutil==2.9.0.post0
torch==2.7.0+cu118
torchaudio==2.7.0+cu118
torchvision==0.22.0+cu118
urllib3==2.3.0




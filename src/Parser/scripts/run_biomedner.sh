#!/bin/bash
cd ..
mkdir logs

####################################
#####          NER             #####
####################################

# run neural NER
nohup python biomedner_server.py \
    --model_name_or_path models/biomedner_multi \
    --biomedner_home . \
    --biomedner_port 18894 >> logs/nohup_multi_ner.out 2>&1 &

nohup python maccrobat_server.py \
    --model_name_or_path d4data/biomedical-ner-all \
    --device "cuda:7" \
    --port 18783 >> logs/nohup_maccrobat.out 2>&1 &

cd resources
# run gnormplus
# cd GNormPlusJava
# nohup java -Xmx16G -Xms16G -jar GNormPlusServer.main.jar 18895 >> ../../logs/nohup_gnormplus.out 2>&1 &
# cd ..
####################################
#####     Normalization        #####
####################################

# Disease (working dir: normalization/)
cd normalization
nohup java -Xmx16G -jar normalizers/disease/disease_normalizer_21.jar \
    "inputs/disease" \
    "outputs/disease" \
    "dictionary/dict_Disease_20210630.txt" \
    "normalizers/disease/resources" \
    9 \
    18892 \
    >> ../../logs/nohup_disease_normalize.out 2>&1 &


# Gene (working dir: normalization/normalizers/gene/, port:18888)
cd normalizers/gene
nohup java -Xmx20G -jar gnormplus-normalization_21.jar \
    18888 \
    >> ../../../../logs/nohup_gene_normalize.out 2>&1 &
cd ../../../..
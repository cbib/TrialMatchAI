#!/bin/bash
cd ..
#!/bin/bash

if [ ! -d "logs" ]; then
  mkdir logs
fi
####################################
#####          NER             #####
####################################

# run neural NER
nohup python biomedner_server.py \
    --model_name_or_path models/finetuned_model_roberta \
    --biomedner_port 18894 >> logs/nohup_multi_ner.out 2>&1 &

nohup python gner_server.py \
    --model_name_or_path gliner-community/gliner_large-v2.5 \
    --gner_port 18783 >> logs/nohup_gner.out 2>&1 &

####################################
#####     Normalization        #####
####################################
cd resources
# Disease (working dir: normalization/)
cd normalization
nohup java -Xmx16G -jar normalizers/disease/disease_normalizer_21.jar \
    "inputs/disease" \
    "outputs/disease" \
    "dictionary/dict_Disease.txt" \
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
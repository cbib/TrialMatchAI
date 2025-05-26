#!/usr/bin/env bash
set -euo pipefail

#=== LOAD .env ===#
if [ ! -f .env ]; then
  echo "[ERROR] .env file not found."
  exit 1
fi

# shellcheck disable=SC1091
source .env

#=== CONFIGURATION FROM ENV ===#
STACK_VERSION="${STACK_VERSION:-8.13.4}"
CLUSTER_NAME="${CLUSTER_NAME:-apptainer-cluster}"
ES_IMAGE="docker.elastic.co/elasticsearch/elasticsearch:${STACK_VERSION}"

BASE_DIR="$(pwd)"
CERTS_DIR="$BASE_DIR/certs"
CONFIG_DIR="$BASE_DIR/config"
DATA_DIR="$BASE_DIR/data"
LOGS_DIR="$BASE_DIR/logs"
SIF_DIR="$BASE_DIR/sif"
TMP_CONFIG="$BASE_DIR/tmp-config"

ES_PORT1="${ES_PORT:-9200}"
ES_PORT2=$((ES_PORT1 + 1))
ES_PORT3=$((ES_PORT1 + 2))

ELASTIC_PASSWORD="${ELASTIC_PASSWORD:?ELASTIC_PASSWORD not set in .env}"

#=== PREPARE FOLDERS ===#
mkdir -p "$CONFIG_DIR/es01" "$CONFIG_DIR/es02" "$CONFIG_DIR/es03"
mkdir -p "$DATA_DIR/es01" "$DATA_DIR/es02" "$DATA_DIR/es03"
mkdir -p "$LOGS_DIR" "$SIF_DIR" "$TMP_CONFIG"

#=== BUILD SIF IMAGE IF NEEDED ===#
if [ ! -f "$SIF_DIR/es.sif" ]; then
  echo "[INFO] Building Elasticsearch SIF..."
  apptainer build "$SIF_DIR/es.sif" "docker://$ES_IMAGE"
fi

#=== CLEAN CONFIG EXTRACTION DIR ===#
rm -rf "$TMP_CONFIG"/*

#=== EXTRACT DEFAULT CONFIG FILES ===#
echo "[INFO] Extracting default Elasticsearch config files..."
apptainer exec --bind "$TMP_CONFIG:/mnt/tmp" "$SIF_DIR/es.sif" \
  bash -c 'cp -r /usr/share/elasticsearch/config/* /mnt/tmp/'

#=== PREPARE NODE CONFIGS ===#
for NODE in es01 es02 es03; do
  cp -r "$TMP_CONFIG"/* "$CONFIG_DIR/$NODE/"
  cp "$CERTS_DIR/ca.crt" "$CONFIG_DIR/$NODE/"
  cp "$CERTS_DIR/$NODE/$NODE.crt" "$CONFIG_DIR/$NODE/"
  cp "$CERTS_DIR/$NODE/$NODE.key" "$CONFIG_DIR/$NODE/"

  cat > "$CONFIG_DIR/$NODE/elasticsearch.yml" <<YML
node.name: $NODE
path.data: /usr/share/elasticsearch/data
path.logs: /usr/share/elasticsearch/logs
network.host: 127.0.0.1
cluster.name: $CLUSTER_NAME
discovery.seed_hosts: ["127.0.0.1:9300", "127.0.0.1:9301", "127.0.0.1:9302"]
cluster.initial_master_nodes: ["es01", "es02", "es03"]
bootstrap.memory_lock: false
xpack.security.enabled: true
xpack.security.http.ssl.enabled: true
xpack.security.http.ssl.key: $NODE.key
xpack.security.http.ssl.certificate: $NODE.crt
xpack.security.http.ssl.certificate_authorities: ["ca.crt"]
xpack.security.transport.ssl.enabled: true
xpack.security.transport.ssl.verification_mode: certificate
xpack.security.transport.ssl.key: $NODE.key
xpack.security.transport.ssl.certificate: $NODE.crt
xpack.security.transport.ssl.certificate_authorities: ["ca.crt"]
xpack.license.self_generated.type: ${LICENSE:-basic}
YML

    echo "[INFO] Injecting password into keystore for $NODE..."

    apptainer exec \
    --bind "$CONFIG_DIR/$NODE:/usr/share/elasticsearch/config" \
    "$SIF_DIR/es.sif" \
    bash -c "
        cd /usr/share/elasticsearch/config
        rm -f elasticsearch.keystore
        elasticsearch-keystore create
        echo '$ELASTIC_PASSWORD' | elasticsearch-keystore add -x 'bootstrap.password'
    "
done

#=== LAUNCH NODES ===#
launch_node() {
  NODE=$1
  PORT=$2
  TRANSPORT_PORT=$3
  echo "[INFO] Launching $NODE on port $PORT..."
  apptainer exec \
    --env ES_JAVA_OPTS="-Xms512m -Xmx512m" \
    --bind "$CONFIG_DIR/$NODE:/usr/share/elasticsearch/config" \
    --bind "$DATA_DIR/$NODE:/usr/share/elasticsearch/data" \
    --bind "$LOGS_DIR:/usr/share/elasticsearch/logs" \
    "$SIF_DIR/es.sif" \
    elasticsearch \
    -E http.port=$PORT \
    -E transport.port=$TRANSPORT_PORT > "$LOGS_DIR/$NODE.log" 2>&1 &
}

launch_node es01 $ES_PORT1 9300
sleep 10
launch_node es02 $ES_PORT2 9301
sleep 10
launch_node es03 $ES_PORT3 9302
sleep 10

#=== WAIT FOR ES01 TO BE READY ===#
echo "[INFO] Waiting for es01 to be ready..."
until curl -s --cacert "$CERTS_DIR/ca.crt" -u elastic:"$ELASTIC_PASSWORD" \
  https://localhost:$ES_PORT1/_cluster/health?pretty | grep -q '"status"'; do
  echo -n "."
  sleep 5
done
echo -e "\n[INFO] Elasticsearch cluster is up."

echo "[INFO] Access Elasticsearch at: https://localhost:$ES_PORT1"

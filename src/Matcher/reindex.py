from elasticsearch import Elasticsearch, ConnectionTimeout, ConnectionError
import time
from elasticsearch.helpers import bulk

# Initialize Elasticsearch client with proper timeout and retries
es = Elasticsearch(
    hosts=["https://localhost:9200"],
    ca_certs="certs/ca.crt",
    basic_auth=("elastic", "QQ7wWoB_WnKe*L*X9tAW"),
    request_timeout=600,  # Set proper request timeout for long-running tasks
    max_retries=10,
    retry_on_timeout=True
)

# Step 1: Create the new index with updated mappings
new_index_name = "eligibility_criteria_v2"
old_index_name = "eligibility_criteria"
vector_dims = 1024  # The dimension of the dense vector

new_index_mappings = {
    "settings": {
        "analysis": {
            "analyzer": {
                "standard_lowercase": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "criteria_id": {"type": "keyword"},
            "criterion": {"type": "text", "analyzer": "standard_lowercase"},
            "criterion_vector": {
                "type": "dense_vector",
                "dims": vector_dims,
                "index": True,
                "similarity": "cosine",
                "index_options": {
                    "type": "hnsw",
                    "m": 16,
                    "ef_construction": 100
                }
            },
            "entities": {
                "type": "nested",
                "properties": {
                    "normalized_id": {"type": "keyword"},
                    "synonyms": {"type": "text", "analyzer": "standard_lowercase"},
                    "entity": {"type": "text", "analyzer": "standard_lowercase"},
                    "class": {"type": "keyword"}
                }
            },
            # The updated nct_id field to be searchable (indexed)
            "nct_id": {"type": "keyword", "index": True}
        }
    }
}

# Create the new index with updated mappings if it doesn't exist
if not es.indices.exists(index=new_index_name):
    es.indices.create(index=new_index_name, body=new_index_mappings)
    print(f"Created new index: {new_index_name} with updated mappings.")
else:
    print(f"Index {new_index_name} already exists.")


# Step 2: Get the last processed document in the new index using `search_after`
def get_last_processed_document(es, index_name):
    try:
        response = es.search(
            index=index_name,
            body={
                "size": 1,
                "sort": [{"criteria_id": "desc"}],  # Sort by criteria_id (or `_id`)
                "_source": ["criteria_id"]
            }
        )

        if response["hits"]["hits"]:
            last_doc = response["hits"]["hits"][0]
            return last_doc["_source"]["criteria_id"]
        return None

    except (ConnectionTimeout, ConnectionError) as e:
        print(f"Error fetching last processed document: {e}")
        return None


# Step 3: Use `search_after` to handle large datasets and reindex remaining documents in batches
def reindex_with_search_after(es, source_index, dest_index, last_processed_id):
    # Initial `search_after` value (None for the first request)
    search_after_value = None

    try:
        while True:
            # Build the base query
            query = {
                "size": 1000,  # Adjust batch size as needed
                "sort": [{"criteria_id": "asc"}],  # Sort by ascending order of `criteria_id`
                "query": {
                    "range": {
                        "criteria_id": {
                            "gt": last_processed_id  # Continue reindexing documents after this `criteria_id`
                        }
                    }
                }
            }

            # Add the `search_after` parameter only if it has a valid value
            if search_after_value:
                query["search_after"] = search_after_value

            # Fetch documents from the source index
            response = es.search(index=source_index, body=query)
            hits = response["hits"]["hits"]

            if not hits:
                print("No more documents to reindex.")
                break  # Exit when there are no more documents to reindex

            # Collect the last hit's `criteria_id` for `search_after` in the next iteration
            search_after_value = hits[-1]["sort"]

            # Prepare the bulk reindex request
            bulk_requests = []
            for hit in hits:
                doc = hit["_source"]
                bulk_requests.append({
                    "_op_type": "index",
                    "_index": dest_index,
                    "_id": hit["_id"],
                    "_source": doc
                })

            # Perform bulk indexing
            success, failed = bulk(es, bulk_requests)
            print(f"Reindexed {success} documents, {failed} failed.")

    except (ConnectionTimeout, ConnectionError) as e:
        print(f"Error during reindexing: {e}")
        time.sleep(30)  # Retry after a delay
        reindex_with_search_after(es, source_index, dest_index, last_processed_id)


# Step 4: Start or resume the reindexing process
last_processed_id = get_last_processed_document(es, new_index_name)

if last_processed_id:
    print(f"Resuming reindexing from criteria_id: {last_processed_id}")
    reindex_with_search_after(es, old_index_name, new_index_name, last_processed_id)
else:
    # If no documents are found, reindex from the start
    print("No documents found in the new index, starting reindexing from the beginning.")
    reindex_with_search_after(es, old_index_name, new_index_name, None)


# Step 5: Verify the new index and document count
try:
    mapping = es.indices.get_mapping(index=new_index_name)
    print(f"Updated mapping for `nct_id`: {mapping[new_index_name]['mappings']['properties']['nct_id']}")

    count = es.count(index=new_index_name)
    print(f"Document count in new index '{new_index_name}': {count['count']}")
except (ConnectionTimeout, ConnectionError) as e:
    print(f"Error fetching index details: {e}")

import subprocess
import os
import time
import requests

def start_multiner_server():
    current_directory = os.getcwd()
    working_directory = "../resources/BERN2/scripts/"
    os.chdir(working_directory)
    run_path = "run_bern2.sh"
    stop_path = "stop_bern2.sh"
    print("Stopping any existing Multi-NER server instance.")
    stop_process = subprocess.Popen(["bash", stop_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    stop_process.wait()
    print("Activating Mutli-NER Server... This can take approx. 1 minute")
    try:
        subprocess.Popen(["bash", run_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        timeout = 120  # Adjust this value as needed
        # Define the server's URL that you want to check
        server_url = "http://localhost:8888"  # Update with the actual URL
        # Wait for the server to become available or reach the timeout
        start_time = time.time()
        while True:
            try:
                # Send a request to the server to check its availability
                response = requests.get(server_url)
                response.raise_for_status()  # Raises an exception for non-2xx status codes
                break  # Server is available, exit the loop
            except (requests.ConnectionError, requests.HTTPError) as e:
                if time.time() - start_time >= timeout:
                    print(f"Server did not become available within {timeout} seconds.")
                    break  # Timeout reached
                else:
                    # Wait for a short time before checking again
                    time.sleep(1)

        # Continue with other tasks
        print("Server is now available.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing the script: {e}")
    os.chdir(current_directory)
    
def stop_multiner_server():
    current_directory = os.getcwd()
    working_directory = "../resources/BERN2/scripts/"
    os.chdir(working_directory)
    stop_path = "stop_bern2.sh"
    stop_process = subprocess.Popen(["bash", stop_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    stop_process.wait()
    print("Multi-NER server instance terminated.")

    

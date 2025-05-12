### **TrialMatchAI**

<img src="img/logo.png" alt="Logo" align="right" width="200" height="200"> 

An AI-driven tool designed to match patients with the most relevant clinical trials. Leveraging state-of-the-art Large Language Models (LLMs), Natural Language Processing (NLP), and Explainable AI (XAI), TrialMatchAI structures trial documentation and patient data to provide transparent, personalized recommendations.

---

## ⚠️ Disclaimer
At this stage, TrialMatchAI is still under active development and largely a **prototype** provided for research and informational purposes only. It is **NOT** medical advice and should not replace consultation with qualified healthcare professionals.

---

## 🔍 Key Features

- **AI-Powered Matching**: Utilizes advanced LLMs to parse complex eligibility criteria and patient records (including unstructured notes and genetic reports).
- **Personalized Recommendations**: Tailors trial suggestions based on each patient’s unique clinical history and genomic profile.
- **Explainable Insights**: Provides clear, chain-of-thought explanations for every recommended trial, enhancing trust and interpretability.
- **Real-Time Updates**: Maintains an up-to-date database of recruiting trials.
- **Scalable Architecture**: Dockerized components enable easy deployment of Elasticsearch indices and indexing pipelines.

---

## ⚙️ System Requirements

- **OS**: Linux or macOS
- **Docker & Docker Compose**: For running the Elasticsearch container
- **Python**: ≥ 3.8
- **GPU**: NVIDIA (e.g., H100) with ≥ 60 GB VRAM (recommended for large-scale processing)
- **Disk Space**: ≥ 100 GB free (for data and indices)

---

## 🚀 Installation & Setup

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/cbib/TrialMatchAI.git  
   cd TrialMatchAI  
   ```  

2. **Ensure the Repository Is Up to Date**  
   ```bash  
   git pull origin main  
   ```  

3. **Make the Setup Script Executable**  
   ```bash
   chmod +x setup.sh
   ```

4. **(Optional) Configure Elasticsearch Password**  
   - Open the `.env` file located in the `docker/` folder.  
   - Update the `ELASTIC_PASSWORD` variable to your desired secure password.  
   ```dotenv
   # docker/.env
   ELASTIC_PASSWORD=YourNewPassword
   ```

4a. **(Optional) Sync `config.json` Password**  
   If you updated `ELASTIC_PASSWORD` above, open `config.json` in the repo root and update the Elasticsearch password field to match:  
   ```json
   {
  "elasticsearch": {
    "host": "https://localhost:9200",
    "username": "elastic",
    "password": "YourNewPassword",
    .
    .
     },
     ...
   }
   ```

5. **Run the Setup Script**  
   ```bash
   ./setup.sh
   ```  
   - Installs Python dependencies  
   - Downloads datasets, resources, and model archives from Zenodo  
   - Verifies GPU availability  
   - Builds the Elasticsearch container via Docker Compose  
   - Launches indexing pipelines in the background  
   - **Estimated Time**: ~60–90 minutes (depending on hardware)  

---

## 🎯 Usage Example

Run the matcher on a sample input directory:

```bash
python -m src.Matcher.main 
```

Results are saved under `results/`, with detailed criterion-level explanations for each recommended trial.

---

## 🤝 Contributing

We welcome community contributions! To contribute:

1. Fork the repository.  
2. Create a feature branch: `git checkout -b feature/YourFeature`.  
3. Commit your changes and push to your branch.  
4. Open a Pull Request against `main`.

Please follow our code style and include tests where applicable.

---

## 🙋 Support & Contact

For questions, issues, or feature requests, open an issue on GitHub or reach out to:

- **Email**: [abdallahmajd7@gmail.com](mailto:abdallahmajd7@gmail.com)

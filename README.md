# CyberCerberus: domain classification with URL, text classification and graph aggregation

## **Install UV on Windows & Linux**

### **Windows**
1. **Using PowerShell** (admin recommended):
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   - Restart your terminal after installation.

### **Linux/macOS**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- Restart your shell or run: `source $HOME/.local/bin/env`

---

## **Initialize & Activate Environment**

### **1. Sync dependencies** (both Windows and Linux)
```bash
uv sync
```

### **2. Activate .venv**

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

## **Weights and data**
### **1. Weights**
Create a weights/ folder. Put the `.npy` weights inside of the folder.

### **2. Data**
Put the `.parquet` file in the data folder. 

## **Training and inference**
### **Training**
Open [prediction notebook](notebooks/final_prediction.ipynb), select the CyberCerberus kernel and Run all.

### **Inference**
Still working, will push later!
# PS-09: Plant Seedling Species Classification
**Domain:** Agriculture & Food | Precision Agriculture & Edge-AI Weed Control  
**Competition Dataset:** [Kaggle: Plant Seedlings Classification (Aarhus University)](https://www.kaggle.com/c/plant-seedlings-classification)

---

## 📌 Problem Overview
Telling crop seedlings from weed seedlings at an early growth stage allows farmers and autonomous field robots to remove weeds before they compete with crops for light, moisture, and soil nutrients. Broadcast spraying wastes up to 85% of herbicide volume and harms the environment. Autonomous optical spot-sprayers require high-speed computer vision to distinguish between **12 seedling species**, several of which appear virtually identical at early vegetative stages (a fine-grained visual classification challenge).

### The 12 Species
| Category | Species Common Name | Scientific Name | Botanical Family |
| :--- | :--- | :--- | :--- |
| **Crop** | Maize | *Zea mays* | Poaceae (Monocot) |
| **Crop** | Common wheat | *Triticum aestivum* | Poaceae (Monocot) |
| **Crop** | Sugar beet | *Beta vulgaris* | Amaranthaceae (Dicot) |
| **Weed** | Black-grass | *Alopecurus myosuroides* | Poaceae (Monocot) |
| **Weed** | Loose Silky-bent | *Apera spica-venti* | Poaceae (Monocot) |
| **Weed** | Charlock | *Sinapis arvensis* | Brassicaceae (Dicot) |
| **Weed** | Cleavers | *Galium aparine* | Rubiaceae (Dicot) |
| **Weed** | Common Chickweed | *Stellaria media* | Caryophyllaceae (Dicot) |
| **Weed** | Fat Hen | *Chenopodium album* | Amaranthaceae (Dicot) |
| **Weed** | Scentless Mayweed | *Tripleurospermum inodorum* | Asteraceae (Dicot) |
| **Weed** | Shepherd's Purse | *Capsella bursa-pastoris* | Brassicaceae (Dicot) |
| **Weed** | Small-flowered Cranesbill | *Geranium pusillum* | Geraniaceae (Dicot) |

---

## 🗂️ Project Structure
```
├── RegisterNo_PS09.ipynb      # End-to-end trained Jupyter Notebook (all cell outputs & figures embedded)
├── RegisterNo_PS09_Report.md  # Comprehensive Agronomic & Computer Vision Technical Report
├── README.md                  # Project overview and documentation
└── .gitignore                 # Git ignore rules (caches, weights, raw dataset archives)
```

---

## 🚀 Quickstart & Notebook Execution

### Open & Run the Jupyter Notebook
Open [`RegisterNo_PS09.ipynb`](file:///c:/Users/USER/Desktop/Activity/RegisterNo_PS09.ipynb) in VS Code, JupyterLab, or Google Colab:
- **Step 1:** Kaggle authentication and credentials setup
- **Step 2:** Automatic archive ingestion from `plant-seedlings-classification.zip` / `kagglehub`
- **Step 3:** Exploratory Data Analysis (EDA, 12-species counts, visual sample gallery)
- **Step 4:** Stratified 70/15/15 train-val-test split (fixed seed 42)
- **Step 5:** Preprocessing, $360^\circ$ rotational augmentation & ImageNet normalization
- **Step 6:** Custom CNN baseline vs. Pretrained ResNet-34 transfer learning models
- **Step 7:** Model training with label smoothing ($\epsilon=0.1$) & cosine annealing scheduler
- **Step 8:** Loss & accuracy convergence curve visualization
- **Step 9:** Single evaluation on hold-out test set (89.76% accuracy, 87.96% Macro F1) & confusion matrix
- **Step 10:** Misclassification analysis with sample images & botanical root-cause explanations
- **Step 11:** Explicit treatment of specific challenge (fine-grained look-alikes & class imbalance)

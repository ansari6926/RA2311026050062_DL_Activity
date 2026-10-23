# Project Report: PS-09 Plant Seedling Species Classification
**Domain:** Agriculture & Food | Precision Agriculture & Edge-AI Weed Management  
**File Name:** `RegisterNo_PS09_Report.md`  
**Companion Notebook:** [`RegisterNo_PS09.ipynb`](file:///c:/Users/USER/Desktop/Activity/RegisterNo_PS09.ipynb)  
**Dataset:** Aarhus University / Kaggle Plant Seedlings Classification Competition  

---

## 🌾 1. Executive Summary & Problem Formulation
In modern precision agriculture, **Site-Specific Weed Management (SSWM)** enables farmers to protect crops by eliminating weeds during the earliest vegetative growth stages. Early weed competition severely deprives crop seedlings of light, water, and soil nutrients, leading to documented crop yield losses ranging from **30% to over 50%**.

### The Conventional Challenge
Traditional farming relies on blanket herbicide broadcast spraying, which coats entire agricultural fields in chemical weedkillers. This practice:
1. Wastes up to **85%** of applied chemical volume on bare soil and non-target crops.
2. Causes dangerous toxic runoff into regional groundwater and aquatic ecosystems.
3. Accelerates herbicide resistance among aggressive weed populations (notably *Black-grass* in Europe).
4. Substantially increases operating costs for farmers.

### The Autonomous Solution
Deploying tractor-mounted camera booms and autonomous agricultural robots equipped with edge-AI deep learning inference. These systems capture top-down seedling imagery in real time, classify each plant into one of **12 species**, and selectively actuate precision micro-spray nozzles or mechanical hoes solely over weed seedlings while leaving crop plants untouched.

---

## 🌿 2. The 12 Agricultural Species (Crops vs. Weeds)

The dataset consists of top-view images of 12 distinct arable plant species at early seedling emergence:

| # | Species Common Name | Botanical Name | Plant Family | Category | Agronomic Significance |
|---|---|---|---|---|---|
| 1 | **Maize** | *Zea mays* | Poaceae (Monocot) | **Crop** | Vital staple cereal crop; broad upright seedling leaves |
| 2 | **Common wheat** | *Triticum aestivum* | Poaceae (Monocot) | **Crop** | Primary grain crop; narrow linear foliage |
| 3 | **Sugar beet** | *Beta vulgaris* | Amaranthaceae (Dicot) | **Crop** | Major industrial sugar crop; fleshy rounded cotyledons |
| 4 | **Black-grass** | *Alopecurus myosuroides* | Poaceae (Monocot) | **Weed** | Highly aggressive, herbicide-resistant grass weed |
| 5 | **Loose Silky-bent** | *Apera spica-venti* | Poaceae (Monocot) | **Weed** | Fast-spreading wind-pollinated grass weed in cereals |
| 6 | **Charlock** | *Sinapis arvensis* | Brassicaceae (Dicot) | **Weed** | Competitive wild mustard broadleaf weed |
| 7 | **Cleavers** | *Galium aparine* | Rubiaceae (Dicot) | **Weed** | Climbing weed that fouls combine harvesters |
| 8 | **Common Chickweed** | *Stellaria media* | Caryophyllaceae (Dicot) | **Weed** | Dense ground-matting competitive broadleaf weed |
| 9 | **Fat Hen** | *Chenopodium album* | Amaranthaceae (Dicot) | **Weed** | High-nitrogen consumer with mealy leaf coating |
| 10 | **Scentless Mayweed** | *Tripleurospermum inodorum* | Asteraceae (Dicot) | **Weed** | Deep-rooting composite weed with feathery leaves |
| 11 | **Shepherd's Purse** | *Capsella bursa-pastoris* | Brassicaceae (Dicot) | **Weed** | Prolific seed producer with heart-shaped siliques |
| 12 | **Small-flowered Cranesbill** | *Geranium pusillum* | Geraniaceae (Dicot) | **Weed** | Low-profile weed with lobed, pubescent foliage |

---

## 🎯 3. Specific Challenge: Fine-Grained Categorization & Class Imbalance (Rule 6)

The competition problem statement specifies:
> *"Classify top-view seedling images into 12 species, several of which look very similar at this growth stage (a fine-grained problem)."*

### 3.1 The Fine-Grained Visual Bottleneck
Fine-grained image classification is characterized by **high intra-class variance** alongside **extremely high inter-class similarity**:
1. **The Monocot Grass Dilemma (*Black-grass* vs. *Loose Silky-bent* vs. *Common wheat*):**
   All three species belong to the botanical family *Poaceae*. At the 1-to-2 leaf seedling stage, ligules, auricles, and floral spikes have not yet developed. They all exhibit erect, needle-like green blades with identical parallel venation. A mistake here is catastrophic: misclassifying *Black-grass* as *Common wheat* allows an herbicide-resistant weed to smother the crop; misclassifying *Common wheat* as *Black-grass* triggers the sprayer to destroy the farmer's crop.
2. **The Broadleaf Dicot Confusion (*Charlock* vs. *Shepherd's Purse* vs. *Common Chickweed*):**
   Early cotyledon leaves of broadleaf seedlings share similar rounded spatulate shapes before true lobed leaves emerge.
3. **Class Imbalance:**
   Sample representation varies from **654 images** (*Loose Silky-bent*) down to **221 images** (*Maize*), biasing unregularized models toward majority classes.

### 3.2 Implemented Algorithmic Mitigations
1. **Deep Transfer Learning Feature Reuse:** Leveraging ResNet-34 pre-trained on ImageNet provides rich spatial hierarchical filters capable of discerning minute leaf margin serrations and micro-textures.
2. **Label Smoothing Regularization ($\epsilon = 0.1$):**
   $$y'_k = (1 - \epsilon) \cdot y_k + \frac{\epsilon}{K}$$
   Prevents overconfidence on borderline visual specimens, allowing the model to smoothly handle morphological convergence.
3. **$360^\circ$ Rotational & Top-Down Invariance Augmentations:**
   Eliminates camera orientation shortcuts and forces the model to learn radial botanical morphology.
4. **Stratified Splitting:**
   Guarantees that class proportions in the training ($70\%$), validation ($15\%$), and hold-out test ($15\%$) sets are identical.

---

## 🔬 4. Methodology & Experimental Setup

### 4.1 Data Ingestion & Splitting (Rule 5a, 5b)
- **Random Seed:** Fixed to `42` (`torch`, `numpy`, `random`, `torch.cuda`).
- **Splits:**
  - **Training Set (70%):** ~3,325 images.
  - **Validation Set (15%):** ~712 images (model selection and checkpointing).
  - **Hold-Out Test Set (15%):** ~713 images (single unbiased evaluation).

### 4.2 Preprocessing & Augmentation (Rule 5d)
- Downward-facing camera simulation:
  - Random Resized Crop ($224 \times 224$, scale $0.75 - 1.0$)
  - Random Horizontal Flip ($p=0.5$)
  - Random Vertical Flip ($p=0.5$)
  - Random Rotation ($\pm 180^\circ$)
  - Color Jitter (Brightness $\pm 0.2$, Contrast $\pm 0.2$, Saturation $\pm 0.2$)
  - ImageNet Channel Normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)

### 4.3 Model Architectures (Rule 5c)
1. **Model A: Custom CNN Trained from Scratch**
   - 4-stage convolutional backbone with progressive channel depth ($32 \rightarrow 64 \rightarrow 128 \rightarrow 256$).
   - Batch Normalization and ReLU activation after each convolution.
   - MaxPooling ($2 \times 2$) and Adaptive Average Pooling ($1 \times 1$).
   - Fully connected classification head with Dropout ($0.4$).
2. **Model B: Transfer Learning (ResNet-34)**
   - Pre-trained ImageNet-1k weights.
   - Replaced final FC layer with:
     `Dropout(0.4) -> Linear(512, 256) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(256, 12)`
   - Optimized with AdamW ($lr = 3 \times 10^{-4}$, weight decay $= 10^{-2}$) and Cosine Annealing scheduler.

---

## 📊 5. Results & Comparative Performance (Rule 5e)

### 5.1 Architecture Comparison

| Model Architecture | Pretrained Weights | Training Loss | Val Loss | Test Accuracy | Test Macro F1 |
|---|---|---|---|---|---|
| **Custom CNN (Scratch)** | None (Random Init) | 2.0941 | 1.9540 | 38.33% | 34.20% |
| **ResNet-34 (Transfer Learning)** | ImageNet-1k | **0.8638** | **0.7478** | **89.76%** | **87.96%** |

*Finding:* Transfer learning improves test accuracy by **+51.43 percentage points** over training from scratch, validating that fine-grained botanical classification requires rich pre-trained edge, texture, and contour representations.

### 5.2 Per-Class Classification Report on Hold-Out Test Set (Rule 5e)

| Species Name | Class Type | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|---|
| **Black-grass** | Weed | 0.6129 | 0.4872 | 0.5429 | 39 |
| **Charlock** | Weed | 0.9500 | 0.9828 | 0.9661 | 58 |
| **Cleavers** | Weed | 0.9512 | 0.9070 | 0.9286 | 43 |
| **Common Chickweed** | Weed | 0.9082 | 0.9674 | 0.9368 | 92 |
| **Common wheat** | Crop | 0.8929 | 0.7576 | 0.8197 | 33 |
| **Fat Hen** | Weed | 0.9692 | 0.8750 | 0.9197 | 72 |
| **Loose Silky-bent** | Weed | 0.8333 | 0.8673 | 0.8500 | 98 |
| **Maize** | Crop | 0.8919 | 1.0000 | 0.9429 | 33 |
| **Scentless Mayweed** | Weed | 0.8621 | 0.9615 | 0.9091 | 78 |
| **Shepherds Purse** | Weed | 1.0000 | 0.7059 | 0.8276 | 34 |
| **Small-flowered Cranesbill** | Weed | 0.9868 | 1.0000 | 0.9934 | 75 |
| **Sugar beet** | Crop | 0.8750 | 0.9655 | 0.9180 | 58 |
| **Macro Average** | - | **0.8945** | **0.8731** | **0.8796** | **713** |
| **Weighted Average** | - | **0.8973** | **0.8976** | **0.8945** | **713** |

---

## 🔍 6. Error Analysis: Misclassified Test Images & Root Cause Explanations (Rule 5f)

To rigorously diagnose the failure modes of the classifier on the hold-out test set, four representative misclassifications were identified and analyzed:

### Error Sample 1: Black-grass $\rightarrow$ Predicted as Loose Silky-bent
* **Image Properties:** 2-leaf emergent seedling against dark gravelly soil substrate.
* **Botanical Cause:** *Black-grass* (*Alopecurus myosuroides*) and *Loose Silky-bent* (*Apera spica-venti*) are both narrow-leaf grass weeds. Before ligule formation at the 3-leaf stage, their leaf blade width and venation patterns are indistinguishable to human agronomists and convolutional kernels alike.
* **Computer Vision Failure:** High soil-to-leaf area ratio causes the feature extractor to capture predominantly background noise rather than leaf margin curvature.

### Error Sample 2: Common Chickweed $\rightarrow$ Predicted as Small-flowered Cranesbill
* **Image Properties:** Dense cluster of young cotyledons with mutual leaf overlap.
* **Botanical Cause:** Both species are low-profile broadleaf weeds with rounded early leaves. Overlapping leaves cast shadows that hide the subtle basal leaf lobes of Cranesbill.
* **Computer Vision Failure:** Occlusion alters the perceived leaf contour, causing the pooling layer to aggregate features that match Chickweed's smooth oval templates.

### Error Sample 3: Shepherd's Purse $\rightarrow$ Predicted as Charlock
* **Image Properties:** Very early seedling (cotyledon stage) prior to true leaf emergence.
* **Botanical Cause:** Both species are members of the *Brassicaceae* family. Cotyledons of crucifers are morphologically identical; true dentate leaves appear only at later growth stages.
* **Computer Vision Failure:** Intrinsic Bayes error limit—the visual information necessary for discrimination does not yet exist on the plant.

### Error Sample 4: Common wheat $\rightarrow$ Predicted as Black-grass
* **Image Properties:** Single elongated shoot emerging at an acute angle.
* **Botanical Cause:** Seedling wheat exhibits linear monocot morphology matching wild grass weeds.
* **Computer Vision Failure:** Subtle auricle hairs (which distinguish wheat from black-grass) are sub-pixel at $224 \times 224$ image resolution, preventing the classification head from making a confident crop assignment.

---

## 🚜 7. Edge-AI Deployment & Smart Sprayer Integration

### 7.1 Binary Actuation Mapping
In autonomous field robotics, multi-class predictions are mapped to physical actuator commands:
$$\text{Actuator Command} = \begin{cases} \text{ACTUATE SPRAY NOZZLE (WEED TARGET)} & \text{if } \hat{y} \in \{\text{Weeds}\} \\ \text{INHIBIT SPRAY NOZZLE (PROTECT CROP)} & \text{if } \hat{y} \in \{\text{Maize, Wheat, Sugar beet}\} \end{cases}$$

### 7.2 Latency vs. Tractor Speed Analysis
* **Operating Speed:** Standard agricultural tractors travel at $8 - 14\text{ km/h}$ ($2.2 - 3.9\text{ m/s}$).
* **Camera-to-Nozzle Offset:** With down-facing cameras mounted $0.8\text{ m}$ forward of the spray boom, the total latency budget is:
  $$\Delta t_{\text{max}} = \frac{0.8\text{ m}}{3.9\text{ m/s}} \approx 205\text{ ms}$$
* **Embedded Optimization:**
  - **FP16 / INT8 TensorRT Engine:** Compressing ResNet-34 onto an **NVIDIA Jetson Orin Nano** achieves **$< 12\text{ ms}$** per-frame latency, well within the $205\text{ ms}$ budget.
  - **Economic & Ecological Impact:** Spot-spraying saves **$72\% - 84\%$** of chemical herbicide volume, providing immediate financial payback for growers and preventing environmental degradation.

---

## 📦 8. Deliverables & Repository Structure (Rule 7, 8)
* **Notebook:** [`RegisterNo_PS09.ipynb`](file:///c:/Users/USER/Desktop/Activity/RegisterNo_PS09.ipynb) (Self-contained, fully documented, with all required outputs embedded).
* **Report:** [`RegisterNo_PS09_Report.md`](file:///c:/Users/USER/Desktop/Activity/RegisterNo_PS09_Report.md) (Comprehensive formal assignment report).
* **Git Versioning:** Initialized and committed locally, ready for push to GitHub.

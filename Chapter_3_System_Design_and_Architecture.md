# Chapter 3: System Design and Architecture

---

## 3.1 Overview of System Architecture

The **DrugShield AI** platform is a unified forensic drug detection system that integrates two distinct artificial intelligence pipelines—an image classification module and a natural language processing (NLP) text classification module—within a single cohesive web application. The system is engineered to assist law enforcement agencies and digital forensic investigators in identifying drug-related content from both visual and textual digital evidence.

The architecture follows a **modular, layered design** comprising four primary layers:

1. **Data Acquisition Layer** – Handles multi-modal input ingestion (images, text, CSV files, and screenshots).
2. **Feature Extraction and Preprocessing Layer** – Transforms raw inputs into machine-learning-ready feature representations.
3. **AI Detection Modules** – Houses the two core classification engines (VGG16-based image classifier and TF-IDF + Logistic Regression text classifier).
4. **Results Aggregation and Reporting Layer** – Consolidates predictions from both modules, computes risk scores, and provides exportable forensic reports.

The platform is delivered through a **Streamlit-based web application** (`unified_app.py`) that serves as the forensic user interface, orchestrating all four layers and presenting results in a dark-themed, glassmorphism-styled dashboard.

### High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     DrugShield AI Platform                       │
│                   (Streamlit Web Application)                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              DATA ACQUISITION LAYER                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │  Image   │  │  Direct  │  │   CSV    │  │Screenshot│  │  │
│  │  │  Upload  │  │  Text    │  │  Upload  │  │   OCR    │  │  │
│  │  │ /Folder  │  │  Input   │  │          │  │          │  │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │  │
│  └───────┼──────────────┼─────────────┼─────────────┼────────┘  │
│          │              │             │             │            │
│  ┌───────▼──────────────▼─────────────▼─────────────▼────────┐  │
│  │        FEATURE EXTRACTION & PREPROCESSING LAYER           │  │
│  │  ┌──────────────────┐    ┌──────────────────────────────┐ │  │
│  │  │  VGG16 Feature   │    │  TF-IDF Vectorization        │ │  │
│  │  │  Extraction      │    │  (max_features=200,          │ │  │
│  │  │  (224×224 RGB →   │    │   ngram_range=(1,2))         │ │  │
│  │  │   25,088-d vector)│    │  + Keyword Fallback          │ │  │
│  │  └────────┬─────────┘    └──────────────┬───────────────┘ │  │
│  └───────────┼─────────────────────────────┼─────────────────┘  │
│              │                             │                    │
│  ┌───────────▼─────────────────────────────▼─────────────────┐  │
│  │               AI DETECTION MODULES                        │  │
│  │  ┌──────────────────┐    ┌──────────────────────────────┐ │  │
│  │  │   Neural Network │    │  Logistic Regression         │ │  │
│  │  │   Classifier     │    │  Classifier                  │ │  │
│  │  │  (512→256→128→1) │    │  (C=0.4, liblinear solver)  │ │  │
│  │  │  Binary: Drug /  │    │  Binary: Suspicious /        │ │  │
│  │  │  Non-Drug        │    │  Normal                      │ │  │
│  │  └────────┬─────────┘    └──────────────┬───────────────┘ │  │
│  └───────────┼─────────────────────────────┼─────────────────┘  │
│              │                             │                    │
│  ┌───────────▼─────────────────────────────▼─────────────────┐  │
│  │         RESULTS AGGREGATION & REPORTING LAYER             │  │
│  │  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐ │  │
│  │  │ Confidence │  │   Risk     │  │  CSV Export /        │ │  │
│  │  │ Scoring    │  │ Assessment │  │  Combined Reports    │ │  │
│  │  └────────────┘  └────────────┘  └─────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              FORENSIC USER INTERFACE                      │  │
│  │   Home │ Image Analysis │ Text Analysis │ Dashboard       │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3.2 Architectural Design Principles

The system architecture is governed by the following design principles that ensure maintainability, scalability, and forensic reliability:

### 3.2.1 Modularity and Separation of Concerns

Each AI pipeline operates independently within its own Python module. The image classification logic resides in `image_predictor.py`, while text classification is encapsulated in `text_predictor.py`. This separation ensures that:

- Each module can be developed, tested, and updated independently.
- Failures in one pipeline do not cascade into the other.
- New detection modules (e.g., audio analysis or video frame extraction) can be integrated without modifying existing components.

### 3.2.2 Transfer Learning for Resource Efficiency

Rather than training a deep convolutional neural network from scratch—which would require millions of labelled images and weeks of GPU compute—the system employs **transfer learning** via VGG16 pre-trained on ImageNet. Only the classification head is custom-trained, dramatically reducing data requirements and training time while leveraging ImageNet's rich learned feature representations.

### 3.2.3 Lazy Loading and Singleton Pattern

Both AI modules load their respective models at **import time** (singleton pattern), ensuring that expensive model initialization occurs only once per application session. Subsequent predictions reuse the loaded model objects, achieving sub-second inference latency for individual samples.

### 3.2.4 Hybrid Detection Strategy

The text analysis module employs a **hybrid machine learning and rule-based approach**. The TF-IDF + Logistic Regression model provides probabilistic classification, while a curated keyword dictionary (`DRUG_KEYWORDS`) acts as a fallback safety net. This hybrid strategy minimises false negatives—messages containing explicit drug terminology are flagged even if the ML model assigns a low probability.

### 3.2.5 Session-Based State Management

The application uses Streamlit's `session_state` to maintain analysis histories across page navigations without requiring a persistent database. This design is appropriate for forensic workstation deployment where each session represents an independent investigation.

### 3.2.6 Defence in Depth (Authentication)

A login gate protects the platform, ensuring only authorised investigators can access the analysis tools. While the current implementation uses hardcoded credentials (`admin/admin`) for demonstration purposes, the architecture supports replacement with enterprise authentication mechanisms (LDAP, OAuth2).

---

## 3.3 Data Acquisition Layer

The Data Acquisition Layer is responsible for ingesting evidence from multiple input modalities. It is implemented across the `image_analysis_page()` and `text_analysis_page()` functions within `unified_app.py`.

### Supported Input Types

| Input Type | Module | Format | Method |
|---|---|---|---|
| Individual Images | Image Analysis | JPG, JPEG, PNG | File upload widget |
| Image Folders | Image Analysis | Directory path | Text input field |
| Direct Text | Text Analysis | Free text | Text area widget |
| CSV Files | Text Analysis | CSV with `text` column | File upload widget |
| Screenshots | Text Analysis | JPG, JPEG, PNG | File upload + OCR |

### Image Acquisition

The image module supports two ingestion methods:

1. **File Upload** – Users upload one or more image files through Streamlit's `file_uploader` widget. Files are temporarily saved to disk using Python's `tempfile.NamedTemporaryFile`, processed by the VGG16 pipeline, and then immediately deleted via `os.unlink()` to prevent evidence contamination.

2. **Folder Path** – Users provide the absolute path to a directory containing evidence images. The system scans the directory for files with `.jpg`, `.jpeg`, or `.png` extensions and processes them sequentially.

### Text Acquisition

The text module supports three ingestion methods:

1. **Direct Text Input** – A text area where investigators can type or paste suspicious messages. Pre-loaded demo examples assist in system familiarisation.

2. **CSV Batch Upload** – Users upload a CSV file containing a `text` column. The system automatically detects common column name variants (`text`, `Text`, `TEXT`, `message`, `Message`, `content`, `Content`) and handles malformed CSV structures where columns may be incorrectly combined.

3. **Screenshot OCR** – Users upload screenshots of conversations (e.g., from messaging apps). The system uses **Tesseract OCR** (via `pytesseract`) to extract text line-by-line, then feeds each extracted line through the NLP classifier.

---

## 3.4 Feature Extraction and Preprocessing Layer

### 3.4.1 Image Feature Extraction

The feature extraction pipeline transforms raw images into 25,088-dimensional feature vectors using the VGG16 convolutional neural network:

1. **Image Loading and Resizing** – Input images are resized to **224 × 224 pixels** (the native input resolution of VGG16) and converted to 3-channel RGB arrays.

2. **VGG16 Preprocessing** – Pixel values are preprocessed using Keras's `preprocess_input()` function, which performs channel-wise mean subtraction (subtracting the ImageNet mean RGB values: R=103.939, G=116.779, B=123.68) to normalise the input distribution.

3. **Convolutional Feature Extraction** – The preprocessed image tensor is passed through all 13 convolutional layers and 5 max-pooling layers of VGG16 (`include_top=False`), producing a feature map of shape **(1, 7, 7, 512)**.

4. **Feature Flattening** – The 3D feature map is flattened into a **25,088-dimensional vector** (7 × 7 × 512 = 25,088), forming a compact and discriminative representation of the image's visual content.

### 3.4.2 Text Feature Extraction

The text preprocessing and feature extraction pipeline operates as follows:

1. **Text Cleaning** – Raw text is converted to lowercase, and non-alphabetic characters are removed using regex substitution (`re.sub(r"[^a-zA-Z ]", "", text)`). Whitespace is normalised by splitting and rejoining tokens.

2. **TF-IDF Vectorization** – Cleaned text is transformed into a sparse feature matrix using **Term Frequency–Inverse Document Frequency (TF-IDF)** with the following parameters:
   - `max_features=200` – Limits the vocabulary to the 200 most informative terms.
   - `ngram_range=(1, 2)` – Captures both unigrams (single words) and bigrams (two-word phrases), enabling the model to recognise multi-word drug slang.
   - `stop_words="english"` – Filters out common English stop words that carry no discriminative signal.

3. **Keyword Feature Augmentation** – In parallel with TF-IDF, the system performs keyword matching against a curated dictionary of 22 drug-related terms (e.g., "cocaine", "heroin", "fentanyl", "ecstasy"). This acts as a feature-level signal that supplements the ML model's probabilistic output.

---

## 3.5 AI Detection Modules

### 3.5.1 NLP Text Classification Module

The NLP module (`text_predictor.py`) implements a **Logistic Regression classifier** that categorises text messages as either **"Suspicious"** (drug-related) or **"Normal"** (benign).

**Model Architecture:**

- **Algorithm:** Logistic Regression
- **Regularisation:** L1 penalty with C=0.4 (inverse regularisation strength)
- **Solver:** `liblinear` (optimised for small-to-medium datasets)
- **Max Iterations:** 1,000
- **Feature Input:** TF-IDF sparse matrix (200 features)

**Decision Logic:**

The module uses a hybrid decision mechanism that combines ML probability with keyword detection:

```
text_vector = vectorizer.transform([text])
probability = model.predict_proba(text_vector)[0][1]

if keyword_flag OR probability > 0.65:
    label = "Suspicious"
    confidence = max(probability, 0.7) × 100
else:
    label = "Normal"
    confidence = (1 - probability) × 100
```

This dual-gate approach ensures:
- Messages with explicit drug terminology are always flagged (keyword gate).
- Messages using coded language or slang are caught by the ML model (probability gate at 0.65 threshold).
- Confidence scores are bounded at a minimum of 70% for flagged messages, providing investigators with actionable certainty.

**Serialisation:**

The trained model and vectorizer are serialised using `joblib`:
- `nlp_model.pkl` – Logistic Regression classifier weights.
- `tfidf_vectorizer.pkl` – Fitted TF-IDF vocabulary and IDF weights.

---

### 3.5.2 VGG16 Image Classification Module

The image module (`image_predictor.py`) implements a **deep neural network classifier** that operates on VGG16-extracted features to classify images as either **"Drug"** or **"Non-Drug"**.

**Feature Extractor:**

- **Model:** VGG16 pre-trained on ImageNet (14.7M parameters)
- **Configuration:** `include_top=False` (removes the 1000-class fully connected layers)
- **Input:** 224 × 224 × 3 RGB images
- **Output:** 25,088-dimensional feature vector (flattened 7 × 7 × 512 feature map)

**Classifier Architecture:**

The custom classification head is a 4-layer fully connected neural network built using TensorFlow/Keras:

```
Input (25,088) → Dense(512, ReLU) → BatchNorm → Dropout(0.5)
              → Dense(256, ReLU) → BatchNorm → Dropout(0.4)
              → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
              → Dense(1, Sigmoid) → Output probability
```

| Layer | Output Shape | Parameters | Purpose |
|---|---|---|---|
| Input | (25088,) | — | VGG16 feature vector |
| Dense (512, ReLU) | (512,) | 12,845,568 | First hidden layer |
| BatchNormalization | (512,) | 2,048 | Stabilises training |
| Dropout (0.5) | (512,) | — | Prevents overfitting |
| Dense (256, ReLU) | (256,) | 131,328 | Second hidden layer |
| BatchNormalization | (256,) | 1,024 | Stabilises training |
| Dropout (0.4) | (256,) | — | Prevents overfitting |
| Dense (128, ReLU) | (128,) | 32,896 | Third hidden layer |
| BatchNormalization | (128,) | 512 | Stabilises training |
| Dropout (0.3) | (128,) | — | Prevents overfitting |
| Dense (1, Sigmoid) | (1,) | 129 | Binary output |

**Total trainable parameters:** ~13,013,505

**Decision Logic:**

```
probability = classifier.predict(features)[0][0]

if probability >= 0.48:
    label = "Drug"
    confidence = probability × 100
else:
    label = "Non-Drug"
    confidence = (1 - probability) × 100
```

The classification threshold is set at **0.48** (slightly below the default 0.50) to increase sensitivity toward drug detection, reflecting the forensic priority of minimising false negatives over false positives.

**Model Serialisation:**

The trained classifier weights are stored in HDF5 format (`model/image_classifier.h5`). At load time, the architecture is reconstructed programmatically via `_build_classifier()` and weights are loaded separately—a design choice that avoids Keras version compatibility issues when deserialising `.h5` configuration metadata.

---

## 3.6 Proposed Hybrid Socio-Technical Framework

The DrugShield AI platform operates within a **hybrid socio-technical framework** that recognises the critical role of human expertise in the forensic workflow. While the AI modules provide automated classification, the system is designed as a **decision-support tool**, not a fully autonomous decision-maker.

### Framework Components:

1. **Technical Layer (Automated AI Analysis)**
   - VGG16 image classification for visual evidence.
   - NLP text classification for linguistic evidence.
   - OCR-based text extraction from screenshots.
   - Confidence scoring and risk assessment.

2. **Human-in-the-Loop (Expert Review)**
   - Investigators review AI-generated classifications before taking action.
   - Confidence scores guide prioritisation—high-confidence flagged content receives immediate attention.
   - Dashboard provides holistic overview across both modalities.

3. **Organisational Layer (Process Integration)**
   - Session-based analysis supports isolated investigation workflows.
   - Exportable CSV reports integrate with existing case management systems.
   - Combined reports merge image and text findings for comprehensive evidence packages.

4. **Ethical Safeguards**
   - Authentication restricts access to authorised personnel.
   - No persistent storage of analysed content (session-only).
   - Human review required for all AI-generated assertions.

### Framework Diagram:

```
                    ┌─────────────────────────────┐
                    │    ORGANISATIONAL CONTEXT    │
                    │  (Law Enforcement / Forensics)│
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     HUMAN EXPERT LAYER      │
                    │   (Forensic Investigator)    │
                    │  - Reviews AI predictions    │
                    │  - Makes final decisions     │
                    │  - Generates legal reports   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      TECHNICAL AI LAYER     │
                    │   ┌────────┐  ┌──────────┐  │
                    │   │ VGG16  │  │ NLP/ML   │  │
                    │   │ Image  │  │ Text     │  │
                    │   │ Module │  │ Module   │  │
                    │   └────────┘  └──────────┘  │
                    └─────────────────────────────┘
```

This framework ensures that the system enhances—rather than replaces—the investigator's judgement, mitigating risks associated with AI bias or misclassification.

---

## 3.7 Results Aggregation and Reporting Layer

The Results Aggregation and Reporting Layer consolidates outputs from both AI modules and presents them through multiple views.

### 3.7.1 Individual Result Rendering

Each classification result is rendered as a styled card with:

- **Input identifier** – Filename (images) or truncated message text (text).
- **Classification label** – "Drug"/"Non-Drug" (images) or "Suspicious"/"Normal" (text).
- **Confidence score** – Percentage value with visual confidence bar.
- **Risk assessment** – "High" (for flagged content) or "Low" (for benign content).
- **Visual coding** – Red-themed styling for flagged content; green-themed for benign content.

### 3.7.2 Session Statistics

Running statistics are maintained throughout each session:

- Total items analysed (per module and combined).
- Count of flagged items (Drug images + Suspicious texts).
- Count of benign items.
- Threat percentage calculations.

### 3.7.3 Dashboard Aggregation

The Dashboard page (`dashboard_page()`) provides a **unified forensic overview** by combining statistics from both modules:

- **4-column stat cards** – Total Analysed, Threats Found, Images Processed, Texts Processed.
- **Side-by-side breakdowns** – Image Analysis and Text Analysis results with visual progress bars.
- **Proportional threat indicators** – Percentage of flagged content per module.

### 3.7.4 Report Export

The system supports three CSV export formats:

| Report Type | Contents | Filename |
|---|---|---|
| Image Report | Image filename, prediction, confidence, risk | `image_report.csv` |
| Text Report | Message excerpt, prediction, confidence | `text_report.csv` |
| Combined Report | Type, input, prediction, confidence, risk | `combined_report.csv` |

These reports conform to a structured tabular format suitable for integration with digital forensics case management platforms and judicial evidence submission requirements.

---

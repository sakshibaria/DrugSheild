# Chapter 4: Methodology

---

## 4.1 Research Methodology Overview

This research adopts a **Design Science Research (DSR)** methodology, which is appropriate for the development and evaluation of IT artefacts that solve identified organisational problems. The DSR framework involves iterative cycles of design, development, and evaluation, ensuring the resulting system is both technically sound and practically relevant.

The methodology encompasses the following phases:

1. **Problem Identification** – Recognising the growing use of social media platforms for illicit drug trafficking and the limitations of manual content moderation at scale.
2. **Solution Design** – Architecting a dual-modality AI system that analyses both visual and textual evidence.
3. **Dataset Acquisition and Preparation** – Sourcing, cleaning, and labelling domain-specific datasets for both image and text modalities.
4. **Model Development** – Training and optimising classification models using transfer learning (images) and traditional machine learning (text).
5. **System Implementation** – Integrating trained models into a unified web-based forensic platform.
6. **Evaluation** – Validating model performance through confusion matrices, classification reports, and cross-validation.

The research adopts a **quantitative, experimental approach** with the following characteristics:

- **Supervised learning paradigm** – Both models are trained on labelled datasets with binary classification targets.
- **Stratified train/test splitting** – Ensures class distribution is preserved across training and evaluation sets.
- **5-fold cross-validation** – Provides robust performance estimates for the NLP model.
- **Threshold tuning** – Decision boundaries are empirically adjusted to optimise sensitivity for forensic applications.

---

## 4.2 Dataset Description and Acquisition

### 4.2.1 Image Dataset: IDDIG

The image dataset is sourced from the **IDDIG (Illicit Drug Detection in Instagram and Google) dataset**, a collection of social media images related to drug promotion and distribution. The dataset comprises images scraped from online platforms, capturing the visual patterns used in drug advertising (e.g., images of pills, powders, cannabis, paraphernalia, and promotional graphics).

**Dataset Statistics:**

| Property | Value |
|---|---|
| Total raw images | ~2,735 (in `dataset/` directory) |
| Image file formats | JPG, JPEG, PNG |
| Image naming convention | Numeric identifiers (social media post IDs) |
| Labels | Binary: `drug` (1) and `non_drug` (0) |
| Labelled dataset structure | `labeled_dataset/drug/` and `labeled_dataset/non_drug/` |
| Average image file size | ~70–150 KB |

**Data Collection:**

The images were collected from social media platforms where drug-related posts were publicly visible. Each image is identified by a unique numeric ID corresponding to the original post. The dataset captures diverse visual representations of drug-related content, including:

- Photographs of illicit substances (powder, pills, crystals, plant material).
- Marketing-style images used by online drug dealers.
- Coded imagery and symbols associated with drug culture.
- Non-drug images (food, everyday objects, scenery) serving as negative examples.

### 4.2.2 Text Dataset: Synthetic Drug-Related Text Corpus

The text dataset (`Final_NLP_Dataset.csv`) is a curated corpus of text messages designed to represent the linguistic patterns found in drug trafficking communications.

**Dataset Statistics:**

| Property | Value |
|---|---|
| Dataset file | `Final_NLP_Dataset.csv` |
| File size | ~78 KB |
| Format | CSV with columns: `text`, `label` |
| Labels | Binary: `1` (drug-related) and `0` (non-drug) |
| Text samples | Short messages (1–2 sentences) |

**Corpus Composition:**

The dataset contains examples of:

- **Drug-related texts (label=1):** Messages referencing drug transactions, delivery arrangements, product descriptions (e.g., "premium powder available tonight contact privately"), supplier advertisements, and coded drug-related language.
- **Normal texts (label=0):** Everyday conversational messages, academic discussions, social invitations, and other benign communications (e.g., "professor uploaded lecture notes today").

The dataset is designed to train the classifier on the **linguistic fingerprints** of drug-related communications, including:
- Encrypted/coded language ("stuff", "premium", "guaranteed").
- Urgency indicators ("tonight", "fast delivery").
- Privacy signals ("privately", "contact me", "DM").
- Product descriptors ("powder", "tablets", "pills").

---

## 4.3 Data Preprocessing

### 4.3.1 Image Preprocessing Pipeline

The image preprocessing pipeline (`image_cleaning_vgg16.py` and `remove_duplicate_images.py`) transforms the raw dataset into a clean, standardised collection suitable for VGG16 feature extraction.

**Step 1: Image Cleaning (`image_cleaning_vgg16.py`)**

```python
# Pseudocode for image cleaning
for each image in dataset/:
    1. Validate file extension (.jpg, .jpeg, .png)
    2. Open image using PIL
    3. Convert to RGB mode (VGG16 requires 3-channel input)
    4. Resize to 224 × 224 pixels
    5. Save to cleaned_dataset/
    6. Skip corrupted images with error logging
```

This step ensures:
- **Colour space consistency** – All images are converted to RGB, handling any grayscale or RGBA images.
- **Dimensional uniformity** – All images are resized to the VGG16 input resolution of 224 × 224.
- **Data integrity** – Corrupted or unreadable files are identified and excluded.

**Step 2: Duplicate Removal (`remove_duplicate_images.py`)**

```python
# Perceptual hashing for near-duplicate detection
for each image in cleaned_dataset/:
    1. Compute perceptual hash (pHash) using imagehash library
    2. Compare against all existing hashes
    3. If hamming distance ≤ 5 → mark as duplicate → skip
    4. If unique → save to deduplicated_dataset/
```

Perceptual hashing (pHash) uses a **discrete cosine transform (DCT)** to create a compact fingerprint of each image's visual content. Unlike cryptographic hashing, pHash is robust to minor transformations (resizing, compression, colour adjustments), enabling detection of **near-duplicate** images that differ only in quality or format.

- **Hash threshold:** 5 (hamming distance). Lower values increase strictness.
- **Purpose:** Removes visually similar images that would inflate dataset size without adding new information, reducing overfitting risk.

**Step 3: Labelling**

After cleaning and deduplication, images are manually sorted into two subdirectories:
- `labeled_dataset/drug/` – Images containing drug-related content.
- `labeled_dataset/non_drug/` – Images containing non-drug content.

This directory-based labelling convention is directly consumed by the feature extraction pipeline.

### 4.3.2 Text Preprocessing Pipeline

The text preprocessing pipeline (`text_preprocessing.py`) normalises raw text messages before TF-IDF feature extraction.

**Processing Steps:**

1. **Dataset Loading and Structural Fix:**
   ```python
   # Handle CSV format where 'text' and 'label' appear as a single combined column
   if "text,label" in df.columns:
       df[['text','label']] = df['text,label'].str.rsplit(',', n=1, expand=True)
   df['label'] = df['label'].astype(int)
   ```

2. **Text Normalisation:**
   ```python
   def clean_text(text):
       text = text.lower()                         # Lowercase conversion
       text = re.sub(r"[^a-zA-Z ]", "", text)      # Remove non-alphabetic characters
       text = text.split()                          # Tokenise
       return " ".join(text)                        # Rejoin with single spaces
   ```

3. **Transformations Applied:**
   - **Lowercasing** – Eliminates case sensitivity (e.g., "Drug" and "drug" become identical).
   - **Special character removal** – Strips numbers, punctuation, and symbols that add noise without semantic value.
   - **Whitespace normalisation** – Removes extra spaces, tabs, and newlines.

---

## 4.4 Model Development and Training

### 4.4.1 NLP Model: Logistic Regression Classifier

**Feature Engineering (`tfidf_features.py`):**

The TF-IDF vectorizer is configured with the following parameters:

| Parameter | Value | Rationale |
|---|---|---|
| `max_features` | 200 | Limits vocabulary to most informative terms, preventing overfitting on rare words |
| `ngram_range` | (1, 2) | Captures single words and two-word phrases (e.g., "fast delivery") |
| `stop_words` | "english" | Removes common words (the, is, and) that carry no discriminative signal |

**Model Training (`model_training.py`):**

```python
# Model configuration
model = LogisticRegression(
    C=0.4,              # Regularisation strength (lower = stronger regularisation)
    solver="liblinear", # Optimised for small datasets with L1/L2 penalty
    max_iter=1000       # Maximum optimisation iterations
)
```

**Training Procedure:**

1. **Data Split:** 75% training / 25% testing (`test_size=0.25, random_state=42`).
2. **Vectorization:** TF-IDF transformation applied to the `text` column.
3. **Model Fitting:** Logistic Regression trained on TF-IDF feature matrix.
4. **Cross-Validation:** 5-fold cross-validation with accuracy scoring.
5. **Evaluation:** Confusion matrix computed on the held-out test set.
6. **Serialisation:** Model and vectorizer saved as `.pkl` files via `joblib`.

**Evaluation Metrics:**

- **Cross-validation accuracy** – Mean accuracy across 5 folds.
- **Confusion matrix** – True positives, true negatives, false positives, false negatives.

**Hybrid Inference Strategy:**

During deployment, the `text_predictor.py` module augments the ML model's output with keyword-based detection:

| Condition | Classification | Confidence |
|---|---|---|
| `keyword_flag = True` OR `probability > 0.65` | Suspicious | max(probability, 0.7) × 100 |
| Otherwise | Normal | (1 - probability) × 100 |

The 22-keyword dictionary includes terms such as: `weed`, `cocaine`, `heroin`, `pills`, `powder`, `fentanyl`, `mdma`, `ecstasy`, `lsd`, `marijuana`, and others.

### 4.4.2 Image Model: VGG16 Transfer Learning

**Feature Extraction Pipeline (`vgg16_feature_extraction_with_labels.py`):**

1. Load VGG16 pre-trained on ImageNet (`weights="imagenet"`, `include_top=False`).
2. Iterate over `labeled_dataset/drug/` and `labeled_dataset/non_drug/`.
3. For each image:
   - Load and resize to 224 × 224.
   - Apply VGG16 preprocessing (ImageNet mean subtraction).
   - Extract features via `model.predict()` → shape (1, 7, 7, 512).
   - Flatten to 25,088-dimensional vector.
   - Assign label: `drug=1`, `non_drug=0`.
4. Save feature matrix as `X_features.npy` and labels as `y_labels.npy`.

**Classifier Training (`train_and_evaluate.py`):**

```python
# Training configuration
Optimizer:       Adam (learning_rate=1e-4)
Loss Function:   Binary Cross-Entropy
Epochs:          60 (with early stopping)
Batch Size:      16
Validation Split: 20% of training data
```

**Training Strategy:**

| Strategy | Implementation | Purpose |
|---|---|---|
| **Feature Standardisation** | `StandardScaler` applied to VGG16 features | Normalises the 25,088 features to zero mean and unit variance |
| **Stratified Splitting** | `stratify=y` in `train_test_split` | Maintains drug/non-drug class ratio in both train and test sets |
| **Class Weighting** | `compute_class_weight("balanced")` | Addresses class imbalance by penalising misclassification of the minority class more heavily |
| **Early Stopping** | `patience=8`, `restore_best_weights=True` | Monitors validation loss and stops training when no improvement is seen for 8 epochs; restores the best model |
| **Learning Rate Scheduling** | `ReduceLROnPlateau(factor=0.5, patience=4)` | Halves the learning rate when validation loss plateaus for 4 epochs |
| **Batch Normalisation** | After each Dense layer | Normalises activations, reduces internal covariate shift, and accelerates convergence |
| **Dropout Regularisation** | 0.5 → 0.4 → 0.3 (decreasing) | Progressively decreasing dropout rates prevent overfitting while allowing deeper layers to specialise |

**Evaluation:**

- **Classification threshold:** 0.48 (tuned below 0.50 to increase drug detection sensitivity).
- **Evaluation metrics:** Confusion matrix and classification report with precision, recall, and F1-score for both classes.
- **Visualisation:** Confusion matrix heatmap using seaborn.

**Model Saving:**

The trained model is saved as `model/image_classifier.h5` in HDF5 format, containing both architecture and learned weights.

---

## 4.5 Ethical Framework and Privacy Considerations

### 4.5.1 Data Privacy

- **No persistent storage:** The application does not store analysed images or text beyond the active session. Once the browser is closed or the session is cleared, all data is purged from memory.
- **Temporary file handling:** Uploaded images are processed through temporary files (`tempfile.NamedTemporaryFile`) that are immediately deleted after analysis via `os.unlink()`.
- **Local processing:** All AI inference occurs on the local machine. No data is transmitted to external servers or cloud APIs.

### 4.5.2 Algorithmic Transparency

- **Confidence scores:** Every prediction is accompanied by a confidence percentage, enabling investigators to assess the reliability of each classification.
- **Risk levels:** Classifications are mapped to "High" and "Low" risk categories, providing clear guidance for prioritisation.
- **No black-box decisions:** The system functions as a decision-support tool; final determination of whether content is drug-related remains with the human investigator.

### 4.5.3 Bias Mitigation

- **Class balancing:** The image classifier uses `compute_class_weight("balanced")` to compensate for potential class imbalance in the training data.
- **Keyword fallback:** The text classifier's keyword dictionary provides a model-independent detection channel, reducing dependence on ML model accuracy alone.
- **Threshold calibration:** Classification thresholds (0.48 for images, 0.65 for text ML probability) are empirically tuned to prioritise recall (minimising false negatives) over precision, aligning with the forensic imperative of not missing potential evidence.

### 4.5.4 Access Control

- **Authentication gate:** The login page restricts access to authorised users, preventing unauthorised access to forensic analysis tools.
- **Session isolation:** Each user session maintains an independent analysis history, preventing cross-contamination between investigations.

### 4.5.5 Responsible AI Use

The system is designed for use by trained forensic investigators operating within legal frameworks. It does not make autonomous enforcement decisions. All AI-generated classifications must be reviewed and validated by qualified human analysts before being used as evidence in legal proceedings.

---

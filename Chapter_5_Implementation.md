# Chapter 5: Implementation

---

## 5.1 Development Environment

The DrugShield AI platform was developed and tested on the following environment:

| Component | Specification |
|---|---|
| **Operating System** | Microsoft Windows 10/11 |
| **Programming Language** | Python 3.x |
| **IDE / Editor** | Visual Studio Code |
| **Virtual Environment** | Python `venv` (local virtual environment in project directory) |
| **Package Manager** | pip |
| **Runtime** | Streamlit server (development mode via `streamlit run unified_app.py`) |
| **Hardware** | Standard desktop/laptop with CPU-based inference (GPU optional for training) |

The project follows a **flat module structure** where all primary scripts reside in the root project directory. Supporting models and datasets are organised within subdirectories:

```
project_folder/
├── unified_app.py                        # Main Streamlit application (1,210 lines)
├── image_predictor.py                    # VGG16 image inference module (91 lines)
├── text_predictor.py                     # NLP text inference module (51 lines)
├── vgg16_inference.py                    # Standalone VGG16 feature extraction
├── vgg16_feature_extraction_with_labels.py  # Labelled feature extraction
├── train_and_evaluate.py                 # Image classifier training script
├── image_cleaning_vgg16.py               # Image preprocessing utility
├── remove_duplicate_images.py            # Perceptual hash deduplication
├── activation_analysis.py                # Neural activation visualisation
├── requirements.txt                      # Python dependency list
│
├── model/
│   └── image_classifier.h5              # Trained image classifier (~156 MB)
│
├── NLP_Text_Model/
│   └── NLP Text work updated/
│       ├── nlp_model.pkl                # Trained Logistic Regression model
│       ├── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
│       ├── Final_NLP_Dataset.csv        # Text training dataset
│       ├── model_training.py            # NLP model training script
│       ├── text_preprocessing.py        # Text cleaning pipeline
│       ├── tfidf_features.py            # TF-IDF feature extraction
│       ├── save_model.py                # Model serialisation script
│       └── app.py                       # Standalone NLP demo app
│
├── dataset/                             # Raw image dataset (~2,735 images)
├── cleaned_dataset/                     # Resized and RGB-converted images
├── deduplicated_dataset/                # Images after duplicate removal
├── labeled_dataset/
│   ├── drug/                            # Labelled drug images
│   └── non_drug/                        # Labelled non-drug images
│
├── X_features.npy                       # Extracted VGG16 features (~21 MB)
├── y_labels.npy                         # Corresponding labels
├── vgg16_features.npy                   # Full dataset VGG16 features (~96 MB)
└── image_names.npy                      # Image filename index
```

---

## 5.2 Tools and Technologies

The following tools and technologies were used in the implementation of DrugShield AI:

### 5.2.1 Core Framework and Libraries

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.x | Primary programming language |
| **Streamlit** | Latest | Web application framework for the forensic UI |
| **TensorFlow / Keras** | Latest | Deep learning framework for VGG16 and neural classifier |
| **scikit-learn** | Latest | Machine learning library for Logistic Regression and TF-IDF |
| **NumPy** | Latest | Numerical computation and array operations |
| **Pandas** | Latest | Data manipulation and CSV handling |
| **Pillow (PIL)** | Latest | Image loading, processing, and format conversion |
| **joblib** | Latest | Model serialisation (pickle-based, optimised for NumPy arrays) |
| **pytesseract** | Latest | OCR engine for screenshot text extraction |

### 5.2.2 Data Processing Tools

| Tool | Purpose |
|---|---|
| **imagehash** | Perceptual hashing for near-duplicate image detection |
| **matplotlib** | Training visualisation (confusion matrix heatmaps, training curves) |
| **seaborn** | Statistical visualisation for model evaluation plots |

### 5.2.3 Pre-Trained Models

| Model | Source | Parameters | Application |
|---|---|---|---|
| **VGG16** | ImageNet (Keras Applications) | 14.7M | Visual feature extraction |
| **Logistic Regression** | scikit-learn | Trained on custom corpus | Text classification |

### 5.2.4 Dependency Specification

All runtime dependencies are declared in `requirements.txt`:

```
streamlit
tensorflow
numpy
pandas
pillow
joblib
scikit-learn
pytesseract
```

Installation: `pip install -r requirements.txt`

---

## 5.3 NLP Module Implementation

The NLP module is implemented across two key files: the training pipeline (`NLP_Text_Model/NLP Text work updated/save_model.py`) and the inference module (`text_predictor.py`).

### 5.3.1 Model Training Pipeline

**Step 1: Dataset Loading and Parsing**

```python
df = pd.read_csv("Final_NLP_Dataset.csv")

# Handle edge-case CSV formatting
if "text,label" in df.columns:
    df[['text','label']] = df['text,label'].str.rsplit(',', n=1, expand=True)
    df = df.drop(columns=['text,label'])

df['label'] = df['label'].astype(int)
```

The CSV parser includes a structural fix for datasets where the `text` and `label` columns are incorrectly combined into a single column (`text,label`). The `rsplit(',', n=1)` ensures the last comma is used as the delimiter, correctly handling messages that contain commas.

**Step 2: TF-IDF Vectorization**

```python
vectorizer = TfidfVectorizer(
    max_features=200,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]
```

The fitted vectorizer learns a vocabulary of the 200 most discriminative terms (including bigrams) from the corpus. Each document is transformed into a sparse vector where each dimension represents the TF-IDF weight of a vocabulary term in that document.

**Step 3: Model Training and Serialisation**

```python
model = LogisticRegression(
    C=0.4,
    solver="liblinear",
    max_iter=1000
)

model.fit(X, y)

joblib.dump(model, "nlp_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
```

The model is trained on the full dataset (not split) for the deployed version, maximising learning from all available examples. The `C=0.4` regularisation parameter applies moderate regularisation, preventing overfitting on the relatively small training corpus.

### 5.3.2 Inference Implementation (`text_predictor.py`)

```python
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLP_DIR = os.path.join(BASE_DIR, "NLP_Text_Model", "NLP Text work updated")

# Singleton: models loaded once at import time
model = joblib.load(os.path.join(NLP_DIR, "nlp_model.pkl"))
vectorizer = joblib.load(os.path.join(NLP_DIR, "tfidf_vectorizer.pkl"))

DRUG_KEYWORDS = [
    "weed", "cocaine", "heroin", "drug", "drugs",
    "pills", "powder", "tablet", "supplier",
    "delivery", "deliver", "stuff", "meth",
    "crack", "fentanyl", "opium", "marijuana",
    "hash", "mdma", "ecstasy", "lsd",
]


def predict_text(text):
    if not text or not text.strip():
        return "Normal", 0.0

    text_lower = text.lower().strip()

    # Gate 1: Keyword detection
    keyword_flag = any(kw in text_lower for kw in DRUG_KEYWORDS)

    # Gate 2: ML probability
    text_vector = vectorizer.transform([text])
    probability = model.predict_proba(text_vector)[0][1]

    # Hybrid decision
    if keyword_flag or probability > 0.65:
        label = "Suspicious"
        confidence = max(probability, 0.7) * 100
    else:
        label = "Normal"
        confidence = (1 - probability) * 100

    return label, confidence
```

**Key Implementation Details:**

1. **Path Resolution** – `os.path.dirname(os.path.abspath(__file__))` ensures model files are located relative to the script's location, enabling the module to work regardless of the working directory.

2. **Singleton Loading** – Models are loaded at module import time (`model = joblib.load(...)` at the module level), ensuring the ~2.5 KB model and ~9.4 KB vectorizer are deserialised only once per application lifecycle.

3. **Empty Input Handling** – The function returns `("Normal", 0.0)` for empty or whitespace-only inputs, preventing vectorization errors.

4. **Confidence Floor** – The `max(probability, 0.7)` expression ensures that keyword-triggered classifications always show at least 70% confidence, preventing confusing low-confidence alerts.

---

## 5.4 VGG16 Image Module Implementation

The image module is implemented across the training pipeline (`vgg16_feature_extraction_with_labels.py` and `train_and_evaluate.py`) and the inference module (`image_predictor.py`).

### 5.4.1 Feature Extraction Pipeline

```python
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = (224, 224)
DATASET_DIR = "labeled_dataset"

model = VGG16(weights="imagenet", include_top=False)

label_map = {"drug": 1, "non_drug": 0}

X, y = [], []

for label_name in label_map:
    folder_path = os.path.join(DATASET_DIR, label_name)

    for img_name in os.listdir(folder_path):
        img = load_img(img_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = model.predict(img_array, verbose=0)
        X.append(features.flatten())       # (1,7,7,512) → (25088,)
        y.append(label_map[label_name])

np.save("X_features.npy", np.array(X))    # Shape: (N, 25088)
np.save("y_labels.npy", np.array(y))       # Shape: (N,)
```

**Output:**
- `X_features.npy` – Feature matrix of shape (N, 25088) containing VGG16 feature vectors for all labelled images (~21 MB).
- `y_labels.npy` – Label vector of shape (N,) with binary values.

### 5.4.2 Classifier Training Pipeline

```python
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load extracted features
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

# Standardise features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stratified split: 75% train / 25% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Compute balanced class weights
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Build classifier
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=4
)

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=60,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr]
)

# Evaluate with adjusted threshold
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.48).astype(int).ravel()

# Save
model.save("model/image_classifier.h5")
```

### 5.4.3 Inference Implementation (`image_predictor.py`)

```python
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image

# --- Model Reconstruction ---
def _build_classifier():
    """Reconstruct architecture and load saved weights."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(25088,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    h5_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model", "image_classifier.h5"
    )

    model.build((None, 25088))          # Initialise weight tensors
    model.load_weights(h5_path)         # Load trained weights only

    return model

# Singleton loading
vgg16 = VGG16(weights="imagenet", include_top=False)
classifier = _build_classifier()

IMAGE_SIZE = (224, 224)


def predict_image(img_path):
    """Predict Drug / Non-Drug for a single image."""
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = vgg16.predict(img_array, verbose=0)
    features = features.flatten().reshape(1, -1)

    prob = classifier.predict(features, verbose=0)[0][0]

    if prob >= 0.48:
        label = "Drug"
        confidence = prob * 100
    else:
        label = "Non-Drug"
        confidence = (1 - prob) * 100

    return label, confidence
```

**Key Implementation Details:**

1. **Architecture Reconstruction** – Instead of loading the entire model via `tf.keras.models.load_model()`, the architecture is manually reconstructed in `_build_classifier()` and only the weights are loaded via `model.load_weights()`. This approach avoids Keras version incompatibilities that can arise when `.h5` configuration metadata references deprecated or renamed classes.

2. **Build Before Load** – `model.build((None, 25088))` is called before `load_weights()` to initialise the layer weight tensors with the correct shapes. Without this step, `load_weights()` would fail because the model has no allocated weight matrices to populate.

3. **Two-Stage Inference** – Each prediction involves two sequential forward passes:
   - **Stage 1:** VGG16 processes the image (224×224×3) → feature vector (25,088).
   - **Stage 2:** Custom classifier processes the feature vector → probability (scalar).

4. **Error Handling** – The function validates file existence and catches image loading errors (corrupted or incompatible files), raising descriptive exceptions rather than failing silently.

---

## 5.5 Forensic User Interface Implementation

The forensic user interface is implemented in `unified_app.py` (1,210 lines) using the **Streamlit** framework. The UI is designed with a dark-themed, glassmorphism-inspired aesthetic that conveys professionalism and forensic seriousness.

### 5.5.1 Application Architecture

The application follows a **single-page application (SPA)** pattern with client-side routing via Streamlit's `session_state`:

```python
# Session state initialisation
st.session_state.logged_in = False    # Authentication state
st.session_state.page = "Home"        # Current page
st.session_state.image_history = []   # Image analysis results
st.session_state.text_history = []    # Text analysis results
st.session_state.input_text = ""      # Text input buffer

# Page routing
if st.session_state.logged_in:
    sidebar_nav()
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Image Analysis":
        image_analysis_page()
    elif st.session_state.page == "Text Analysis":
        text_analysis_page()
    elif st.session_state.page == "Dashboard":
        dashboard_page()
else:
    login()
```

### 5.5.2 UI Theme and Styling

The interface uses approximately **300 lines of custom CSS** injected via `st.markdown()` with `unsafe_allow_html=True`. Key design elements include:

| Element | Implementation |
|---|---|
| **Dark theme** | Background gradient: `#0a0a0f → #0d0d1a → #0f0a1a` |
| **Typography** | Google Fonts "Inter" (weights 300–800) |
| **Glass cards** | `backdrop-filter: blur(20px)` with semi-transparent backgrounds |
| **Accent colour** | Purple (#8b5cf6) with gradient variations |
| **Animations** | CSS `fadeIn` keyframes for page transitions |
| **Hover effects** | `translateY(-2px)` lift + box-shadow glow |
| **Colour coding** | Red (#ef4444) for threats, Green (#22c55e) for safe |
| **Confidence bars** | CSS gradient-filled progress bars |
| **Badges** | Gradient-styled pill labels ("Drug", "Safe", "Suspicious", "Normal") |

### 5.5.3 Page Implementations

**Login Page (`login()`):**
- Centred glassmorphism card with brand logo and title.
- Username and password input fields.
- Credential validation (default: `admin`/`admin`).
- Session state update and automatic page redirect on success.

**Home Page (`home_page()`):**
- Hero section with animated title and gradient text.
- Two feature cards describing Image Analysis and Text Analysis capabilities.
- "How It Works" section with 4-step process flow (Upload → Process → Analyse → Report).
- Technology tags (VGG16, Neural Network, TF-IDF, Logistic Regression).

**Image Analysis Page (`image_analysis_page()`):**
- Summary statistics bar (Total Analysed, Drug Detected, Non-Drug).
- Two-tab interface: File Upload and Folder Path.
- Progress bar during batch processing.
- Scrollable results list with colour-coded prediction cards.
- CSV download button for export.
- Clear history button.

**Text Analysis Page (`text_analysis_page()`):**
- Summary statistics bar (Total Analysed, Suspicious, Normal).
- Three-tab interface: Direct Text, CSV Upload, Screenshot OCR.
- Demo examples with one-click loading.
- Inline result rendering with confidence bars.
- CSV batch processing with progress indication.
- Screenshot OCR with line-by-line text extraction and analysis.

**Dashboard Page (`dashboard_page()`):**
- Four top-level metric cards (Total, Threats, Images, Texts).
- Side-by-side module breakdowns with proportional threat bars.
- Three-column export section (Image Report, Text Report, Combined Report).
- Clear All History function.

### 5.5.4 Sidebar Navigation

The sidebar provides:
- Brand identity (logo, title, subtitle).
- Navigation buttons for all four pages with emoji icons.
- Live session statistics (images and texts analysed).
- Logout button with session state reset.

### 5.5.5 Running the Application

The application is launched via the Streamlit CLI:

```bash
streamlit run unified_app.py
```

This starts a local development server (default: `http://localhost:8501`) that serves the forensic interface. The server handles:
- WebSocket-based bidirectional communication between frontend and backend.
- Automatic re-execution of the Python script on user interaction.
- File upload handling via Streamlit's built-in upload manager.
- Session state persistence across script re-runs within the same browser session.

---

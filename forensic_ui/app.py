import os
from flask import Flask, render_template, request
from image_model import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html", results=None, summary=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    results = []
    total = 0
    drug_count = 0
    non_drug_count = 0
    high_risk_count = 0

    if "files" not in request.files:
        return render_template("index.html", results=None, summary=None)

    files = request.files.getlist("files")

    for file in files:
        if file.filename == "":
            continue

        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        try:
            label, confidence = predict_image(filepath)
        except Exception as e:
            print("Prediction Error:", e)
            continue

        risk = "High" if label == "Drug" else "Low"

        total += 1
        if label == "Drug":
            drug_count += 1
        else:
            non_drug_count += 1

        if risk == "High":
            high_risk_count += 1

        results.append({
            "image": file.filename,
            "prediction": label,
            "confidence": f"{confidence:.2f}%",
            "risk": risk
        })

    summary = {
        "total": total,
        "drug": drug_count,
        "non_drug": non_drug_count,
        "high_risk": high_risk_count
    }

    return render_template("index.html", results=results, summary=summary)


if __name__ == "__main__":
    app.run(debug=True)

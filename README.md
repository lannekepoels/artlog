# 🖼️ Sinopia — Web App

A local web application that automatically extracts metadata from scanned art catalogue pages using Google Cloud Vision and Gemini AI, and lets you review and correct the results through a built-in annotation interface.

---

## 🧰 What You Need Before Starting

- **Python 3** installed on your computer ([download here](https://www.python.org/downloads/) if needed)
- A **Google Cloud Vision API key**
- A **Gemini API key**
- A **ZIP file** containing your catalogue scan images (`.jpg`, `.png`, `.tif`, etc.)

---

## 🚀 Getting Started

These steps only need to be done once.

### Step 1 — Download the project

Click the green **Code** button on this GitHub page and choose **Download ZIP**. Unzip it somewhere on your computer.

### Step 2 — Open a terminal in the project folder

On Mac: open the **Terminal** app, then drag the project folder into the terminal window and press Enter. You should see the folder path appear in the prompt.

### Step 3 — Install the required Python packages

Copy and paste this command into the terminal and press Enter:

```bash
pip install -r requirements.txt
```

This downloads all the Python libraries the app needs. You only need to do this once.

> **Having trouble?** Try one of these alternatives:
>
> ```bash
> pip install -r requirements.txt --break-system-packages
> ```
>
> Or, if you want to keep things isolated with a virtual environment:
>
> ```bash
> python3 -m venv venv && source venv/bin/activate
> pip install -r requirements.txt
> ```

### Step 4 — Start the app

```bash
python app.py
```

You should see a message saying the server is running.

### Step 5 — Open in your browser

Open any browser (Chrome, Safari, Firefox) and go to:

```text
http://localhost:5001
```

The app will open. You can now use it like any normal website — it's just running locally on your machine.

---

## ⚙️ Using the App

### Upload page

1. Enter your **Google Cloud Vision API key** and **Gemini API key** in the fields provided
2. Upload your **ZIP file** of catalogue scans
3. Click **Run Extraction Pipeline** — the app will start processing in the background
4. A live log will show progress as each page is handled
5. When finished, click **Open Annotation Interface**

> Each time you run the pipeline, it gets its own unique job ID so you can run multiple batches without them interfering with each other.

### Annotation interface

Once the pipeline finishes, you land in the annotation interface where you can review and correct every extracted record.

The screen is split into three panels:

- **Left panel** — list of all records with thumbnails; use the search box to filter by artist or title
- **Center panel** — image viewer; toggle between the cropped artwork and the full page scan
- **Right panel** — all extracted metadata fields, fully editable by clicking directly into them

#### Actions per record

| Action | How |
| ------ | --- |
| Edit any field | Click on it and type |
| Mark as validated | Click **✓ Mark Validated** or press `V` on your keyboard |
| Flag for review | Click **⚑ Flag** |
| Navigate records | Click Prev / Next, or use the `←` `→` arrow keys |
| Filter records | Type in the search box on the left |

#### Saving and exporting

| Button | What it does |
| ------ | ------------ |
| **Export Images** | Downloads all your cropped images (in a ZIP file) |
| **Export CSV** | Downloads all records as a CSV file you can open in Excel |

---

## 🔒 Privacy & Security

- **Your API keys are never stored** — they are only held in memory while the pipeline is running, and forgotten as soon as it finishes
- Everything runs locally on your own machine — no data is sent anywhere except to the Google Cloud Vision and Gemini APIs for processing

---

## 📁 File Structure

For reference, here is what the project folder contains:

```text
├── app.py                  ← The main application (Flask server + pipeline)
├── requirements.txt        ← List of Python packages needed to run the app
├── README.md               ← This file
├── templates/
│   ├── index.html          ← Upload & progress page
│   └── annotate.html       ← Annotation interface
├── uploads/                ← Where uploaded ZIPs and images are stored (auto-created)
├── results/                ← Where exported CSVs are saved (auto-created)
├── data/                   ← Local catalogue data (not included in this repo)
└── raw_scripts/            ← Earlier standalone pipeline scripts, kept for reference
```

---

## 🛠️ Troubleshooting

### The app won't start — "No module named 'cv2'"

Run this in the terminal:

```bash
pip install opencv-python
```

### The pipeline gets stuck or shows "429" errors

This means the Gemini API is receiving too many requests too quickly. The app handles this automatically by waiting and retrying — just leave it running and it will recover on its own. For very large batches, it may take a while.

### "No supported image files found in ZIP"

Make sure your ZIP file contains images (`.jpg`, `.png`, `.tif`) either at the top level or inside one folder. If you created the ZIP on a Mac, the hidden `__MACOSX` folder inside it is automatically ignored.

### The page at localhost:5001 won't load

Make sure you ran `python app.py` first and that the terminal is still open — closing the terminal stops the app.

---

Made by: Max, Tamara, Yiliu, Lanneke and Nederlands Instituut voor Kunstgeschiedenis (RKD).




## Installation

- [macOS](#installation--macos)
- [Windows](#installation--windows)

## Installation — macOS

1) Check that Python is installed

Open Terminal and run:

python3 --version

You should see something like:

Python 3.11.x

If you get an error, install Python from the official Python website.

2) Download and unzip the project

Download the repository ZIP from GitHub and unzip it somewhere on your computer.

3) Open Terminal in the project folder

For example, if the folder is in Downloads:

cd ~/Downloads/interface

Adjust the path if your folder is elsewhere.

4) Install the dependencies

Run:

If that fails, try:

pip install -r requirements.txt --break-system-packages

If you prefer a cleaner setup, use a virtual environment:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

5) Start the app

Run:

python app.py

You should see the local server start on port 5001.

6) Open it in your browser

Open Chrome, Safari, or Firefox and go to:

http://localhost:5001


## Installation — Windows
...

1) Install Python

Download Python from the official Python website.

During installation, make sure to:
	•	check Add Python.exe to PATH
	•	check Use admin privileges when installing
	•	click Install Now
	•	click Disable path length limit if Windows shows that option

Then open Command Prompt and run:

python --version

You should see something like:

Python 3.x.x

2) Unzip the project

Right-click the downloaded ZIP and choose Extract All.

3) Open Command Prompt in the project folder

Move into the extracted folder. Example:

cd C:\Users\YourUsername\Downloads\interface

Then confirm you are in the right place:

dir

You should see requirements.txt.

4) Install dependencies

Run:

pip install -r requirements.txt

If you hit a permissions error, try:

pip install -r requirements.txt --break-system-packages

5) Start the app

Run:

python app.py

You should see the local server start on port 5001.

6) Open it in your browser

Open Chrome, Edge, or Firefox and go to:

http://localhost:5001


How to use the app

1) Upload page

On the start page:
	1.	Paste in your Google Cloud Vision API key
	2.	Paste in your Gemini API key
	3.	Upload a ZIP file of catalogue scan images
	4.	Click Run Extraction Pipeline

The app will process the images in the background and show a live log as it runs.

Each pipeline run gets its own unique job ID so batches do not interfere with each other.

2) Annotation interface

When processing is complete, open the annotation interface to review and correct extracted records.

Layout

Left panel
	•	list of all records with thumbnails
	•	search box to filter by artist or title

Center panel
	•	image viewer
	•	switch between cropped artwork and full page image

Right panel
	•	editable metadata fields
	•	click directly into a field to correct it



Saving and exporting

Save edits

When records are saved, the app updates the current job and also writes an annotated CSV file into the results/ folder.

Export CSV

Exports all non-deleted records as a CSV file for Excel, pandas, or further analysis.

Export Images

Exports the cropped artwork images as a ZIP file.

If a crop is missing, the app may fall back to the original page image.

⸻

Extracted metadata fields

The pipeline is designed to extract fields such as:
	•	Artist
	•	Title
	•	Date
	•	Search_Margin_Begin
	•	Search_Margin_End
	•	Genre
	•	Object_Name
	•	Medium
	•	Shape
	•	Height
	•	Width
	•	Unit
	•	Signature_Inscription
	•	Signature_Location
	•	Provenance
	•	Artwork_Number
	•	Image_Number
	•	FullEntryText

It also adds internal review fields such as validation status, notes, record ID, original image URL, and crop URL.

⸻

Privacy and security
	•	The app runs locally on your own computer
	•	Your API keys are entered at runtime and should not be committed to the repository
	•	Data is only sent externally when calling:
	•	Google Cloud Vision
	•	Gemini

Do not paste API keys into documentation, screenshots, or shared files.

⸻

Troubleshooting

No module named 'cv2'

Install OpenCV:

pip install opencv-python

Missing dependencies such as PIL

Reinstall dependencies:

pip install -r requirements.txt

If needed, install key packages directly:

pip install pillow pandas requests google-cloud-vision opencv-python

429 errors or slow processing

This means Gemini is receiving too many requests too quickly.

The app is designed to wait and retry automatically. For large batches, leave it running.

No supported image files found in ZIP

Make sure your ZIP contains supported image files such as .jpg, .png, .tif, .tiff, or .bmp.

If the ZIP was created on a Mac, hidden folders such as __MACOSX are ignored automatically.

localhost:5001 does not open

Make sure:
	•	you ran python app.py
	•	the terminal or command prompt window is still open
	•	the app did not stop because of an earlier error

Closing the terminal stops the server.

Timeout during upload or processing

Very large images can slow down processing significantly.

If needed:
	•	resize oversized images before uploading
	•	test with a smaller batch first
	•	retry after reducing image dimensions

Wrong Google credential type

Use an API key, not an OAuth client secret JSON file.

Protobuf / Google Cloud Vision compatibility error

If you see protobuf-related errors, try:

pip install "protobuf>=4.21,<6"

Python version issues

Older Python versions may fail on newer syntax or dependency combinations.

If possible, use Python 3.11+.

⸻

Evaluation workflow

The annotation interface can also be used for evaluation and scoring.

Recommended workflow:
	1.	Review extracted records in the annotation interface
	2.	Overwrite incorrect values directly in the metadata fields
	3.	Export the corrected CSV
	4.	Calculate scores later with pandas or another analysis tool

Notes for evaluation
	•	Use the Image_Number field to evaluate crop quality
	•	Skip blank pages or pages with no matching artwork/metadata set
	•	Use annotator notes to record:
	•	hallucinations
	•	OCR errors
	•	ambiguous cases
	•	missing image matches
	•	possible rate-limit problems

Suggested weighting

Tier 1
	•	Artist
	•	Title
	•	Date
	•	Object_Name
	•	Medium

Tier 2
	•	Genre
	•	Shape
	•	Height
	•	Width
	•	Unit
	•	Search_Margin_Begin / Search_Margin_End
	•	Signature fields

Tier 3
	•	Artwork_Number
	•	Image_Number
	•	Provenance
	•	FullEntryText

⸻

Notes for developers
	•	The Flask app runs on 0.0.0.0:5001
	•	Upload size limit is set to 500 MB
	•	use_reloader=False is intentional so Flask does not restart when uploaded ZIPs are written, which would otherwise wipe the in-memory job store
	•	Jobs are stored in memory during runtime
	•	uploads/ and results/ are created automatically if missing


# Artlog — Local Metadata Extraction Web App for Art Catalogues

Artlog is a local web application that extracts metadata from scanned art catalogue pages using **Google Cloud Vision** and **Gemini**, then lets you review and correct the results in a built-in annotation interface.

The app runs on your own computer and opens in your browser at:

**http://localhost:5001**

---

## What the app does

- Upload a ZIP of catalogue scan images
- Detect and extract artwork metadata from each page
- Create one or more records per page when multiple entries are found
- Show progress in a live processing log
- Let you review, edit, validate, flag, and manually recrop records
- Export corrected metadata as CSV
- Export cropped artwork images as a ZIP

The app also:

- assigns each run a unique job ID
- ignores hidden folders such as `__MACOSX`
- creates `BLANK` records for blank pages
- retries automatically when Gemini hits rate limits

---

## Before you start

You will need:

- **Python 3**
- a **Google Cloud Vision API key**
- a **Gemini API key**
- a **ZIP file** containing catalogue scan images

### Supported image formats

- `.jpg`
- `.jpeg`
- `.png`
- `.tif`
- `.tiff`
- `.bmp`

### Recommended Python version

Python **3.11 or newer** is recommended.

---

## Project structure

```text
artlog/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   ├── index.html
│   └── annotate.html
├── uploads/         # created automatically
├── results/         # created automatically
├── data/            # optional local data, not included in repo
└── raw_scripts/     # earlier standalone scripts kept for reference













⸻

Credits

Made by Max, Tamara, Yiliu, Lanneke, and RKD.

# YouTube Comment Sentiment Analyzer

An advanced deep learning application that harvests YouTube comments for any given video and leverages NLP techniques to extract nuanced sentiment insights. It not only generates comprehensive summaries of viewer feedback using powerful pre-trained transformer models and Google's Generative AI (Gemini) API but also employs a dedicated backup neural network for robust sentiment analysis. This innovative solution delivers deep, actionable insights into online discourse and audience engagement. With a accuracy of 88%


## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [API Keys & Configuration](#api-keys--configuration)
- [How to Run the Program](#how-to-run-the-program)
- [How the Program Works](#how-the-program-works)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **YouTube Comments Retrieval:**  
  Fetches up to 200 comments from a specified YouTube video using the YouTube Data API.

- **Sentiment Classification:**  
  Uses Facebook’s `bart-large-mnli` model in a zero-shot classification pipeline to categorize comments (e.g., positive, negative, neutral, suggestions/feedback, spam/promotional).

- **Summarization:**  
  Generates a detailed summary of the comments by leveraging the Gemini API from Google Generative AI. If the Gemini API call fails, a fallback summarizer (based on Hugging Face's `sshleifer/distilbart-cnn-12-6`) is used.

- **Backup Sentiment Analysis:**  
  A custom Keras-based neural network (trained on Amazon reviews data) serves as an additional backup for sentiment classification.

- **Downloadable Results:**  
  Provides a CSV download of the analyzed results for further review or reporting.

## Installation

### 1. Clone the Repository
Clone the repository to your local machine:
```
git clone <https://github.com/Humer-Shaik/YouTube-Comment-Sentiment-Analysis.git>
cd youtube-comment-sentiment-analyzer
```

2. Set Up a Virtual Environment
It is recommended to use a virtual environment:

```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```


3. Install Dependencies
   Then install all required packages using the provided requirements.txt:

  ```
  pip install -r requirements.txt
  ```

## API Keys & Configuration
This project requires you to supply your own API keys. In the code, placeholders have been used. You must replace these with your actual keys:

### Gemini API Key:
- Obtain this from Google Cloud Generative AI. Replace the placeholder "YOUR_GEMINI_API_KEY" in the code with your actual Gemini API key.

### YouTube Data API Key:
- Create a project in the Google Cloud Console, enable the YouTube Data API v3, and generate an API key. Replace "YOUR_YOUTUBE_API_KEY" in the code with your actual YouTube Data API key.

## How to Run the Program
###Start the Flask Application:
With your virtual environment activated, run:

  ```
python app.py
  ```
The application will start on http://0.0.0.0:5002.

Access the Web Interface:
Open your browser and navigate to http://localhost:5002.

Enter the YouTube video URL into the provided form.
The application will fetch up to 200 comments, classify their sentiment, generate a summary, and display categorized insights.
You can also download the analysis as a CSV file using the provided download link.

## Interface 
<img width="493" alt="Screenshot 2025-02-28 at 9 33 44 PM" src="https://github.com/user-attachments/assets/b06f3466-ac06-469e-8047-55ce35fbff73" />

##
<img width="512" alt="Screenshot 2025-02-28 at 9 36 18 PM" src="https://github.com/user-attachments/assets/47bbb5d8-3063-4903-9bc9-6403d8eb4108" />

## How the Program Works

### YouTube Comments:

- The application uses the YouTube Data API to collect up to 200 comments from the specified video. The comments are fetched via the API and then processed for analysis.


  
## Sentiment Analysis & Classification


### Zero-Shot Classification:
- Uses Facebook’s bart-large-mnli model to perform zero-shot classification. The model categorizes each comment into labels such as "positive", "negative", "neutral", "suggestions/feedback", and "spam/promotional".

### Backup Neural Network:
- A Keras-based neural network (trained on an Amazon reviews dataset) is used as a backup model for sentiment classification. This model can be loaded from disk if previously trained, or retrained if necessary.



### Summarization
- Primary Summarization with Gemini API:
- The application concatenates all fetched comments into a single prompt and sends it to the Gemini API (using Google's Generative AI client) for generating a detailed summary.



### Fallback Summarization:
- If the Gemini API call fails (due to network issues, API errors, etc.), the application uses a local Hugging Face summarization pipeline (sshleifer/distilbart-cnn-12-6) as a fallback.

### Output & Download
- Display of Results:
- The application displays:

### Overall sentiment counts.
- A detailed summary of the comments.
- Highlighted top positive and negative comments.



### CSV Download:
- The analysis results, including sentiment counts and comment classifications, can be downloaded as a CSV file.



## Troubleshooting


### API Key Errors:
- Ensure that your Gemini and YouTube Data API keys are valid and correctly set in the code.


### Dependency Issues:
- If you encounter errors during dependency installation, verify that your virtual environment is active and that your pip is updated.

### Model & Summarization Errors:
- The application has built-in error handling. If the Gemini API fails, the code falls back to a local summarization model. Check the console for detailed error messages to help diagnose any issues.


## License
- This project is licensed under the Apache License 2.0. See the [License](#license) file for details.





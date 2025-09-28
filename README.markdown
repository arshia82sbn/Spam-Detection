# Spam Detection with Naive Bayes 📧🔍

Welcome to the **Spam Detection** project! 🚀 This project implements a text classification system to distinguish between spam and non-spam (ham) messages using a custom Naive Bayes classifier. Built in Python using a Jupyter Notebook, it processes a dataset of text messages, trains a model, evaluates its performance, and saves predictions for submission. Let's dive into how this project works and how you can use it! 😎

## Project Overview 🌟

The goal of this project is to classify text messages as **spam** (e.g., promotional or malicious messages) or **ham** (legitimate messages) using a Naive Bayes classifier. The pipeline includes data loading, text preprocessing, model training, prediction, evaluation, and result storage. The dataset (`spam.csv`) contains messages labeled as "ham" or "spam," which we process to build an effective classifier. 📊

### Key Features
- 📝 **Text Preprocessing**: Cleans and tokenizes text for modeling.
- 🤖 **Naive Bayes Classifier**: Custom implementation with Laplace smoothing.
- 📈 **Evaluation Metrics**: Calculates Precision, Recall, F1 Score, and Accuracy.
- 📉 **Confusion Matrix**: Visualizes classification performance.
- 💾 **Result Storage**: Saves predictions and zips files for submission.

## How It Works 🛠️

The project is implemented in a Jupyter Notebook (`spam_detection.ipynb`) and follows these steps:

1. **Load Libraries** 📚
   - Imports `pandas`, `numpy`, `re`, `string`, `collections`, `matplotlib`, and `seaborn` for data handling, preprocessing, and visualization.
   - Sets a random seed (`np.random.seed(42)`) for reproducibility.

2. **Load Data** 📂
   - Reads `spam.csv` using 'latin-1' encoding.
   - Selects and renames columns: `v1` (label) to `label` and `v2` (message) to `text`.

3. **Inspect Data** 🔎
   - Displays the first 5 rows to verify the dataset's structure.

4. **Encode Labels** 🔢
   - Converts labels (`ham` → 0, `spam` → 1) for compatibility with the model.

5. **Preprocess Text** 🧹
   - Converts text to lowercase.
   - Removes numbers, punctuation, and special characters using regex.
   - Normalizes whitespace and splits text into tokens (words).
   - Stores tokens in a `processed_text` column.

6. **Split Data** ✂️
   - Splits the dataset into training (80%) and testing (20%) sets based on index order.

7. **Naive Bayes Implementation** 🧠
   - Defines a `NaiveBayes` class with:
     - **Initialization**: Sets up data structures for class probabilities, word counts, and vocabulary.
     - **Train Method**: Computes class prior probabilities and word probabilities using training data.
     - **Predict Method**: Uses log-probabilities and Laplace smoothing to classify test messages.
   - Laplace smoothing prevents zero-probability issues for unseen words.

8. **Train Model** 🚂
   - Trains the Naive Bayes model on the training data (`X_train`, `y_train`).

9. **Predict Labels** 🔮
   - Predicts labels for the test set (`X_test`), producing numerical predictions (0 for ham, 1 for spam).

10. **Evaluate Model** 📏
    - Calculates True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
    - Computes Precision, Recall, F1 Score, and Accuracy, handling division-by-zero cases.
    - Prints metrics rounded to two decimal places.

11. **Visualize Confusion Matrix** 📉
    - Plots a heatmap of the confusion matrix using `seaborn` to show classification performance.

12. **Save Results** 💾
    - Saves test predictions to `predictions.csv`, converting numerical labels back to 'ham' and 'spam'.
    - Zips the notebook and predictions into `submission.zip` for submission.

## Project Structure 📁

- `spam_detection.ipynb`: The main Jupyter Notebook with the complete pipeline.
- `predictions.csv`: Output file with test indices and predicted labels.
- `submission.zip`: Zipped file containing the notebook and predictions.
- `../Data/spam.csv`: Input dataset (not included in the repository; must be provided).

## Requirements 🛠️

To run this project, install the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

Install them using pip:

```bash
pip install pandas numpy matplotlib seaborn
```

No additional NLTK resources are required, as the project uses custom preprocessing without NLTK dependencies.

## Setup Instructions ⚙️

1. **Clone or Download the Repository** 📥
   - Clone this repository or download the project files.
   - Place the `spam.csv` dataset in the `../Data/` directory relative to the notebook.

2. **Set Up the Environment** 🖥️
   - Use Python 3.6 or higher.
   - Install the required libraries listed above.

3. **Run the Notebook** 🚀
   - Open `spam_detection.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells in order to:
     - Load and preprocess the data.
     - Train the Naive Bayes model.
     - Evaluate performance and generate visualizations.
     - Save predictions and create the submission zip.

4. **Check Outputs** ✅
   - Verify that `predictions.csv` and `submission.zip` are generated in the working directory.

## Usage Instructions 📋

1. **Run the Notebook**:
   - Open `spam_detection.ipynb` and run all cells sequentially.
   - The notebook will process the data, train the model, generate predictions, and create visualizations.

2. **Inspect Results**:
   - Check `predictions.csv` for the test set predictions (columns: `id`, `predicted_label`).
   - Extract `submission.zip` to confirm it contains `spam_detection.ipynb` and `predictions.csv`.

3. **Submit Results**:
   - Upload `submission.zip` to the grading system or share it as needed.

## Notes 📝

- **Dataset**: Ensure `spam.csv` is in the `../Data/` directory. The dataset should contain messages labeled as "ham" or "spam."
- **Data Split**: The split is deterministic (first 80% for training), which may introduce bias if the data is not shuffled. For better results, consider using `sklearn.model_selection.train_test_split`.
- **Preprocessing**: The text preprocessing is robust, correctly handling lowercase conversion and removing irrelevant characters.
- **Naive Bayes**: The custom implementation uses Laplace smoothing and log-probabilities for numerical stability and to handle unseen words.
- **Evaluation**: Metrics focus on the positive class (spam), suitable for imbalanced datasets like spam detection.
- **Visualization**: The confusion matrix provides a clear view of classification performance, with 'Blues' colormap for readability.

## Contact Me 📬

Have questions or run into issues? Reach out to me at **arshia82sbn@gmail.com**. I'm happy to help! 😊
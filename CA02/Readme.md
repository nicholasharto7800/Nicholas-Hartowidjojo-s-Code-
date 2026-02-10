# CA02 · Email Spam Classifier (Naive Bayes)

A compact, end‑to‑end text classification project that trains a *Multinomial Naive Bayes* model to detect *spam vs. non‑spam (ham)* emails.  
It (1) builds a vocabulary of the *top‑3000 most frequent* alphabetic tokens from the training set, (2) converts each email to a bag‑of‑words count vector, (3) trains the classifier, and (4) evaluates accuracy on a held‑out test set.

	⁠*Dataset*: 702 training emails (balanced between spam and ham) and 260 test emails, provided with the assignment. 

---

# What This Program Does

1.⁠ ⁠*Prepare data: Unzips ⁠ train-mails.zip ⁠ and ⁠ test-mails.zip ⁠ into the current working directory without changing folder/file names. Uses **relative paths*: ⁠ ./train-mails ⁠, ⁠ ./test-mails ⁠.
2.⁠ ⁠*Build dictionary: From **training emails only, keeps alphabetic tokens, removes single‑character tokens, and selects the most frequent **3000* words as features.
3.⁠ ⁠3. *Extract features*: Converts each email into a fixed‑length count vector aligned to the training dictionary.  
4.⁠ ⁠*Train & evaluate: Fits ⁠ MultinomialNB ⁠ on training features/labels and prints the **test accuracy*. The dataset uses file name patterns (⁠ spmsg*.txt ⁠ for spam; patterns like ⁠ 3-1msg1.txt ⁠ for ham) for labels.

---

# Requirements

•⁠  ⁠Python ≥ 3.8 (tested in Jupyter/Colab)
•⁠  ⁠Packages:
  - ⁠ numpy ⁠
  - ⁠ scikit-learn ⁠
  - (Optional for local notebooks) ⁠ jupyter ⁠

	⁠The assignment constraints require preserving the original dataset folder names and using *relative paths* exactly as ⁠ ./train-mails ⁠ and ⁠ ./test-mails ⁠. Do *not* rename or modify the original data.

# install dependencies
pip install numpy scikit-learn jupyter

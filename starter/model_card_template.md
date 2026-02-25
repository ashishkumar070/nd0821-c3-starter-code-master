# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model type:** Logistic Regression (sklearn)
- **Version:** 1.0
- **Training framework:** scikit-learn
- **Input features:** age, workclass, education, marital-status, occupation, relationship, race, sex, hours-per-week, native-country
- **Output:** Binary classification — `>50K` or `<=50K` annual income

## Intended Use
- **Primary use:** Predict whether an individual earns more or less than $50,000 per year based on census demographic data
- **Intended users:** Researchers and developers exploring income prediction models
- **Out-of-scope uses:** This model should not be used for making real financial or hiring decisions about individuals

## Training Data
- **Dataset:** UCI Census Income dataset (census.csv)
- **Size:** ~26,000 training samples (80% of full dataset)
- **Features used:** Categorical features encoded with OneHotEncoder; continuous features used as-is
- **Label:** salary column binarized with LabelBinarizer (`>50K` = 1, `<=50K` = 0)

## Evaluation Data
- **Dataset:** Same UCI Census Income dataset, held-out test split
- **Size:** ~6,500 samples (20% of full dataset)
- **Split method:** Random train/test split using sklearn `train_test_split`

## Metrics
The model was evaluated using precision, recall, and F-beta (beta=1) score on the test set:

| Metric    | Score  |
|-----------|--------|
| Precision | 0.714  |
| Recall    | 0.564  |
| Fbeta     | 0.630  |

Slice metrics were also computed for each unique value of the `education` feature and saved in `slice_output.txt`.

## Ethical Considerations
- The dataset contains sensitive demographic attributes such as race, sex, and native country which may introduce bias into predictions
- The model should not be used to make decisions that could discriminate against individuals based on these protected attributes
- Model performance may vary across demographic groups — slice evaluation is recommended before any real-world use

## Caveats and Recommendations
- The logistic regression model did not fully converge during training (max_iter=1000); increasing iterations or scaling features may improve performance
- The model was trained on 1990s US census data and may not reflect current income distributions
- It is recommended to retrain periodically with more recent data
- Feature scaling and alternative models (e.g. Random Forest, XGBoost) may yield better performance
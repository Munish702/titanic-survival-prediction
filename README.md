# Titanic: Machine Learning from Disaster

Predicting passenger survival on the Titanic ([Kaggle competition](https://www.kaggle.com/competitions/titanic)).

**Leaderboard score: 0.791** (gradient boosting with engineered features)

## Results

| Submission | What changed | Score |
|---|---|---|
| Baseline | "Women survive" rule | 0.766 |
| Random forest | Title, age fill, family size, cabin flag | 0.778 |
| Gradient boosting | + group survival feature | **0.791** |

## Approach

1. **EDA:** survival by sex, class, port, age and family size.
2. **Feature engineering:**
   - `Title` extracted from names (Mr, Mrs, Miss, Master, Rare)
   - `Age` filled with the median per title
   - `HasCabin` flag instead of the 77%-empty Cabin column
   - `FamilySize` and `IsAlone`
   - `GroupSurvival`: survival rate of the other known members of a passenger's family (surname + fare) or ticket group
3. **Modelling:** logistic regression, random forest, gradient boosting and SVM compared with 5-fold cross-validation.

## Key takeaway

Without the group feature, four different models all scored 0.825–0.836 in cross-validation. The bottleneck was the features, not the algorithm. Adding information the models didn't have before (families surviving together) produced the biggest jump.

## Running it

Place `train.csv` and `test.csv` next to `titanic_survival.ipynb` and run all cells. Requires pandas, matplotlib, seaborn and scikit-learn. The notebook writes `submission.csv`.

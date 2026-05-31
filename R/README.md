# R/ — Academic Origin (Graduate Course Project)

This folder contains the **original R analysis** that this repository's Python pipeline
is modelled on. It was a graduate Business Analytics group project analysing 8 months of
real (anonymised) hotel booking data from a hospitality client.

The dataset was small (~2,000 bookings) and confidential — only a 50-row anonymised
sample is committed (`data/sample/sample_50_rows.csv`). The Python pipeline in `src/`
rebuilds and scales this methodology on synthetic data at 70K rows.

## Files

| File | Content |
|------|---------|
| `01_load_and_clean.R` | Load 8 monthly CSVs, bind rows, drop redundant columns, factor conversion |
| `02_eda.R` | Exploratory plots — booking source, revenue, room type, cancellation distribution |
| `03_cancellation_models.R` | Classification tree (rpart) + logistic regression for cancellation prediction |
| `04_revenue_regression.R` | Multiple linear regression + stepwise backward selection for Total Amount |
| `05_clustering.R` | Hierarchical clustering (Ward + single linkage) + K-Means with elbow plot |

## How to run

```r
install.packages(c("dplyr", "ggplot2", "rpart", "rpart.plot",
                   "caret", "leaps", "cluster", "factoextra"))

# Set DATA_PATH in 01_load_and_clean.R to your folder of monthly CSVs, then:
source("R/01_load_and_clean.R")
source("R/02_eda.R")
source("R/03_cancellation_models.R")
source("R/04_revenue_regression.R")
source("R/05_clustering.R")
```

> Original monthly CSV files are not included (confidential business data).

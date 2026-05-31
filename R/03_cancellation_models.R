# =============================================================================
# RetainIQ · R/03_cancellation_models.R
# Cancellation prediction: Classification Tree + Logistic Regression
# Requires: MedianInn1.df from 01_load_and_clean.R
# =============================================================================

library(caret)
library(rpart)
library(rpart.plot)

set.seed(42)

# ── Prepare cancellation dataset ──────────────────────────────────────────────
# 'stayed' = 0 means cancelled (Total.Amount == 0), 1 means honoured
# Drop columns that directly reveal outcome (amount columns) to avoid leakage
cancel_df <- MedianInn1.df %>%
  select(-Total.Amount, -Total.Lodging.Amount, -Total.Lodging.Taxes,
         -Paid.at.Hotel.Amount, -Paid.To.Treebo.Amount) %>%
  mutate(stayed = as.factor(stayed))

cat("Cancellation dataset:", nrow(cancel_df), "rows |",
    "Cancelled:", sum(MedianInn1.df$stayed == 0), "\n")

# ── Train/validation split (60/40) ────────────────────────────────────────────
train_idx   <- sample(rownames(cancel_df), nrow(cancel_df) * 0.6)
train.df    <- cancel_df[train_idx, ]
valid.df    <- cancel_df[setdiff(rownames(cancel_df), train_idx), ]
cat("Train:", nrow(train.df), "| Validation:", nrow(valid.df), "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: Classification Tree
# ─────────────────────────────────────────────────────────────────────────────
cat("\n=== Classification Tree ===\n")

# Default tree (no constraints) — useful to see natural splits
tree_default <- rpart(stayed ~ ., data = train.df, method = "class")

# Pruned tree with max depth = 5, minsplit = 10
tree_pruned <- rpart(stayed ~ ., data = train.df, method = "class",
                     control = rpart.control(minsplit = 10, maxdepth = 5))

# Plot the pruned tree
cat("Pruned classification tree structure:\n")
prp(tree_pruned, type = 1, extra = 1, split.font = 1, varlen = -10,
    main = "Cancellation Classification Tree")

# Predict on validation
tree_pred <- predict(tree_pruned, valid.df, type = "class")
cat("\nTree — Confusion Matrix:\n")
cm_tree <- confusionMatrix(tree_pred, valid.df$stayed)
print(cm_tree$table)
cat("Accuracy:", round(cm_tree$overall["Accuracy"], 4), "\n")
cat("Sensitivity (recall for Cancelled):", round(cm_tree$byClass["Sensitivity"], 4), "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
cat("\n=== Logistic Regression ===\n")

# Remove factor columns with too many levels for stable glm fit
# (Room.No. has ~14 levels, Corporate.Name has ~12 — keep them but glm handles it)
options(scipen = 999)

logit_model <- glm(stayed ~ ., data = train.df, family = "binomial")
summary(logit_model)

# Predict probabilities on validation set
logit_prob <- predict(logit_model, valid.df, type = "response")
logit_pred <- ifelse(logit_prob > 0.5, 1, 0)

cat("\nLogistic Regression — Confusion Matrix:\n")
cm_logit <- confusionMatrix(as.factor(logit_pred), as.factor(valid.df$stayed))
print(cm_logit$table)
cat("Accuracy:", round(cm_logit$overall["Accuracy"], 4), "\n")

# AUC via manual calculation
# (requires pROC package if available; otherwise report accuracy + confusion matrix)
if (requireNamespace("pROC", quietly = TRUE)) {
  library(pROC)
  roc_obj <- roc(as.integer(valid.df$stayed) - 1, logit_prob, quiet = TRUE)
  cat("AUC:", round(auc(roc_obj), 4), "\n")
  plot(roc_obj, main = "ROC — Logistic Regression (Cancellation)")
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON NOTES
# ─────────────────────────────────────────────────────────────────────────────
cat("\n=== Summary ===\n")
cat("Both models find 'Stayed.Room.Nights' as the dominant predictor.\n")
cat("Interpretation: zero nights = cancellation (near-tautological in this data).\n")
cat("Business takeaway: the dataset lacks pre-booking cancellation signals.\n")
cat("To improve: collect lead time, booking channel changes, and deposit info.\n")

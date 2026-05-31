# =============================================================================
# RetainIQ · R/04_revenue_regression.R
# Multiple Linear Regression + Stepwise Backward Selection
# Target: Total.Amount (honoured bookings only)
# Requires: MedianInn_withoutzeros from 01_load_and_clean.R
# =============================================================================

library(caret)
library(leaps)

set.seed(42)

# ── Prepare regression dataset ────────────────────────────────────────────────
# Remove columns that directly partition amount (avoid leakage):
#   Paid.at.Hotel.Amount + Paid.To.Treebo.Amount = Total.Amount by construction
#   Total.Lodging.Amount feeds into Total.Amount via tax
reg_df <- MedianInn_withoutzeros %>%
  select(-Paid.at.Hotel.Amount, -Paid.To.Treebo.Amount, -Total.Lodging.Amount,
         -Total.Lodging.Taxes)

cat("Regression dataset:", nrow(reg_df), "honoured bookings\n")
cat("Predicting: Total.Amount (₹)\n")
cat("Mean Total.Amount:", round(mean(reg_df$Total.Amount), 2), "\n\n")

# ── Train/validation split (60/40) ────────────────────────────────────────────
train_idx  <- sample(rownames(reg_df), nrow(reg_df) * 0.6)
train2.df  <- reg_df[train_idx, ]
valid2.df  <- reg_df[setdiff(rownames(reg_df), train_idx), ]
cat("Train:", nrow(train2.df), "| Validation:", nrow(valid2.df), "\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: Full Multiple Linear Regression
# ─────────────────────────────────────────────────────────────────────────────
cat("=== Full MLR ===\n")
options(scipen = 999)
mlr_full <- lm(Total.Amount ~ ., data = train2.df)
print(summary(mlr_full))

# Validation RMSE and R²
mlr_pred <- predict(mlr_full, valid2.df)
mlr_rmse <- sqrt(mean((mlr_pred - valid2.df$Total.Amount)^2, na.rm = TRUE))
mlr_r2   <- cor(mlr_pred, valid2.df$Total.Amount, use = "complete.obs")^2
cat("\nFull MLR — Validation RMSE:", round(mlr_rmse, 2),
    "| R²:", round(mlr_r2, 4), "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: Stepwise Backward Selection
# ─────────────────────────────────────────────────────────────────────────────
cat("\n=== Stepwise Backward Selection ===\n")
mlr_step <- step(mlr_full, direction = "backward", trace = 0)
cat("Selected formula:\n")
print(formula(mlr_step))
print(summary(mlr_step))

step_pred <- predict(mlr_step, valid2.df)
step_rmse <- sqrt(mean((step_pred - valid2.df$Total.Amount)^2, na.rm = TRUE))
step_r2   <- cor(step_pred, valid2.df$Total.Amount, use = "complete.obs")^2
cat("\nStepwise MLR — Validation RMSE:", round(step_rmse, 2),
    "| R²:", round(step_r2, 4), "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Revenue by booking source (key finding)
# ─────────────────────────────────────────────────────────────────────────────
cat("\n=== Revenue Contribution by Booking Source ===\n")
rev_source <- MedianInn_withoutzeros %>%
  group_by(Booking.Source) %>%
  summarise(
    total_revenue = sum(Total.Amount, na.rm = TRUE),
    n_bookings    = n(),
    avg_per_booking = mean(Total.Amount, na.rm = TRUE)
  ) %>%
  mutate(pct_revenue = round(total_revenue / sum(total_revenue) * 100, 1)) %>%
  arrange(desc(total_revenue))
print(rev_source)

cat("\nKey finding: Corporate bookings generate the most revenue despite fewer transactions.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Classification tree on revenue (which variables drive high/low bookings?)
# ─────────────────────────────────────────────────────────────────────────────
library(rpart); library(rpart.plot)
cat("\n=== Revenue Classification Tree (High vs Low) ===\n")

# Discretise target for tree
reg_df_tree <- reg_df %>%
  mutate(revenue_tier = ifelse(Total.Amount > median(Total.Amount), "High", "Low"))

rev_tree <- rpart(revenue_tier ~ Booking.Source + Room.Type + Pax + Stayed.Room.Nights,
                  data = reg_df_tree, method = "class",
                  control = rpart.control(minsplit = 20, maxdepth = 4))
prp(rev_tree, type = 1, extra = 1, split.font = 1, varlen = -10,
    main = "Revenue Tier Classification Tree")

cat("\n✓ Revenue regression complete.\n")

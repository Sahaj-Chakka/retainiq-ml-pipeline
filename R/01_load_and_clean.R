# =============================================================================
# RetainIQ · R/01_load_and_clean.R
# Academic origin: Graduate Business Analytics group project
# Task: Load 8 months of hotel booking CSVs → clean → analysis-ready dataframe
# =============================================================================

library(dplyr)

# ── Set this to the folder containing your monthly CSV files ──────────────────
DATA_PATH <- "data/raw/"   # e.g. May.csv, June.csv, ... inside this folder

# ── 1. Load and combine 8 months ──────────────────────────────────────────────
month_files <- c("May.csv", "April.csv", "June.csv", "July.csv",
                 "August.csv", "September.csv", "October.csv", "November.csv")

raw_list <- lapply(month_files, function(f) {
  path <- file.path(DATA_PATH, f)
  if (file.exists(path)) read.csv(path, stringsAsFactors = FALSE) else NULL
})
raw_list <- Filter(Negate(is.null), raw_list)

boom <- bind_rows(raw_list)
cat("Combined rows:", nrow(boom), "| columns:", ncol(boom), "\n")

# ── 2. First-pass column drop: IDs, timestamps, redundant billing cols ────────
# Columns 1-6:  internal IDs / header artifacts
# Col 11:       Third.Party.Booking.ID (not analytically useful after joining)
# Col 17:       Payment.Mode (mostly NA in raw)
# Cols 19,21:   Modified check-in/out (duplicates of original)
# Cols 27-35:   Monthly breakdowns of inclusions (redundant with totals)
# Cols 39-50:   Withholding tax / SBC / KKC legacy columns (all near-zero)
newData.df <- boom[ , -c(1:6, 11, 17, 19, 21, 27:35, 39:50)]
cat("After first drop:", ncol(newData.df), "columns\n")

# ── 3. Second-pass: keep only the 12 analytically meaningful columns ──────────
# Drops: Guest sharer name (col 2,3), redundant amount breakdowns (cols 8,9),
#        monthly aggregates (cols 20-30, 32-50)
MedianInn.df <- newData.df[ , -c(2, 3, 8, 9, 20:30, 32:50)]
cat("After second drop:", ncol(MedianInn.df), "columns\n")
cat("Columns kept:\n"); print(names(MedianInn.df))

# ── 4. Type conversion: categorical → factor ───────────────────────────────────
MedianInn.df$Booking.Source         <- as.factor(MedianInn.df$Booking.Source)
MedianInn.df$Corporate.Name         <- as.factor(MedianInn.df$Corporate.Name)
MedianInn.df$Room.No.               <- as.factor(MedianInn.df$Room.No.)
MedianInn.df$Room.Type              <- as.factor(MedianInn.df$Room.Type)
MedianInn.df$Pax                    <- as.factor(MedianInn.df$Pax)
MedianInn.df$Guest.Invoice.issued.by <- as.factor(MedianInn.df$Guest.Invoice.issued.by)

# Remove remaining non-essential categorical detail (invoice agent, POS flag, etc.)
MedianInn1.df <- MedianInn.df[ , -c(8:12, 16)]

cat("\nFinal dataframe dimensions:", nrow(MedianInn1.df), "×", ncol(MedianInn1.df), "\n")
summary(MedianInn1.df)

# ── 5. Create a clean working copy excluding zero-revenue rows (cancellations) ─
# Zero Total.Lodging.Amount = cancelled / no-show booking
MedianInn_withoutzeros <- MedianInn1.df %>%
  filter(Total.Lodging.Amount != 0.00)

cat("\nRows after removing zero-lodging cancellations:", nrow(MedianInn_withoutzeros), "\n")
cat("Rows removed (cancellations):", nrow(MedianInn1.df) - nrow(MedianInn_withoutzeros), "\n")

# ── 6. Flag for cancellation analysis (binary: 0 = cancelled, 1 = stayed) ─────
MedianInn1.df$stayed <- ifelse(MedianInn1.df$Total.Amount == 0, 0, 1)

# Quick sanity check: any suspicious entries?
suspicious <- MedianInn_withoutzeros[MedianInn_withoutzeros$Paid.To.Treebo.Amount < -1000, ]
if (nrow(suspicious) > 0) {
  cat("\n⚠ Suspicious negative Treebo payments:\n")
  print(suspicious[, c("Booking.Source", "Corporate.Name", "Total.Amount", "Paid.To.Treebo.Amount")])
}

cat("\n✓ Data loading and cleaning complete.\n")
cat("Objects available: MedianInn1.df (all rows), MedianInn_withoutzeros (honoured only)\n")

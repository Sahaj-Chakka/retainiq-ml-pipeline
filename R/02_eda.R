# =============================================================================
# RetainIQ · R/02_eda.R
# Exploratory Data Analysis
# Requires: MedianInn1.df and MedianInn_withoutzeros from 01_load_and_clean.R
# =============================================================================

library(ggplot2)
library(dplyr)

# ── 1. Booking source distribution ───────────────────────────────────────────
cat("=== Booking Source Counts ===\n")
print(summary(MedianInn_withoutzeros$Booking.Source))

ggplot(MedianInn_withoutzeros) +
  geom_bar(aes(x = Booking.Source, fill = Booking.Source), show.legend = FALSE) +
  theme_minimal(base_size = 12) +
  labs(title = "Number of Bookings by Source",
       x = "Booking Source", y = "Count") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# ── 2. Revenue by booking source ──────────────────────────────────────────────
cat("\n=== Revenue by Booking Source (Total Lodging Amount) ===\n")
rev_by_source <- MedianInn_withoutzeros %>%
  group_by(Booking.Source) %>%
  summarise(total_revenue = sum(Total.Lodging.Amount, na.rm = TRUE),
            mean_revenue   = mean(Total.Lodging.Amount, na.rm = TRUE),
            n_bookings     = n()) %>%
  arrange(desc(total_revenue))
print(rev_by_source)

ggplot(MedianInn_withoutzeros) +
  geom_boxplot(aes(x = Booking.Source, y = Total.Lodging.Amount, fill = Booking.Source),
               show.legend = FALSE, outlier.alpha = 0.3) +
  theme_minimal(base_size = 12) +
  labs(title = "Revenue Distribution by Booking Source",
       x = "Booking Source", y = "Total Lodging Amount (₹)") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# ── 3. Room type preference ───────────────────────────────────────────────────
cat("\n=== Room Type Preference ===\n")
print(prop.table(table(MedianInn_withoutzeros$Room.Type)) * 100)

ggplot(MedianInn_withoutzeros) +
  geom_bar(aes(x = Room.Type, fill = Room.Type), show.legend = FALSE) +
  theme_minimal(base_size = 12) +
  labs(title = "Room Type Demand", x = "Room Type", y = "Count")

# ── 4. Cancellation overview ──────────────────────────────────────────────────
cat("\n=== Cancellations ===\n")
cancel_df <- MedianInn1.df %>% count(stayed) %>%
  mutate(label = ifelse(stayed == 0, "Cancelled", "Honoured"),
         pct   = round(n / sum(n) * 100, 1))
print(cancel_df)

ggplot(cancel_df, aes(x = label, y = n, fill = label)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = paste0(pct, "%")), vjust = -0.3, size = 4) +
  theme_minimal(base_size = 12) +
  labs(title = "Booking Outcome: Honoured vs Cancelled",
       x = "", y = "Count")

# ── 5. Stayed room nights distribution ────────────────────────────────────────
cat("\n=== Stayed Room Nights Distribution ===\n")
print(table(MedianInn_withoutzeros$Stayed.Room.Nights))

ggplot(MedianInn_withoutzeros) +
  geom_bar(aes(x = factor(Stayed.Room.Nights)), fill = "#E85D04") +
  theme_minimal(base_size = 12) +
  labs(title = "Distribution of Length of Stay",
       x = "Nights Stayed", y = "Count")

# ── 6. Corporate bookings: revenue concentration ──────────────────────────────
cat("\n=== Top Corporate Accounts by Revenue ===\n")
corp <- MedianInn_withoutzeros %>%
  filter(!is.na(Corporate.Name) & Corporate.Name != "") %>%
  group_by(Corporate.Name) %>%
  summarise(revenue = sum(Total.Lodging.Amount, na.rm = TRUE),
            bookings = n()) %>%
  arrange(desc(revenue))
print(corp)

ggplot(MedianInn_withoutzeros, aes(x = Corporate.Name, y = Total.Lodging.Amount)) +
  geom_point(colour = "navy", alpha = 0.6, size = 2) +
  theme_minimal(base_size = 11) +
  labs(title = "Revenue by Corporate Client",
       x = "Corporate Name", y = "Total Lodging Amount (₹)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ── 7. Pax (guest count) breakdown ───────────────────────────────────────────
cat("\n=== Pax (Guest Count) ===\n")
print(summary(MedianInn_withoutzeros$Pax))

# ── 8. Correlation among numeric features ─────────────────────────────────────
num_df <- MedianInn_withoutzeros %>%
  select(Total.Lodging.Amount, Total.Amount, Paid.at.Hotel.Amount, Paid.To.Treebo.Amount)
cat("\n=== Correlation Matrix (numeric features) ===\n")
print(round(cor(num_df, use = "complete.obs"), 3))

cat("\n✓ EDA complete.\n")

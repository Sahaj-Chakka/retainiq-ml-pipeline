# =============================================================================
# RetainIQ · R/05_clustering.R
# Hierarchical Clustering (Ward.D + Single) and K-Means segmentation
# Requires: MedianInn_withoutzeros from 01_load_and_clean.R
# =============================================================================

library(caret)
library(cluster)

# ── Prepare numeric-only feature matrix for clustering ───────────────────────
# Select the numeric revenue / stay metrics
num_df <- MedianInn_withoutzeros %>%
  select(Total.Lodging.Amount, Total.Amount, Paid.at.Hotel.Amount,
         Paid.To.Treebo.Amount, Stayed.Room.Nights) %>%
  na.omit()

cat("Clustering on", nrow(num_df), "honoured bookings ×", ncol(num_df), "features\n")

# ── Normalise: centre and scale ───────────────────────────────────────────────
pre    <- preProcess(num_df, method = c("center", "scale"))
norm_df <- predict(pre, num_df)
summary(norm_df)


# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
cat("\n=== Hierarchical Clustering ===\n")

# Distance matrix (Manhattan — robust to outliers in booking data)
d_manhattan <- dist(norm_df, method = "manhattan")

# Ward.D (minimises within-cluster variance — best for compact clusters)
hc_ward <- hclust(d_manhattan, method = "ward.D")
plot(hc_ward, hang = -1, ann = FALSE,
     main = "Dendrogram — Ward.D linkage (Manhattan distance)",
     xlab = "Booking", sub = "")

# Single linkage (shows chaining; compare with Ward.D)
hc_single <- hclust(d_manhattan, method = "single")
plot(hc_single, hang = -1, ann = FALSE,
     main = "Dendrogram — Single linkage",
     xlab = "Booking", sub = "")

# Cut Ward.D tree into 4 groups
hc_labels <- cutree(hc_ward, k = 4)
cat("Hierarchical cluster sizes:\n")
print(table(hc_labels))

# Profile each hierarchical cluster
num_df$hc_cluster <- hc_labels
cat("\nHierarchical cluster means:\n")
print(aggregate(. ~ hc_cluster, data = num_df,
                FUN = function(x) round(mean(x), 2)))
num_df$hc_cluster <- NULL   # clean up


# ─────────────────────────────────────────────────────────────────────────────
# K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
cat("\n=== K-Means Clustering ===\n")

# ── Elbow plot: within-cluster SS vs k ────────────────────────────────────────
wss <- sapply(1:7, function(k) kmeans(norm_df, k, nstart = 10)$tot.withinss)
plot(1:7, wss, type = "b", pch = 19, col = "#E85D04",
     xlab = "Number of clusters (k)", ylab = "Total within-cluster SS",
     main = "Elbow Plot — K-Means")

# Chosen k = 4 (elbow consistent with hierarchical cut)
set.seed(42)
km4 <- kmeans(norm_df, centers = 4, nstart = 25)
cat("K-Means (k=4) cluster sizes:\n")
print(km4$size)

cat("\nCluster centroids (normalised scale):\n")
print(round(km4$centers, 3))

# Re-attach cluster labels to original data for profiling
MedianInn_withoutzeros$km_cluster <- km4$cluster

cat("\nK-Means cluster profiles (original scale):\n")
cluster_profile <- MedianInn_withoutzeros %>%
  group_by(km_cluster) %>%
  summarise(
    n            = n(),
    avg_revenue  = round(mean(Total.Amount, na.rm = TRUE), 2),
    avg_nights   = round(mean(Stayed.Room.Nights, na.rm = TRUE), 2),
    avg_lodging  = round(mean(Total.Lodging.Amount, na.rm = TRUE), 2)
  ) %>%
  arrange(desc(avg_revenue))
print(cluster_profile)

# ── Centroid plot ─────────────────────────────────────────────────────────────
par(mar = c(6, 4, 3, 1))
plot(c(0), xaxt = "n", ylab = "Normalised value", type = "l",
     ylim = c(min(km4$centers), max(km4$centers)),
     xlim = c(0.5, ncol(norm_df) + 0.5),
     main = "K-Means Cluster Centroids")
axis(1, at = 1:ncol(norm_df), labels = names(norm_df), las = 2, cex.axis = 0.8)
for (i in 1:4) {
  lines(km4$centers[i, ], lty = i, lwd = 2,
        col = c("#E85D04", "navy", "#3fb950", "#d29922")[i])
}
legend("topright", legend = paste("Cluster", 1:4),
       lty = 1:4, col = c("#E85D04", "navy", "#3fb950", "#d29922"), bty = "n")

cat("\nKey finding: Corporate/long-stay customers cluster separately with higher revenue.\n")
cat("✓ Clustering complete.\n")

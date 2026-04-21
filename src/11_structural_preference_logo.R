#!/usr/bin/env Rscript

# =============================================================================
# Structural Preference "Pseudo-Logo" Plot
# Maps NetSurfP-3.0 Q8 states across 5 spatial regions (N-flank to C-flank)
# =============================================================================

library(tidyverse)
library(ggseqlogo)
library(ggplot2)
source("src/functions.R")
set_working_directory()

# 1. Load Data
df_all <- read_csv("data/processed/df_all.csv", show_col_types = FALSE)

# Ensure label is numeric 0/1
df_clean <- df_all |> 
  filter(!is.na(label)) |> 
  mutate(label = as.numeric(label))

df_1 <- df_clean |> filter(label == 1)
df_0 <- df_clean |> filter(label == 0)

# 2. Define our 8 "Letters" (Q8 States) and 5 "Positions" (Regions)
q8_states <- c("H", "G", "I", "E", "B", "T", "S", "C")
regions <- c("nflank", "n_pep", "peptide", "pep_c", "cflank")
region_labels <- c("N-Flank", "N→Pep Boundary", "Peptide", "Pep→C Boundary", "C-Flank")

# 3. Helper function to map a state and region to your exact column names
get_col_name <- function(state, region) {
  if (region %in% c("nflank", "peptide", "cflank")) {
    return(paste0("q8point_", state, "_", region))
  } else {
    return(paste0("q8trans_", state, "_", region))
  }
}

# 4. Build the Difference Matrix
# Rows = Q8 states, Cols = Regions
diff_mat <- matrix(0, nrow = 8, ncol = 5)
rownames(diff_mat) <- q8_states
colnames(diff_mat) <- region_labels

cat("Calculating structural probabilities...\n")
for (i in seq_along(q8_states)) {
  for (j in seq_along(regions)) {
    col <- get_col_name(q8_states[i], regions[j])
    
    if (col %in% names(df_clean)) {
      # Proportion of sequences having this structure in Positives vs Negatives
      prop_1 <- mean(df_1[[col]], na.rm = TRUE)
      prop_0 <- mean(df_0[[col]], na.rm = TRUE)
      diff_mat[i, j] <- prop_1 - prop_0
    } else {
      warning(paste("Column not found:", col))
    }
  }
}

# 5. Define a Custom Color Scheme for Secondary Structures
# Helices = Blues, Strands = Reds, Turns/Bends = Greens, Coil = Grey
structural_colors <- make_col_scheme(
  chars = c("H", "G", "I", "E", "B", "T", "S", "C"),
  groups = c("Helix", "Helix", "Helix", "Strand", "Strand", "Turn/Bend", "Turn/Bend", "Coil"),
  cols = c("#08519c", "#3182bd", "#6baed6", "#a50f15", "#fb6a4a", "#238b45", "#74c476", "#525252")
)

# 6. Generate the "Pseudo-Logo"
cat("Generating Structural Differential Logo...\n")

p_struct_logo <- ggseqlogo(diff_mat, method = "custom", col_scheme = structural_colors) +
  labs(
    title = "Differential Structural Logo: Presented vs. Not Presented",
    subtitle = "Top = Structure Enriched in Presented | Bottom = Structure Depleted in Presented",
    x = "Spatial Region",
    y = "Difference in Probability"
  ) +
  scale_x_continuous(breaks = 1:5, labels = region_labels) +
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.5) +
  theme_logo() +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey30"),
    axis.text.x = element_text(size = 11, face = "bold", angle = 0),
    panel.grid.major.y = element_line(color = "grey90")
  )

# 7. Save
dir.create("results/figures/categorical", showWarnings = FALSE, recursive = TRUE)
out_file <- "results/figures/categorical/structural_differential_logo.png"
ggsave(out_file, plot = p_struct_logo, width = 10, height = 5, dpi = 300)
cat("Saved:", out_file, "\n")
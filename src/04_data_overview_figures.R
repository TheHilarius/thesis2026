library(networkD3)
library(dplyr)
library(tidyverse)
library(eulerr)
source("src/functions.R")
set_working_directory()

# ============================================================================
# 0. LOAD COUNTS
# ============================================================================

# --- Hardcoded: from upstream scripts that already ran ---
# These steps happened before pos_EL_all_epitopes_hla0201.csv was saved,
# so the information is no longer recoverable from that file.
n_raw_assays   <- 200381
n_duplicates   <- 123531
n_unique_pairs <- 76850
n_ptm          <- 15138
n_no_ptm       <- 61712
n_non9mer      <- 30975
n_9mer_raw     <- 30737   # before O60361 removal and FASTA filter

# --- Dynamic: read from pipeline outputs ---
df_9mer       <- read_csv("data/processed/pos_EL_9mers_epitopes_hla0201.csv")
df_combined   <- read_csv("data/processed/df_combined_pos_and_neg.csv")
df_binders    <- read_csv("data/processed/netmhcpan_9mer_binders.csv")
df_net_only   <- read_csv("data/processed/netmhcpan_9mer_only.csv")
df_iedb_only  <- read_csv("data/processed/iedb_9mer_only.csv")

n_9mer_verified <- nrow(df_9mer)
n_obsolete      <- n_9mer_raw - n_9mer_verified
n_positives     <- sum(df_combined$label == 1)
n_fn            <- nrow(df_iedb_only)

n_predicted_binders <- nrow(df_binders)
n_overlap           <- n_positives
n_unconfirmed       <- nrow(df_net_only)
n_negatives         <- sum(df_combined$label == 0)
n_affinity_removed  <- n_unconfirmed - n_negatives
n_rank_matched      <- n_negatives

cat("=== Positives Pipeline ===\n")
cat("Raw assays:             ", scales::comma(n_raw_assays), "\n")
cat("Duplicates:             ", scales::comma(n_duplicates), "\n")
cat("Unique pairs:           ", scales::comma(n_unique_pairs), "\n")
cat("PTMs:                   ", scales::comma(n_ptm), "\n")
cat("No-PTMs:                ", scales::comma(n_no_ptm), "\n")
cat("Non-9-mers:             ", scales::comma(n_non9mer), "\n")
cat("9-mers (pre-FASTA):     ", scales::comma(n_9mer_raw), "\n")
cat("Obsolete UniProt:       ", scales::comma(n_obsolete), "\n")
cat("9-mers (FASTA-verified):", scales::comma(n_9mer_verified), "\n")
cat("Non-binders (FN):       ", scales::comma(n_fn), "\n")
cat("Binders (TP):           ", scales::comma(n_positives), "\n")

cat("\n=== Negatives Pipeline ===\n")
cat("Predicted binders:      ", scales::comma(n_predicted_binders), "\n")
cat("Overlap (IEDB):         ", scales::comma(n_overlap), "\n")
cat("Unconfirmed:            ", scales::comma(n_unconfirmed), "\n")
cat("Affinity removed:       ", scales::comma(n_affinity_removed), "\n")
cat("Rank-matched (TN):      ", scales::comma(n_rank_matched), "\n")

# Sanity checks
stopifnot(n_duplicates + n_unique_pairs == n_raw_assays)
stopifnot(n_ptm + n_no_ptm == n_unique_pairs)
stopifnot(n_non9mer + n_9mer_raw == n_no_ptm)
stopifnot(n_obsolete + n_9mer_verified == n_9mer_raw)
stopifnot(n_fn + n_positives == n_9mer_verified)

cat("\n✓ All cascade counts verified.\n")

# ==============================================================================
# 1. POSITIVES PIPELINE SANKEY
# ==============================================================================

nodes_pos <- data.frame(name = c(
  paste0("Raw Assays: ",        scales::comma(n_raw_assays)),        # 0
  paste0(scales::comma(n_duplicates)),        # 1
  paste0("Unique Peptides: ",      scales::comma(n_unique_pairs)),      # 2
  paste0("PTMs: ",              scales::comma(n_ptm)),               # 3
  paste0(scales::comma(n_no_ptm)),            # 4
  paste0("Non-9-mers: ",        scales::comma(n_non9mer)),           # 5
  paste0(scales::comma(n_9mer_raw)),          # 6
  paste0("Missing UniProt Fasta: ",  scales::comma(n_obsolete)),          # 7
  paste0(scales::comma(n_9mer_verified)),     # 8
  paste0("Non-binders: ",       scales::comma(n_fn)),                # 9
  paste0("9-mer Binders (TP): ",      scales::comma(n_positives))          # 10
))

links_pos <- data.frame(
  source = c(0, 0, 2, 2, 4, 4, 6, 6, 8, 8),
  target = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
  value  = c(n_duplicates, n_unique_pairs, n_ptm, n_no_ptm,
             n_non9mer, n_9mer_raw, n_obsolete, n_9mer_verified,
             n_fn, n_positives)
)

nodes_pos$group <- c("raw",     "discard", "keep",    "discard", "keep",
                     "discard", "keep",    "discard", "keep",    "discard", "keep")
links_pos$group <- c("discard", "keep",    "discard", "keep",
                     "discard", "keep",    "discard", "keep",
                     "discard", "keep")

color_pos <- 'd3.scaleOrdinal()
  .domain(["keep", "discard", "raw"])
  .range(["#2ecc71", "#e0e0e0", "#cccccc"]);'

sankey_pos <- sankeyNetwork(
  Links = links_pos, Nodes = nodes_pos, Source = "source", Target = "target",
  Value = "value", NodeID = "name", NodeGroup = "group", LinkGroup = "group",
  colourScale = color_pos, sinksRight = FALSE,
  fontFamily = "Arial", fontSize = 16, nodeWidth = 20, nodePadding = 25,
  margin = list(t = 20, r = 20, b = 20, l = 20),
  iterations = 0
)

# ==============================================================================
# 2. NEGATIVES PIPELINE SANKEY
# ==============================================================================

nodes_neg <- data.frame(name = c(
  paste0("Predicted Binders: ",  scales::comma(n_predicted_binders)),
  paste0("Overlap (IEDB): ",     scales::comma(n_overlap)),
  paste0("Unconfirmed: ",        scales::comma(n_unconfirmed)),
  paste0("Affinity Bias: ",      scales::comma(n_affinity_removed)),
  paste0("Rank-Matched (TN): ",  scales::comma(n_rank_matched))
))

links_neg <- data.frame(
  source = c(0, 0, 2, 2),
  target = c(1, 2, 3, 4),
  value  = c(n_overlap, n_unconfirmed, n_affinity_removed, n_rank_matched)
)

nodes_neg$group <- c("raw", "discard", "keep", "discard", "keep")
links_neg$group <- c("discard", "keep", "discard", "keep")

color_neg <- 'd3.scaleOrdinal()
  .domain(["keep", "discard", "raw"])
  .range(["#e74c3c", "#e0e0e0", "#cccccc"]);'

sankey_neg <- sankeyNetwork(
  Links = links_neg, Nodes = nodes_neg, Source = "source", Target = "target",
  Value = "value", NodeID = "name", NodeGroup = "group", LinkGroup = "group",
  colourScale = color_neg, sinksRight = FALSE,
  fontFamily = "Arial", fontSize = 16, nodeWidth = 20, nodePadding = 25,
  margin = list(t = 20, r = 20, b = 20, l = 20),
  iterations = 0
)

# ==============================================================================
# 3. PROPORTIONAL VENN DIAGRAM
# ==============================================================================

venn_data <- c(
  "NetMHCpan" = n_unconfirmed,
  "IEDB" = n_fn,
  "NetMHCpan&IEDB" = n_overlap
)

fit <- euler(venn_data)

venn_plot <- plot(
  fit,
  quantities = list(cex = 1.3, font = 2),
  labels = FALSE,
  legend = list(
    labels = c("NetMHCpan Binders (<2% Rank)", "IEDB Confirmed (Positives)"),
    cex = 1.1,
    side = "bottom"
  ),
  fills = list(fill = c("#e74c3c", "#2ecc71"), alpha = 0.7),
  edges = list(col = "black", lwd = 2),
  main = "9-mer Peptides: Predicted vs. Biologically Presented"
)

# ==============================================================================
# 4. VIEW & EXPORT
# ==============================================================================

sankey_pos
sankey_neg
venn_plot

saveNetwork(sankey_pos, "results/sankey_positives_polished.html")
saveNetwork(sankey_neg, "results/sankey_negatives_polished.html")

png("results/venn_diagram_overlap.png", width = 800, height = 650, res = 150)
print(venn_plot)
dev.off()

cat("✅ Saved all overview figures to results/ folder\n")





library(tidyverse)
library(ggplot2)

# ==============================================================================
# 1. DEFINE NODES WITH GENEROUS SPACING
# ==============================================================================
# We use a much wider coordinate system (0 to 20) to give huge margins.

df_nodes <- tibble(
  id    = 1:4,
  x     = c(2.5, 7.5, 12.5, 17.5), # Centers spread far apart
  y     = c(2, 2, 2, 2),
  width = c(4.2, 4.2, 4.2, 4.2),   # Much wider boxes
  height= c(2.0, 2.0, 2.0, 2.0),   # Taller boxes for vertical padding
  
  # Split text into Title and Subtitle for professional typography
  title = c(
    "1. Source Proteins",
    "2. Sliding Window",
    "3. NetMHCpan 4.2",
    "4. Predicted Binders"
  ),
  subtitle = c(
    "(9,995 FASTA sequences)",
    "(Extract all 9-mers)",
    "(Predict EL Rank)",
    "Rank < 2% (N = 271,428)"
  ),
  
  # Colors: Clean white boxes for 1-3, Sankey-matching Blue for 4
  fill  = c("#ffffff", "#ffffff", "#ffffff", "#3498db"),
  color = c("#ced4da", "#ced4da", "#ced4da", "#2980b9"), # Thin elegant borders
  
  # Text Colors
  title_col = c("#212529", "#212529", "#212529", "#ffffff"),
  sub_col   = c("#6c757d", "#6c757d", "#6c757d", "#f1f5f9")
)


# ==============================================================================
# 2. DEFINE ARROWS
# ==============================================================================

df_edges <- tibble(
  x    = c(4.8, 9.8, 14.8),  # Start slightly away from box edge
  xend = c(5.2, 10.2, 15.2), # End slightly away from next box edge
  y    = c(2, 2, 2),
  yend = c(2, 2, 2)
)

# ==============================================================================
# 3. BUILD THE GGPLOT
# ==============================================================================

p_pipeline <- ggplot() +
  
  # 1. Drop Shadows (draw slightly offset light-grey boxes first)
  geom_rect(data = df_nodes,
            aes(xmin = x - width/2 + 0.08, xmax = x + width/2 + 0.08,
                ymin = y - height/2 - 0.08, ymax = y + height/2 - 0.08),
            fill = "#e9ecef", color = NA) +
  
  # 2. Draw arrows
  geom_segment(data = df_edges, 
               aes(x = x, y = y, xend = xend, yend = yend),
               arrow = arrow(length = unit(0.3, "cm"), type = "closed"),
               color = "#adb5bd", linewidth = 1.2) +
  
  # 3. Draw main boxes
  geom_rect(data = df_nodes,
            aes(xmin = x - width/2, xmax = x + width/2,
                ymin = y - height/2, ymax = y + height/2,
                fill = fill, color = color),
            linewidth = 0.8) + # Thinner, cleaner border
  
  # 4. Draw Titles (Bold, centered upper)
  geom_text(data = df_nodes,
            aes(x = x, y = y + 0.25, label = title, color = title_col),
            fontface = "bold", size = 5.5) +
  
  # 5. Draw Subtitles (Regular, centered lower)
  geom_text(data = df_nodes,
            aes(x = x, y = y - 0.35, label = subtitle, color = sub_col),
            fontface = "plain", size = 4.5) +
  
  # Apply exact hex codes
  scale_fill_identity() +
  scale_color_identity() +
  
  # Set perfect aspect ratio limits (20 wide x 4 high = 5:1 ratio)
  coord_cartesian(xlim = c(0, 20), ylim = c(0, 4)) +
  theme_void() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(10, 10, 10, 10)
  )

# ==============================================================================
# 4. EXPORT
# ==============================================================================
# Save with a matching 5:1 aspect ratio to prevent any stretching
ggsave("results/methods_generation_pipeline.png", p_pipeline, 
       width = 15, height = 3, dpi = 300)

cat("✅ Polished pipeline saved to results/methods_generation_pipeline.png\n")


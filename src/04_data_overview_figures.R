library(networkD3)
library(tidyverse)
source("src/functions.R")
set_working_directory()
library(eulerr)


# ============================================================================
# 0. LOAD COUNTS
# ============================================================================

counts_raw <- read_csv("data/processed/sankey_counts.csv", show_col_types = FALSE)
ct <- setNames(counts_raw$count, counts_raw$stage)

# Sanity checks
stopifnot(ct["duplicates"] + ct["unique_pairs"] == ct["raw_assays"])
stopifnot(ct["ptm"] + ct["no_ptm"] == ct["unique_pairs"])
stopifnot(ct["non_9mer"] + ct["9mer_all"] == ct["no_ptm"])
stopifnot(ct["na_uniprot"] + ct["o60361"] + ct["dedup"] +
            ct["missing_fasta"] + ct["selenocysteine"] +
            ct["nsp3_length"] + ct["alphafold"] +
            ct["9mer_verified"] == ct["9mer_all"])
stopifnot(ct["iedb_recovered"] + ct["iedb_missed"] == ct["9mer_verified"])

cat("✓ All cascade counts verified.\n\n")

# Print for reference
cat("=== Positives Pipeline ===\n")
cat("Raw assays:             ", scales::comma(ct["raw_assays"]), "\n")
cat("Duplicates:             ", scales::comma(ct["duplicates"]), "\n")
cat("Unique pairs:           ", scales::comma(ct["unique_pairs"]), "\n")
cat("PTMs:                   ", scales::comma(ct["ptm"]), "\n")
cat("No-PTMs:                ", scales::comma(ct["no_ptm"]), "\n")
cat("Non-9-mers:             ", scales::comma(ct["non_9mer"]), "\n")
cat("9-mers:                 ", scales::comma(ct["9mer_all"]), "\n")
cat("  NA uniprot_id:        ", scales::comma(ct["na_uniprot"]), "\n")
cat("  O60361:               ", scales::comma(ct["o60361"]), "\n")
cat("  Dedup:                ", scales::comma(ct["dedup"]), "\n")
cat("  Missing FASTA:        ", scales::comma(ct["missing_fasta"]), "\n")
cat("  Selenocysteine:       ", scales::comma(ct["selenocysteine"]), "\n")
cat("Verified 9-mers:        ", scales::comma(ct["9mer_verified"]), "\n")
cat("  Recovered (TP):       ", scales::comma(ct["iedb_recovered"]), "\n")
cat("  Missed (FN):          ", scales::comma(ct["iedb_missed"]), "\n")

cat("\n=== Negatives Pipeline ===\n")
cat("Predicted binders:      ", scales::comma(ct["predicted_binders"]), "\n")
cat("Overlap (TP):           ", scales::comma(ct["tp_in_combined"]), "\n")
cat("Unconfirmed:            ", scales::comma(ct["netmhcpan_only"]), "\n")
cat("Affinity removed:       ", scales::comma(ct["affinity_removed"]), "\n")
cat("Rank-matched (TN):      ", scales::comma(ct["negatives_combined"]), "\n")

# ==============================================================================
# 1. POSITIVES PIPELINE SANKEY
# ==============================================================================

nodes_pos <- data.frame(name = c(
  paste0("Raw Assays: ",              scales::comma(ct["raw_assays"])),       # 0
  paste0(scales::comma(ct["duplicates"])),                                    # 1
  paste0("Unique Peptides: ",         scales::comma(ct["unique_pairs"])),     # 2
  paste0("PTMs: ",                    scales::comma(ct["ptm"])),              # 3
  paste0(scales::comma(ct["no_ptm"])),                                        # 4
  paste0("Non-9-mers: ",             scales::comma(ct["non_9mer"])),          # 5
  paste0(scales::comma(ct["9mer_all"])),                                      # 6
  paste0("No UniProt ID: ",          scales::comma(ct["na_uniprot"])),        # 7
  paste0("No FASTA: ",               scales::comma(ct["missing_fasta"])),     # 8
  paste0("Selenocysteine: ",         scales::comma(ct["selenocysteine"])),    # 9
  paste0("Verified 9-mers: ",        scales::comma(ct["9mer_verified"])),     # 10
  paste0("Not Predicted (FN): ",     scales::comma(ct["iedb_missed"])),       # 11
  paste0("Predicted (TP): ",         scales::comma(ct["iedb_recovered"]))     # 12
))

links_pos <- data.frame(
  source = c(0, 0, 2, 2, 4, 4, 6, 6, 6, 6, 10, 10),
  target = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
  value  = c(ct["duplicates"], ct["unique_pairs"],
             ct["ptm"], ct["no_ptm"],
             ct["non_9mer"], ct["9mer_all"],
             ct["na_uniprot"], ct["missing_fasta"], ct["selenocysteine"],
             ct["9mer_verified"],
             ct["iedb_missed"], ct["iedb_recovered"])
)

nodes_pos$group <- c("raw",     "discard", "keep",    "discard", "keep",
                     "discard", "keep",
                     "discard", "discard", "discard", "keep",
                     "discard", "keep")
links_pos$group <- c("discard", "keep",
                     "discard", "keep",
                     "discard", "keep",
                     "discard", "discard", "discard", "keep",
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
  paste0("Predicted Binders: ",  scales::comma(ct["predicted_binders"])),
  paste0("Overlap (IEDB): ",     scales::comma(ct["tp_in_combined"])),
  paste0("Unconfirmed: ",        scales::comma(ct["netmhcpan_only"])),
  paste0("Affinity Bias: ",      scales::comma(ct["affinity_removed"])),
  paste0("Rank-Matched (TN): ",  scales::comma(ct["negatives_combined"]))
))

links_neg <- data.frame(
  source = c(0, 0, 2, 2),
  target = c(1, 2, 3, 4),
  value  = c(ct["tp_in_combined"], ct["netmhcpan_only"],
             ct["affinity_removed"], ct["negatives_combined"])
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

#####
#####
#####
#####
#####
#####
#####
#####
#####
#####
#####
#####


# ==============================================================================
# 3. PROPORTIONAL VENN DIAGRAM
# ==============================================================================

venn_data <- c(
  "NetMHCpan" = ct["netmhcpan_only"],
  "IEDB" = ct["iedb_missed"],
  "NetMHCpan&IEDB" = ct["iedb_recovered"]
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

dir.create("results/sankey", showWarnings = FALSE, recursive = TRUE)

saveNetwork(sankey_pos, "results/sankey/sankey_positives.html")
saveNetwork(sankey_neg, "results/sankey/sankey_negatives.html")

cat("Saved: results/sankey/sankey_positives.html\n")
cat("Saved: results/sankey/sankey_negatives.html\n")

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


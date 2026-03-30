library(networkD3)
library(dplyr)
library(eulerr)

# ==============================================================================
# 1. POSITIVES PIPELINE SANKEY
# ==============================================================================

nodes_pos <- data.frame(name = c(
  "Raw Assays: 200,381",      
  "Duplicates: 123,531",      
  "Unique Pairs: 76,850",     
  "PTMs: 15,138",             
  "No-PTMs: 61,712",          
  "Non-9-mers: 30,975",       
  "9-mers: 30,737",           
  "Non-binders: 7,185",       
  "Binders (TP): 23,552"      
))

links_pos <- data.frame(
  source = c(0, 0, 2, 2, 4, 4, 6, 6),
  target = c(1, 2, 3, 4, 5, 6, 7, 8),
  value  = c(123531, 76850, 15138, 61712, 30975, 30737, 7185, 23552)
)

# Colors: 'raw' for start, 'discard' for up-branches, 'keep' for bottom
nodes_pos$group <- c("raw", "discard", "keep", "discard", "keep", "discard", "keep", "discard", "keep")
links_pos$group <- c("discard", "keep", "discard", "keep", "discard", "keep", "discard", "keep")

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
  "Predicted Binders: 274,122",   
  "Overlap (IEDB): 23,552",       
  "Unconfirmed: 250,570",         
  "Affinity Bias: 220,507",       
  "Rank-Matched (TN): 30,063"     
))

links_neg <- data.frame(
  source = c(0, 0, 2, 2),
  target = c(1, 2, 3, 4),
  value  = c(23552, 250570, 220507, 30063)
)

nodes_neg$group <- c("raw", "discard", "keep", "discard", "keep")
links_neg$group <- c("discard", "keep", "discard", "keep")

color_neg <- 'd3.scaleOrdinal()
  .domain(["keep", "discard", "raw"])
  .range(["#3498db", "#e0e0e0", "#cccccc"]);'

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
  "NetMHCpan" = 250570,
  "IEDB" = 7300,
  "NetMHCpan&IEDB" = 23552
)

fit <- euler(venn_data)

venn_plot <- plot(
  fit,
  quantities = list(
    cex = 1.3,           
    font = 2             
  ),
  labels = FALSE,  # Hides the inline text to prevent overlapping/clutter
  legend = list(
    labels = c("NetMHCpan Binders\n(<2% Rank)", "IEDB Confirmed\n(Positives)"),
    cex = 1.1,
    side = "right"   # <--- Changed to right
  ),
  fills = list(
    fill = c("#3498db", "#2ecc71"), 
    alpha = 0.7                     
  ),
  edges = list(
    col = "black",       
    lwd = 2              
  ),
  main = "Predicted vs. Biologically Presented 9-mers"
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
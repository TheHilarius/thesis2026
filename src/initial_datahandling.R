# ----------------------------
# Setup
# ----------------------------
#setwd("//wsl$/Ubuntu/home/hilarius/special_course_spring2026")
library(tidyverse)
library(patchwork)
library(corrplot)
library(ggseqlogo)
library(Peptides)
library(factoextra)

current_user <- Sys.info()[["user"]]

if (current_user == "olive") {
  setwd("C:/Users/olive/Documents/R/special_course_spring2026")
} else if (current_user == "mj607") {
  setwd("//wsl$/Ubuntu/home/hilarius/special_course_spring2026")
} else {
  stop("Unknown user. Please set working directory manually.")
}

df_table4_raw <- read_csv("data/arrson_2009_initial.csv")

df_assarsson_raw <- read_csv("data/epitope_table_export_1770817956.csv")

df_assarsson_raw_clean <- df_assarsson_raw |>
  select(where(~ !all(is.na(.)))) |>
  select(-c("Epitope ID - IEDB IRI",  "Epitope - Source Molecule IRI", "Epitope - Source Organism IRI", "Epitope - Species IRI", "Epitope - Source Organism", "Epitope - Species", "Epitope - Object Type"))

colnames(df_assarsson_raw_clean) <- sub("^Epitope - ", "", colnames(df_assarsson_raw_clean))

df_merged <- df_table4_raw |> 
  left_join(df_assarsson_raw_clean, by =c("Sequence" = "Name")) |>
  mutate(across(
    everything(),
    as.character
  ))

df_merged <- df_merged |>
  # Capture reference e and f in "notes" column
  mutate(
    notes = case_when(
      str_detect(Infection, "[ef]$") ~ str_extract(Infection, "[ef]$"),
      str_detect(Immunogenic, "[ef]$") ~ str_extract(Immunogenic, "[ef]$"),
      TRUE ~ NA_character_
    )
  ) |>
  # Remove references
  mutate(
    Infection   = str_remove(Infection, "[ef]$"),
    Immunogenic = str_remove(Immunogenic, "[ef]$")
  ) |>
  # Turn "?" into NA
  mutate(
    across(everything(), ~ na_if(.x, "?"))
  ) |>
  mutate(
    Infection = as.numeric(Infection),
    Immunogenic = as.numeric(Immunogenic),
    `Starting Position` = as.integer(`Starting Position`),
    `Ending Position` = as.integer(`Ending Position`)
  ) |>
  mutate(Classification = as.factor(Classification))

df_merged <- df_merged |>
  mutate(
    # Replace "<1" with 0.1
    Affinity = case_when(
      Affinity == "<1" ~ "0.1",
      TRUE ~ Affinity
    )
  ) |>
  mutate(
    Affinity = as.numeric(Affinity)
  ) |>
  rename_with(tolower) |>
  rename(
    start_pos = `starting position`,
    end_pos = `ending position`,
    mol_source = `source molecule`,
    mol_parent = `molecule parent`,
    mol_parent_iri = `molecule parent iri`,
    orf = `orf name`,
  ) |>
  mutate(
    processing_disclosed = if_else(processed == "ND", "ND", "Disclosed"),
    processed = na_if(processed, "ND"),
    processed = as.numeric(processed),
    pep_length = nchar(sequence)
  ) |>
  relocate(processing_disclosed, .after = processed) |>
  mutate(
    bio_meaning = case_when(
      classification == "Dominant"    ~ "Passes ALL filters → recognized during infection",
      classification == "Subdominant" ~ "Immunogenic but NOT recognized during infection",
      classification == "Cryptic"     ~ "Binds MHC but fails immunogenicity/processing",
      classification == "Negative"    ~ "Fails at binding or immunogenicity"
    ),
    # Simplified: was it naturally processed?
    naturally_processed = case_when(
      classification == "Dominant"    ~ "Yes",
      classification == "Subdominant" ~ "Partially",
      classification == "Cryptic"     ~ "No",
      classification == "Negative"    ~ "No"
    )
  ) |>
  mutate(
    dc_processed = if_else(
      classification %in% c("Dominant", "Subdominant"),
      "Processed by DC", "Not processed"
    )
  )

colnames(df_assarsson_raw) <- sub("^Epitope - ", "", colnames(df_assarsson_raw))
df_protein_info <- df_assarsson_raw |>
  rename_with(tolower) |>
  rename(
    mol_source = `source molecule`,
    mol_source_iri = `source molecule iri`,
    mol_parent = `molecule parent`,
    mol_parent_iri = `molecule parent iri`,
  ) |>
  select(c(mol_source,mol_source_iri,mol_parent,mol_parent_iri)) |>
  mutate(protein_id = str_extract(mol_parent_iri, "[^/]+$")) |>
  distinct(protein_id, .keep_all = TRUE)

write_csv(df_protein_info,"data/assarsson_protein_info.csv")

#df_merged |> summarize(unique_count = n_distinct(mol_parent))
#df_merged_assarsson72 |> summarize(unique_count = n_distinct(mol_parent))

write_csv(
  df_merged,
  "data/merged_assarsson_data.csv"
)

df_notes <- tibble(
  letter = c("e", "f", "affinity_transform"),
  column_relevance = c("Immunogenic", "Infection", "Affinity"),
  description = c(
    "Positive in two of three experiments.",
    "Peptide responses are either not statistically significant or restricted by mouse MHC class I.",
    "Values reported as '<1' in the original dataset were recoded to 0.1 for numerical analysis."
  )
)
df_merged_assarsson72 <- df_merged |>
  filter(!is.na(start_pos) & !is.na(end_pos))

df_merged_leftover <-df_merged |>
  filter(is.na(start_pos) & is.na(end_pos))


# --- 1.1 Classification Distribution: Full vs Mapped vs Leftover ---
p_class_full <- ggplot(df_merged, aes(x = classification, fill = classification)) +
  geom_bar() +
  labs(title = "All peptides", x = NULL, y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

p_class_72 <- ggplot(df_merged_assarsson72, aes(x = classification, fill = classification)) +
  geom_bar() +
  labs(title = "Mapped peptides (n=72)", x = NULL, y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

p_class_leftover <- ggplot(df_merged_leftover, aes(x = classification, fill = classification)) +
  geom_bar() +
  labs(title = "Unmapped", x = NULL, y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

p_class_full + p_class_72 + p_class_leftover +
  plot_annotation(
    title = "Classification Distribution",
    subtitle = "No Dominant/Subdominant peptides lost in unmapped subset"
  )


# --- 1.2 Peptide Length ---
ggplot(df_merged_assarsson72, aes(x = as.factor(pep_length), fill = classification)) +
  geom_bar(position = "dodge") +
  labs(
    title = "Peptide Length Distribution",
    subtitle = "MHC-I typically presents 9-11 mers; all peptides are 9 or 10-mers",
    x = "Peptide Length (aa)",
    y = "Count"
  ) +
  theme_minimal()

# --- 1.3 Assarsson Epitope Selection Funnel ---
df_funnel <- tibble(
  step = factor(c(
    "1. All 9/10-mers\n(~70,000)",
    "2. High-affinity\nbinders (2.5%)",
    "3. Immunogenic\n(56% of binders)",
    "4. Naturally\nprocessed (15%)",
    "5. Dominant during\ninfection (1/10)"
  ), levels = c(
    "1. All 9/10-mers\n(~70,000)",
    "2. High-affinity\nbinders (2.5%)",
    "3. Immunogenic\n(56% of binders)",
    "4. Naturally\nprocessed (15%)",
    "5. Dominant during\ninfection (1/10)"
  )),
  count = c(70000, 1750, 980, 141, 15),
  label = c("70,000", "~1,750", "~980", "~141", "15")
)

ggplot(df_funnel, aes(x = step, y = count, fill = step)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = label), vjust = -0.5, size = 4) +
  scale_y_log10() +
  labs(
    title = "The Epitope Selection Funnel (Assarsson et al. 2008)",
    subtitle = "Project focus: Step 4 — predicting antigen processing/cleavage",
    x = NULL,
    y = "Number of peptides (log scale)"
  ) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(size = 9))

# ============================================================
# SECTION 2: Key Relationships
# ============================================================

# --- 2.1 Affinity by Classification ---
ggplot(df_merged_assarsson72, aes(x = classification, y = affinity, fill = classification)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 1.5) +
  scale_y_log10() +
  labs(
    title = "Binding Affinity by Classification",
    subtitle = "Strong binding alone does not predict immunodominance",
    x = "Classification",
    y = "Affinity (nM, log scale)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

df_merged_assarsson72 |>
  filter(affinity != 0.1) |>
  ggplot(aes(x = classification, y = affinity, fill = classification)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 1.5) +
  scale_y_log10() +
  labs(
    title = "Binding Affinity by Classification (without <1)",
    subtitle = "Strong binding alone does not predict immunodominance",
    x = "Classification",
    y = "Affinity (nM, log scale)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# --- 2.2 Affinity vs Immunogenicity ---
df_merged_assarsson72 |>
  filter(affinity != 0.1) |>
  ggplot(aes(x = affinity, y = immunogenic, color = classification)) +
  geom_point(alpha = 0.7, size = 2) +
  geom_smooth(
    method = "lm", se = TRUE, color = "grey30", linetype = "dashed",
    inherit.aes = FALSE,
    aes(x = affinity, y = immunogenic)
  ) +
  scale_x_log10() +
  labs(
    title = "Binding Affinity vs Immunogenicity",
    subtitle = "Spearman ρ = -0.38 — affinity is a weak predictor alone",
    x = "Affinity (nM, log scale)",
    y = "Immunogenic Response",
    color = "Classification"
  ) +
  theme_minimal()

# --- 2.3 Processing vs Immunogenicity ---
ggplot(df_merged_assarsson72 |> filter(!is.na(processed)),
       aes(x = processed, y = immunogenic, color = classification)) +
  geom_point(alpha = 0.7, size = 2) +
  geom_smooth(
    method = "lm", se = TRUE, color = "grey30", linetype = "dashed",
    inherit.aes = FALSE,
    aes(x = processed, y = immunogenic)
  ) +
  labs(
    title = "Antigen Processing vs Immunogenicity",
    subtitle = "Spearman ρ = 0.45 — processing is more informative than affinity",
    x = "Processing Score",
    y = "Immunogenic Response",
    color = "Classification"
  ) +
  theme_minimal()

# --- 2.4 Correlation Matrix ---
cor_matrix <- df_merged_assarsson72 |>
  select(affinity, infection, immunogenic, processed, pep_length) |>
  drop_na() |>
  cor(method = "spearman")

corrplot(cor_matrix,
         method = "color",
         type = "upper",
         addCoef.col = "black",
         tl.col = "black",
         title = "Spearman Correlation Between Numeric Features",
         mar = c(0, 0, 2, 0))





# ============================================================
# SECTION 3: DC Processing — The Core Question
# ============================================================

# --- 3.1 Affinity: Processed vs Not Processed ---
p_dc_aff <- ggplot(df_merged_assarsson72,
                   aes(x = dc_processed, y = affinity, fill = dc_processed)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 1.5) +
  scale_y_log10() +
  labs(
    title = "Affinity",
    subtitle = "Many strong binders are NOT processed",
    x = NULL, y = "Affinity (nM, log scale)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# --- 3.2 Processing Score: Processed vs Not ---
p_dc_proc <- df_merged_assarsson72 |>
  filter(!is.na(processed)) |>
  ggplot(aes(x = dc_processed, y = processed, fill = dc_processed)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 1.5) +
  labs(
    title = "Processing Score",
    subtitle = "Higher in truly processed peptides",
    x = NULL, y = "Processing Score"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

p_dc_aff + p_dc_proc +
  plot_annotation(
    title = "What distinguishes DC-processed peptides?",
    subtitle = "Processed = Dominant + Subdominant | Not processed = Cryptic + Negative"
  )

# --- 3.3 Dominant vs Subdominant ---
df_processed <- df_merged_assarsson72 |>
  filter(classification %in% c("Dominant", "Subdominant"))

p_dom_aff <- ggplot(df_processed, aes(x = classification, y = affinity, fill = classification)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  scale_y_log10() +
  labs(title = "Affinity", y = "Affinity (nM)") +
  theme_minimal() + theme(legend.position = "none")

p_dom_proc <- df_processed |>
  filter(!is.na(processed)) |>
  ggplot(aes(x = classification, y = processed, fill = classification)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  labs(title = "Processing Score", y = "Processing Score") +
  theme_minimal() + theme(legend.position = "none")

p_dom_imm <- df_processed |>
  filter(!is.na(immunogenic)) |>
  ggplot(aes(x = classification, y = immunogenic, fill = classification)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  labs(title = "Immunogenicity", y = "Immunogenic Response") +
  theme_minimal() + theme(legend.position = "none")

p_dom_aff + p_dom_proc + p_dom_imm +
  plot_annotation(
    title = "What separates Dominant from Subdominant epitopes?",
    subtitle = "Assarsson: 'other mechanisms limit the repertoire by a factor of 10'"
  )

# --- 3.4 Multi-Variable Bubble Plot ---
df_merged_assarsson72 |>
  filter(!is.na(processed) & !is.na(immunogenic)) |>
  ggplot(aes(x = affinity, y = processed,
             size = immunogenic, color = classification,
             shape = classification)) +
  geom_point(alpha = 0.7) +
  scale_x_log10() +
  labs(
    title = "Three Variables That Determine Epitope Fate",
    subtitle = "Affinity × Processing × Immunogenicity",
    x = "Binding Affinity (nM, log scale)",
    y = "Processing Score",
    size = "Immunogenic\nResponse",
    color = "Classification",
    shape = "Classification"
  ) +
  theme_minimal()

# ============================================================
# SECTION 4: Sequence Analysis
# ============================================================

# --- 4.1 Anchor Residues ---
df_anchors <- df_merged_assarsson72 |>
  mutate(
    p2 = substr(sequence, 2, 2),
    p_omega = substr(sequence, nchar(sequence), nchar(sequence))
  )

p_anchor2 <- ggplot(df_anchors, aes(x = p2, fill = classification)) +
  geom_bar(position = "dodge") +
  labs(
    title = "Position 2 Anchor Residue",
    subtitle = "MHC-I (HLA-A*0201) binding anchor",
    x = "Amino Acid", y = "Count"
  ) +
  theme_minimal()

p_anchor_c <- ggplot(df_anchors, aes(x = p_omega, fill = classification)) +
  geom_bar(position = "dodge") +
  labs(
    title = "C-terminal Anchor Residue",
    subtitle = "Determined by proteasomal cleavage",
    x = "Amino Acid", y = "Count"
  ) +
  theme_minimal()

p_anchor2 / p_anchor_c +
  plot_annotation(
    title = "MHC-I Anchor Positions",
    subtitle = "C-terminal residue is directly relevant to cleavage prediction"
  )

# --- 4.2 Sequence Logos by Classification (9-mers) ---
seq_9mers <- df_merged_assarsson72 |>
  filter(pep_length == 9) |>
  group_by(classification) |>
  filter(n() >= 3) |>
  summarise(seqs = list(sequence), .groups = "drop")

seq_list <- setNames(seq_9mers$seqs, seq_9mers$classification)

ggseqlogo(seq_list, ncol = 2) +
  plot_annotation(
    title = "Sequence Logos by Classification (9-mers)",
    subtitle = "Position 2 & 9 = anchor residues | Differences may reflect processing preferences"
  )

# ============================================================
# SECTION 5: Clustering
# ============================================================

# --- 5.1 Physicochemical Feature Calculation ---
df_features <- df_merged_assarsson72 |>
  mutate(
    hydrophobicity = sapply(sequence, hydrophobicity, scale = "KyteDoolittle"),
    charge = sapply(sequence, charge, pH = 7.4),
    mw = sapply(sequence, mw),
    pI = sapply(sequence, pI),
    aliphatic_index = sapply(sequence, aIndex),
    boman_index = sapply(sequence, boman)
  )

feature_matrix <- df_features |>
  select(hydrophobicity, charge, mw, pI, aliphatic_index,
         boman_index, affinity, pep_length) |>
  as.matrix()
rownames(feature_matrix) <- df_features$sequence

# --- 5.2 PCA Biplot ---
pca_res <- prcomp(scale(feature_matrix))

fviz_pca_biplot(pca_res,
                geom = "point",
                habillage = df_features$classification,
                addEllipses = TRUE,
                palette = "jco",
                pointsize = 2) +
  labs(
    title = "PCA: Physicochemical Properties + Affinity",
    subtitle = "Do processed epitopes cluster separately? Arrows show feature contributions"
  )

# --- 5.3 Optimal Clusters ---
fviz_nbclust(scale(feature_matrix), kmeans, method = "silhouette") +
  labs(title = "Optimal Number of Clusters (Physicochemical Features)")

# --- 5.4 K-Means ---
set.seed(42)
km_res <- kmeans(scale(feature_matrix), centers = 3, nstart = 25)

fviz_cluster(km_res, data = scale(feature_matrix),
             palette = "jco",
             ggtheme = theme_minimal()) +
  labs(title = "K-Means Clustering (Physicochemical Features)")

# Compare clusters to classification
cluster_vs_class <- table(
  Cluster = km_res$cluster,
  Classification = df_merged_assarsson72$classification
)
print(cluster_vs_class)

# --- 5.5 Cluster vs Classification Heatmap ---
cluster_vs_class |>
  as.data.frame() |>
  ggplot(aes(x = Classification, y = as.factor(Cluster), fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_viridis_c() +
  labs(
    title = "K-Means Clusters vs Immunological Classification",
    subtitle = "Do unsupervised clusters align with known categories?",
    x = "Classification",
    y = "Cluster",
    fill = "Count"
  ) +
  theme_minimal()

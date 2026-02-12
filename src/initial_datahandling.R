setwd("//wsl$/Ubuntu/home/hilarius/special_course_spring2026")
library(tidyverse)
# Data path will change from /Documents, once we have made a github repo

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



# Plots
ggplot(df_merged, aes(x = Affinity)) +
  geom_histogram(bins = 30)

ggplot(df_merged, aes(x = Infection, y = Immunogenic)) +
  geom_point() +
  geom_smooth(method = "lm")

ggplot(df_merged, aes(x = Affinity, y = Immunogenic)) +
  geom_point() +
  geom_smooth(method = "lm")

ggplot(df_merged, aes(Classification)) +
  geom_bar()

df_merged |>
  count(`ORF Name`) |>
  arrange(desc(n))

ggplot(df_merged |> count(`ORF Name`),
       aes(x = reorder(`ORF Name`, n), y = n)) +
  geom_col() +
  coord_flip()






# sub("^Epitope - ", "", colnames(df_assarsson_raw_clean))

write_csv(
  df_merged,
  "data/merged_assarsson_data.csv"
)


#columns to remove: "Epitope ID - IEDB IRI"  "Epitope - Source Molecule IRI" "Epitope - Source Organism IRI" "Epitope - Species IRI" "Epitope - Source Organism" "Epitope - Species"

# make a small dataframe with the links and information from: "Epitope - Source Organism IRI" "Epitope - Species IRI" "Epitope - Source Organism" "Epitope - Species"
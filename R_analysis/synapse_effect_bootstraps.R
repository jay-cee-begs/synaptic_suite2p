library(dplyr)
library(readr)
library(janitor)
library(ggplot2)
library(future.apply)

    
bootstrap_analysis_vectorized <- function(
    csv_file,
    param,
    test_groups = NULL,
    classification_filter = NULL,
    reference_group,
    n_boot = 10000,
    y_min,
    y_max,
    file_base = NULL
) {
  # Load and filter data
  data <- read.csv(csv_file) %>%
    clean_names() %>%
    filter(is.null(test_groups) | experimental_group %in% test_groups) %>%
    mutate(experimental_group = relevel(factor(experimental_group), ref = reference_group))
  
  if (!is.null(classification_filter)) {
    data <- filter(data, classification == classification_filter)
  }
  
  groups <- levels(data$experimental_group)
  n_groups <- length(groups)
  
  #  Precompute row indices per file per group
  file_groups <- data %>%
    group_by(experimental_group, file_name) %>%
    group_split(.keep = TRUE)
  
  files_by_group <- split(seq_along(file_groups), sapply(file_groups, function(df) as.character(df$experimental_group[1])))
  
  rows_by_file <- lapply(file_groups, function(df) which(data$file_name %in% df$file_name & data$experimental_group == df$experimental_group[1]))
  
  #  Preallocate matrices
  boot_control_means <- numeric(n_boot)
  boot_contrasts <- matrix(NA_real_, nrow = n_boot, ncol = n_groups - 1)
  colnames(boot_contrasts) <- groups[groups != reference_group]
  
  #  Bootstrap iterations
  for (i in seq_len(n_boot)) {
    boot_rows <- unlist(lapply(groups, function(g) {
      file_ids <- files_by_group[[g]]
      sampled_file_ids <- sample(file_ids, length(file_ids), replace = TRUE)
      unlist(rows_by_file[sampled_file_ids])
    }))
    
    boot_vals <- data[[param]][boot_rows]
    boot_groups <- as.character(data$experimental_group[boot_rows])  # ⚠ convert factor -> character
    
    # Safe group means using dplyr
    boot_dat <- tibble(val = boot_vals, group = boot_groups)
    group_means <- boot_dat %>%
      group_by(group) %>%
      summarise(mean_val = mean(val, na.rm = TRUE), .groups = "drop")
    
    ref_mean <- group_means %>% filter(group == reference_group) %>% pull(mean_val)
    boot_control_means[i] <- ref_mean
    
    # Fill contrasts safely
    contrasts <- setNames(rep(NA_real_, n_groups - 1), groups[groups != reference_group])
    for (j in seq_len(nrow(group_means))) {
      g <- group_means$group[j]
      if (g != reference_group){
        percent_change <- (group_means$mean_val[j] - ref_mean) / ref_mean * 100
        contrasts[g] <- 1 + percent_change / 100
      }
    }
    boot_contrasts[i, ] <- contrasts
  }
  
  # Convert to data.frame
  boot_df <- as.data.frame(boot_contrasts)
  
  #  Summaries
  contrast_summary <- boot_df %>%
    pivot_longer(cols = everything(), names_to = "group", values_to = "percent_diff") %>%
    group_by(group) %>%
    summarise(
      estimate = median(percent_diff, na.rm = TRUE),
      ci_low = quantile(percent_diff, 0.025, na.rm = TRUE),
      ci_high = quantile(percent_diff, 0.975, na.rm = TRUE),
      estimate_raw = NA,
      ci_low_raw = NA,
      ci_high_raw = NA,
      .groups = "drop"
    ) %>%
    mutate(significant = ci_low > 0 | ci_high < 0)
  
  control_summary <- tibble(
    group = reference_group,
    estimate = 1,   # reference = 1 on odds-ratio scale
    ci_low = 1,
    ci_high = 1,
    estimate_raw = median(boot_control_means, na.rm = TRUE),
    ci_low_raw = quantile(boot_control_means, 0.025, na.rm = TRUE),
    ci_high_raw = quantile(boot_control_means, 0.975, na.rm = TRUE),
    significant = NA
  )
  
  boot_summary_all <- bind_rows(control_summary, contrast_summary)
  
  #  Write outputs
  if (is.null(file_base)) file_base <- tools::file_path_sans_ext(basename(csv_file))
  output_prefix <- paste0("bootstraps/", file_base, "_", param)
  write_csv(boot_df, paste0(output_prefix, "_bootstrap_contrasts.csv"))
  write_csv(boot_summary_all, paste0(output_prefix, "_summary_bootstrap_results.csv"))
  
  #  Plot

  boot_forest_plot <- ggplot(contrast_summary, aes(x = group, y = estimate)) +
    geom_point(size = 4, color = "black", shape = 15) +
    geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.25, linewidth = 1.2) +
    geom_hline(yintercept = 1, linetype = 'dashed', linewidth = 1) +  # reference = 1
    coord_flip() +
    scale_y_continuous(
      trans = 'log10',          # log scale is typical for ratios
      limits = c(y_min, y_max),
      breaks = scales::trans_breaks("log10", function(x) 10^x),
      labels = scales::number_format(accuracy = 0.01)
    ) +
    theme_minimal() +
    labs(
      title = paste("Bootstrap Estimate for", param, "Relative to", reference_group),
      y = paste(param, "Bootstrap Ratio (deviation from 1)"),
      x = "Experimental Group"
    )
  
  ggsave(paste0(output_prefix, "bootstrap_plot.png"), plot = boot_forest_plot, width = 8, height = 6, dpi = 300, device = 'png')
  ggsave(paste0(output_prefix, "bootstrap_plot.svg"), plot = boot_forest_plot, dpi = 300, device = 'svg')
  
      
      return(list(
        contrasts = boot_df,
        control_means = boot_control_means,
        summary = boot_summary_all
      ))


# ----Example Workflow for a experiment csv file---- #


# data_dir <- "C:/Users/username/Documents/csv_file_folder"
# if (wd != data_dir) {
#   setwd(data_dir)
#   wd <- getwd()
# }
# source("glmm_effect_functions.R")

#What parameters do you wish to check
# param_list <- c("spikes_freq", 'avg_amplitude')

#Option to look at synaptic, dendritic, or all events
# classifications <- list("synaptic_event","dendritic_event", NULL)

# for (param in param_list){
#   for (classification in classifications){
#     bootstrap_analysis_vectorized("example-imaging_experiment_summary.csv",
#                     param,
#                     test_groups = c("group1", "group2", "group3", "group4"),
#                     classification_filter = classification,
#                     reference_group = 'group1',
#                     y_min= 0.6, y_max = 2.5,
#                     file_base = "experiment_name")
#   }
# }

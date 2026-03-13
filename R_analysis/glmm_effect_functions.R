library(glmmTMB)
library(dplyr)
library(readr)
library(janitor)
library(ggplot2)
library(DHARMa)
library(scales)
library(svglite)
library(performance)
library(sjPlot)
library(broom)
library(broom.mixed)
library(tidyr)
library(dplyr)
library(JWileymisc)
library(fitdistrplus)


analyze_model <- function(csv_file, reference_group, outcome_var, test_groups = NULL, pool_baseline = FALSE, filter_extreme_values = FALSE, multiple_replicates = FALSE, classification_filter = NULL, file_base = NULL, use_forest_lim = TRUE, ymin = 0.10, ymax = 3.0) {
  graphics.off()
  
  # Load and clean data
  data <- read_csv(csv_file, show_col_types = FALSE) %>%
    clean_names() %>%
    filter(is.null(test_groups) | experimental_group %in% test_groups)
  
  #Pool baseline to group all baseline files with "base" string in the file name
  if (pool_baseline){
    data <- data %>% 
      mutate(experimental_group_combined = ifelse(
        grepl("base", experimental_group),
        "baseline",
        as.character(experimental_group)
      ),
      experimental_group_combined = relevel(factor(experimental_group_combined), ref = "baseline")
      )
    group_var <- "experimental_group_combined"
  } else {
    data <- data %>% 
      mutate(
        experimental_group = relevel(factor(experimental_group), ref = reference_group)
      )
    group_var <- "experimental_group"
  }
  data <- data %>% 
    mutate(
      replicate_no = factor(replicate_no),
      file_name = factor(file_name)
    )
  
  #load in file_base name based on arguments or default from csv file name
  cat("Successfully Loaded ", csv_file, "\n")
  if (!is.null(file_base)){
    file_base <- file_base
  } else {
    file_base <- tools::file_path_sans_ext(basename(csv_file))
  }
  
  output_prefix <- paste0("graphs/", file_base, "_", outcome_var)
  
  # Optionally filter by classification (e.g., "synaptic_event" or "dendritic_event")
  if (!is.null(classification_filter)) {
    if (!"classification" %in% colnames(data)) {
      stop("The 'classification' column is required for filtering but is missing.
           ... check the input csv or leave classification_filter==NULL")
    }
    data <- data %>% filter(classification == classification_filter)
    #Update output_prefix for files based on if classification filter is used
    output_prefix <- paste0(output_prefix, "_", classification_filter)
  }
  
  data <- data %>%
    drop_na(outcome_var) #TODO need to check this for decay time
  
  if (filter_extreme_values){
    fit_gamma <- fitdist(data[["spikes_freq"]], "gamma")
    shape_est <- fit_gamma$estimate["shape"]
    rate_est <- fit_gamma$estimate["rate"]
    lower_cut <- qgamma(0.001, shape = shape_est, rate = rate_est)
    upper_cut <- qgamma(0.999, shape = shape_est, rate = rate_est)
    below_idx <- which(data[["spikes_freq"]] < lower_cut)
    above_idx <- which(data[["spikes_freq"]] > upper_cut)
    
    # Create summary table
    removed_info <- data.frame(
      index       = c(below_idx, above_idx),
      value       = data[["spikes_freq"]][c(below_idx, above_idx)],
      cutoff_type = c(rep("lower", length(below_idx)), rep("upper", length(above_idx)))
    )
    # Save removed extreme values TODO readd
    # write_csv(
    #   removed_info,
    #   paste0(output_prefix, "_extreme_values.csv")
    # )
    
    # Print quick summary in console
    cat("Extreme value removal summary:\n")
    cat("  Below lower cutoff:", length(below_idx), "\n")
    cat("  Above upper cutoff:", length(above_idx), "\n")
    cat("  Total removed:", nrow(removed_info), "\n")
    cat("Percentage of ROIs removed:",
        sprintf("%.2f", (nrow(removed_info) / nrow(data)) * 100),
        "%\n")
    
    data <- subset(data, data[["spikes_freq"]] >= lower_cut & data[["spikes_freq"]] <= upper_cut)
  }

  if (multiple_replicates){
    formula <- as.formula(paste(outcome_var, "~", group_var, "+ (1| replicate_no / file_name)")) # Remove synapse_id as a tag (1| replicate_no / file_name)
    
  }
  else{
    formula <- as.formula(paste(outcome_var, "~", group_var, " + (1| file_name )")) # Remove synapse_id as a tag (1| replicate_no / file_name)

  }
  if (outcome_var == 'spikes_freq'){
    data <- filter(data, as.numeric(spikes_count) >= 2) 
      
    formula <- as.formula(paste(outcome_var, "~", group_var, " + (1| file_name )")) # Remove synapse_id as a tag (1| replicate_no / file_name)
    model <- glmmTMB(
      formula,
      data = data,
      family = Gamma(link = 'log'),
      # dispformula = as.formula(paste("~", group_var)),
      control = glmmTMBControl(
        optCtrl = list(iter.max = 2000, eval.max = 2000),
        profile = TRUE
      )
    )
  }
  if (outcome_var == 'avg_amplitude'){
    
    formula <- as.formula(paste0("log(",outcome_var, ") ~", group_var, "+ (1| file_name)")) # Remove synapse_id as a tag (1| replicate_no / file_name)
    
    model <- glmmTMB(
      formula,
      data = data,
      family = gaussian(),
      # dispformula = as.formula(paste("~", group_var)),
      control = glmmTMBControl(
        optCtrl = list(iter.max = 2000, eval.max = 2000),
        profile = TRUE
      )
    )
  }
  
  sim_res <- simulateResiduals(model, parallel = TRUE, ncores = 8)
  disp <- testDispersion(sim_res)
  out  <- testOutliers(sim_res)
  uni  <- testUniformity(sim_res)
  safe_scalar <- function(x) {
    if (is.null(x) || length(x) == 0) NA_real_ else as.numeric(x)
  }
  DHARMa_df <- data.frame(
    dispersion = safe_scalar(disp$dispersion),
    dispersion_p = safe_scalar(disp$p.value),
    outliers_observed = safe_scalar(out$observed),
    outliers_expected = safe_scalar(out$expected),
    outliers_p = safe_scalar(out$p.value),
    uniformity_D = safe_scalar(uni$statistic),
    uniformity_p = safe_scalar(uni$p.value)
  )
  
  fitted_vals <- fitted(model)
  residuals_vals <- residuals(model, type = "pearson") # standardized residuals
  res_df <- data.frame(
    fitted = fitted_vals,
    residuals = residuals_vals
  )
  residual_plot <- ggplot(res_df, aes(fitted, residuals)) +
    geom_point(alpha = 0.3) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    theme_minimal()


  ggsave(paste0(output_prefix, "_res_v_fitted_plot.png"), plot = residual_plot, width = 8, height = 6, 'png')
  ggsave(paste0(output_prefix, "_res_v_fitted_plot.svg"), plot = residual_plot, width = 8, height = 6, dpi = 300, 'svg')    
    
  cat("Model for", outcome_var, if (!is.null(classification_filter)) paste(classification_filter) else "", "has been generated!\n")
  
  # Extract estimates
  coefs <- as.data.frame(summary(model)$coefficients$cond)
  coefs$group <- rownames(coefs)
  est_df <- coefs %>%
    filter(group != "(Intercept)") %>%
    mutate(
      group = case_when(
        grepl("^experimental_group", group) ~ gsub("experimental_group", "", group),
        TRUE ~ group
      )
    )
  complete_df <- est_df
  # est_df <- est_df %>%
  #   mutate(
  #     OR = exp(Estimate),
  #     # percent_change = (exp(Estimate) - 1) * 100,
  #     lower = exp(Estimate - 1.96 * `Std. Error`),
  #     upper = exp(Estimate + 1.96 * `Std. Error`)
  #   )
  print(est_df)
  
  fixed_effects <- tidy(model, effects = "fixed", conf.int = TRUE)
  
  ci <- confint(model, parm="beta_", level=0.95, method="profile")
  ci_df <- as.data.frame(ci)
  ci_df$group <- rownames(ci_df)
  colnames(ci_df) <- c("lower", "upper", "group")
  
  
  # Merge with point estimates
  est_df <- coefs %>%
    left_join(ci_df, by="group") %>%
    mutate(
      OR = exp(Estimate),
      lower = exp(lower),
      upper = exp(upper),
      group = case_when(
        group == "(Intercept)" ~ "Baseline (Intercept)",
        grepl("^experimental_group", group) ~ gsub("experimental_group", "", group),
        TRUE ~ group
      )
    )
  # 4. Extract random effect variances and 95% CIs
  random_variances <- as.data.frame(VarCorr(model)$cond)
  random_ci <- confint(model, parm = "theta_", level = 0.95)
  
  random_ci <- exp(random_ci)  # Convert log-scale CIs to SD-scale
  random_ci_df <- as.data.frame(random_ci)
  random_ci_df$group <- rownames(random_ci_df)
  random_variances_long <- random_variances %>%
    pivot_longer(
      cols = starts_with("X.Intercept"),
      names_to = "var_col",
      values_to = "variance_est"
    )
  
  # combined_random_df <- bind_cols(random_variances_long, random_ci_df) %>%
  #   select(group, var_col, variance_est, `2.5 %`, `97.5 %`, Estimate)
  
  icc_df <- icc(model, by_group = TRUE) %>% as.data.frame()
  cat("Model statistics calculated\n")
  
  if (use_forest_lim) {
    y_min <- ymin
    y_max <- ymax
  }
  else{
    y_min <- 0.8
    y_max <- 1.5
    
  }
  
  forest_plot <- ggplot(est_df%>% filter(group != "Baseline (Intercept)"), aes(x = group, y = OR)) +
    geom_point(size = 4, color = "#000000", shape = 15, fill = "black") +
    geom_errorbar(aes(ymin = lower, ymax = upper),
                  width = 0.25, linewidth = 1.2, color = "#000000") +
    geom_hline(yintercept = 1, linetype = "dashed", color = "black", linewidth = 1) +
    coord_flip() +
    scale_y_continuous(
      trans = 'log10',
      limits = c(y_min, y_max),
      # breaks = c(0.25, 0.5, 1, 2, 4),
      breaks = scales::trans_breaks("log10", function(x) 10^x),
      labels = scales::number_format(accuracy = 0.01)
    ) +
    theme_set(theme_minimal()) +
    theme(
      axis.text.y = element_text(size = 14),
      axis.title = element_text(size = 16, face = "bold"),
      plot.title = element_text(size = 18, face = "bold"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank(),
      plot.margin = margin(5, 5, 5, 5),
      axis.ticks.y = element_blank(),
      legend.position = "none"
    ) +
    labs(
      title = paste("Odds Ratios for", outcome_var, "Relative to", reference_group),
      y = "Odds Ratio (log scale)",
      x = "Experimental Group"
    )
  #print(forest_plot)
  
  ggsave(paste0(output_prefix, "_forest_plot.png"), plot = forest_plot, width = 8, height = 6, dpi = 300, 'png')
  ggsave(paste0(output_prefix, "_forest_plot.svg"), plot = forest_plot, dpi = 300, 'svg')
  # Residual diagnostics

  # QQ plot
  qqdata <- data.frame(
    expected = sort(runif(length(sim_res$scaledResiduals))),
    observed = sort(sim_res$scaledResiduals)
  )
  
  qq_plot <- ggplot(qqdata, aes(x = expected, y = observed)) + 
    geom_point(alpha = 0.6, size = 4) + 
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = 'solid', linewidth = 2) + 
    # labs(x = "Expected Uniform Quantiles", y = "Observed Scaled Residuals") + 
    theme_minimal(base_size = 32) + 
    theme(
      panel.border = element_blank(),
      panel.grid = element_blank(),
      axis.line.x = element_line(size = 0.5, color = "black", linetype = "solid"),
      axis.line.y = element_line(size = 0.5, color = "black", linetype = "solid"),
      # axis.tick.length = unit(5, "pt"),
      axis.ticks = element_line(color = 'black'),
      
      # Customize text sizes
      axis.text = element_text(size = 28),
      axis.title = element_text(size = 32, face = "bold"),
      plot.title = element_text(size = 36, face = "bold"),
      
      # Margin around plot
      plot.margin = margin(10, 10, 10, 10)
    )
  
  ggsave(paste0(output_prefix, "_qq_plot.png"), plot = qq_plot, width = 8, height = 6, dpi = 300, 'png')
  ggsave(paste0(output_prefix, "_qq_plot.svg"), plot = qq_plot, dpi = 300, "svg")
  #print(forest_plot)  
  model_aic <- AIC(model)
  model_r2 <- tryCatch({
    r2(model)
  }, error = function(e) {
    warning("Could not calculate R2: ", e$message)
    return(data.frame(R2_marginal = NA, R2_conditional = NA))
  })  
  print(model_r2)
  metrics_df <- data.frame(
    AIC = model_aic,
    R2_marginal = model_r2$R2_marginal,
    R2_conditional = model_r2$R2_conditional
  )
  write.csv(metrics_df, paste0(output_prefix, "_model_metrics.csv"), row.names = FALSE)
  write_csv(est_df, paste0(output_prefix, "_OddRatio_change_CI.csv"))
  write_csv(icc_df, paste0(output_prefix,"icc_values.csv"))
  write_csv(DHARMa_df, paste0(output_prefix, "DHARMa_analysis.csv"))
  cat("All plots and metrics saved.\nModel summary:\n")
  model_summary <- summary(model)
  capture.output(summary(model))
  print(metrics_df)
  library(openxlsx)
  
  # Create a workbook
  wb <- createWorkbook()
  
  # Add baseline estimates
  addWorksheet(wb, "Base_model")
  writeData(wb, "Base_model", model$frame)
  
  # Add percent change estimates
  addWorksheet(wb, "Percent Change")
  writeData(wb, "Percent Change", est_df)
  
  # Add fixed effects (raw log-scale estimates + CI)
  addWorksheet(wb, "Fixed Effects")
  writeData(wb, "Fixed Effects", fixed_effects)
  
  # Add ICC values
  addWorksheet(wb, "ICC")
  writeData(wb, "ICC", icc_df)
  
  # Add model metrics (AIC, R2)
  addWorksheet(wb, "Model Metrics")
  writeData(wb, "Model Metrics", metrics_df)
  
  # Add DHARMa diagnostics
  addWorksheet(wb, "DHARMa dx")
  writeData(wb, "DHARMa dx", DHARMa_df)
  
  # Save workbook
  saveWorkbook(wb, paste0(output_prefix, "_all_results.xlsx"), overwrite = TRUE)
}

# ----Example Workflow for a experiment csv file---- #
# data_dir <- "C:/Users/user_name/Documents/r_analysis"

# outcome_vars <- c("spikes_freq","avg_amplitude")

# for (outcome in outcome_vars){
#   analyze_model("example.csv", multiple_replicates = FALSE, outcome = outcome, pool_baseline = FALSE, reference_group = "MGO_base", test_groups = c('MGO_base','MGO_tx',"003_tx",'003_base'), classification_filter = 'synaptic_event', file_base = "260302_NMDAR_AB_test", ymin = 0.5, ymax = 1.5)
# }

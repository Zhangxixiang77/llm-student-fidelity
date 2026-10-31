#!/usr/bin/env Rscript
# Independent GDINA-package cross-check for the real-data CDM comparison.
# This script has not been executed in the development environment (R unavailable).
# It performs no LLM calls and writes only derived local summaries.

args <- commandArgs(trailingOnly = TRUE)
usage <- paste(
  "Usage: Rscript scripts/r_cdm_crosscheck.R",
  "--dataset data/processed/frcsub --output runs/r_cdm_crosscheck",
  "[--splits 20] [--seed 42] [--dina-starts 10] [--max-iter 300]",
  "[--split-indices path/to/split_indices.csv]", sep = "\n"
)
get_arg <- function(name, default = NULL) {
  index <- match(name, args)
  if (is.na(index)) return(default)
  if (index == length(args)) stop("Missing value for ", name)
  args[index + 1L]
}
if ("--help" %in% args) {
  cat(usage, "\n")
  quit(save = "no", status = 0L)
}
if (!requireNamespace("GDINA", quietly = TRUE)) {
  stop("Install the GDINA package first: install.packages('GDINA')")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Install jsonlite first: install.packages('jsonlite')")
}

dataset_dir <- get_arg("--dataset")
output_dir <- get_arg("--output")
if (is.null(dataset_dir) || is.null(output_dir)) stop(usage)
n_splits <- as.integer(get_arg("--splits", "20"))
seed <- as.integer(get_arg("--seed", "42"))
dina_starts <- as.integer(get_arg("--dina-starts", "10"))
max_iter <- as.integer(get_arg("--max-iter", "300"))
split_indices_path <- get_arg("--split-indices", NA_character_)
stopifnot(n_splits >= 1L, dina_starts >= 1L, max_iter >= 1L)

responses <- read.csv(
  file.path(dataset_dir, "responses.csv"), stringsAsFactors = FALSE,
  check.names = FALSE
)
q_matrix <- read.csv(
  file.path(dataset_dir, "q_matrix.csv"), stringsAsFactors = FALSE,
  check.names = FALSE
)
stopifnot(nrow(responses) == nrow(q_matrix))
stopifnot(identical(responses$item_id, q_matrix$item_id))
attribute_names <- setdiff(names(q_matrix), "item_id")
response_names <- setdiff(names(responses), "item_id")
dat <- t(as.matrix(responses[, response_names, drop = FALSE]))
storage.mode(dat) <- "integer"
Q <- as.matrix(q_matrix[, attribute_names, drop = FALSE])
storage.mode(Q) <- "integer"
stopifnot(all(dat %in% c(0L, 1L)), all(Q %in% c(0L, 1L)))

# Optional columns: split_index (0-based), role (fit/holdout), and either
# student_index (0-based) or student_id. This allows exact Python split reuse.
# Without this file, R uses its own seeded permutations; those are not claimed
# to reproduce NumPy's random-number stream.
external_splits <- NULL
if (!is.na(split_indices_path)) {
  external_splits <- read.csv(split_indices_path, stringsAsFactors = FALSE)
  required <- c("split_index", "role")
  stopifnot(all(required %in% names(external_splits)))
  stopifnot(any(c("student_index", "student_id") %in% names(external_splits)))
}

make_split <- function(split_index) {
  if (!is.null(external_splits)) {
    rows <- external_splits[external_splits$split_index == split_index, ]
    fit_rows <- rows[rows$role == "fit", ]
    holdout_rows <- rows[rows$role == "holdout", ]
    get_indices <- function(x) {
      if ("student_index" %in% names(x)) return(as.integer(x$student_index) + 1L)
      match(x$student_id, rownames(dat))
    }
    return(list(fit = get_indices(fit_rows), holdout = get_indices(holdout_rows)))
  }
  set.seed(seed + split_index)
  permutation <- sample.int(nrow(dat), replace = FALSE)
  midpoint <- floor(nrow(dat) / 2L)
  list(fit = permutation[seq_len(midpoint)],
       holdout = permutation[(midpoint + 1L):nrow(dat)])
}

heldout_mean_log_likelihood <- function(fit, holdout) {
  # Evaluate new students from full-class probabilities and the attribute prior,
  # rather than reusing their in-sample training posteriors.
  class_prob <- GDINA::extract(fit, "LCprob.parm")
  patterns <- GDINA::extract(fit, "attributepattern")
  prior <- as.numeric(GDINA::extract(fit, "att.prior"))
  if (is.list(class_prob)) stop("Expected a binary LC probability matrix")
  class_prob <- as.matrix(class_prob)
  if (nrow(class_prob) != nrow(patterns) && ncol(class_prob) == nrow(patterns)) {
    class_prob <- t(class_prob)
  }
  stopifnot(nrow(class_prob) == length(prior), ncol(class_prob) == ncol(holdout))
  stopifnot(all(class_prob > 0 & class_prob < 1), abs(sum(prior) - 1) < 1e-6)
  student_class_ll <- vapply(seq_len(nrow(class_prob)), function(class_index) {
    p <- class_prob[class_index, ]
    correct <- sweep(holdout, 2L, log(p), "*")
    incorrect <- sweep(1L - holdout, 2L, log1p(-p), "*")
    rowSums(correct + incorrect)
  }, numeric(nrow(holdout)))
  if (nrow(holdout) == 1L) student_class_ll <- matrix(student_class_ll, nrow = 1L)
  log_prior <- log(prior)
  individual_ll <- apply(student_class_ll, 1L, function(values) {
    terms <- values + log_prior
    maximum <- max(terms)
    maximum + log(sum(exp(terms - maximum)))
  })
  mean(individual_ll)
}

convergence_text <- function(fit) {
  value <- GDINA::extract(fit, "convergence")
  paste(capture.output(str(value)), collapse = " ")
}

records <- list()
record_index <- 1L
for (split_index in seq_len(n_splits) - 1L) {
  split <- make_split(split_index)
  fitting <- dat[split$fit, , drop = FALSE]
  holdout <- dat[split$holdout, , drop = FALSE]
  for (model in c("DINA", "logitGDINA")) {
    message("split=", split_index, " model=", model)
    fit <- GDINA::GDINA(
      dat = fitting, Q = Q, model = model, att.dist = "saturated",
      nstarts = if (model == "DINA") dina_starts else 3L,
      randomseed = seed + split_index + 500L,
      control = list(maxitr = max_iter), verbose = 0
    )
    records[[record_index]] <- data.frame(
      split_index = split_index, model = model,
      n_fit = nrow(fitting), n_holdout = nrow(holdout),
      training_log_likelihood = as.numeric(GDINA::extract(fit, "logLik")),
      holdout_mean_log_likelihood = heldout_mean_log_likelihood(fit, holdout),
      convergence = convergence_text(fit), stringsAsFactors = FALSE
    )
    record_index <- record_index + 1L
  }
}
results <- do.call(rbind, records)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(results, file.path(output_dir, "r_split_metrics.csv"), row.names = FALSE)
dina <- results[results$model == "DINA", ]
gdina <- results[results$model == "logitGDINA", ]
merged <- merge(dina, gdina, by = "split_index", suffixes = c("_dina", "_gdina"))
merged$gdina_minus_dina <- merged$holdout_mean_log_likelihood_gdina -
  merged$holdout_mean_log_likelihood_dina
summary <- list(
  status = "r_crosscheck_completed",
  package_version = as.character(utils::packageVersion("GDINA")),
  n_splits = n_splits, dina_n_starts = dina_starts,
  link = "logit", attribute_distribution = "saturated",
  exact_external_split_indices = !is.null(external_splits),
  n_splits_gdina_better = sum(merged$gdina_minus_dina > 0),
  mean_gdina_minus_dina = mean(merged$gdina_minus_dina),
  median_gdina_minus_dina = median(merged$gdina_minus_dina)
)
jsonlite::write_json(
  summary, file.path(output_dir, "r_summary.json"),
  auto_unbox = TRUE, pretty = TRUE, digits = NA
)
print(summary)

#!/usr/bin/env Rscript
# Plot training curves from Python-exported metrics (base R only).

cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
script_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg), winslash = "/"))
} else {
  normalizePath(getwd(), winslash = "/")
}
root <- normalizePath(file.path(script_dir, ".."), winslash = "/")

csv_path <- file.path(root, "outputs", "metrics.csv")
if (!file.exists(csv_path)) {
  stop("Missing ", csv_path, " — run train.py from the project root first.", call. = FALSE)
}

df <- read.csv(csv_path, stringsAsFactors = FALSE)
out_png <- file.path(root, "outputs", "training_curves.png")
png(out_png, width = 900, height = 420, res = 110)
par(mfrow = c(1, 2), mar = c(4, 4, 2, 1))
plot(
  df$epoch, df$train_loss,
  type = "o", pch = 16, col = "#2563eb",
  xlab = "Epoch", ylab = "Train MSE", main = "Denoising training loss"
)
grid()
plot(
  df$epoch, df$val_psnr,
  type = "o", pch = 16, col = "#16a34a",
  xlab = "Epoch", ylab = "PSNR (dB)", main = "Validation PSNR (vs clean)"
)
grid()
dev.off()
cat(sprintf("Wrote %s\n", normalizePath(out_png, winslash = "/")))

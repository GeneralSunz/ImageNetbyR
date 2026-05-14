library(keras3)
library(magick)
# ── 1. 加载图片 ──
image_names <- list.files("book_images", pattern = "\\.(jpg|jpeg|png|bmp)$",
                          ignore.case = TRUE)
num_images <- length(image_names)
x <- array(dim = c(num_images, 224, 224, 3))
for(i in seq_along(image_names)) {
  img <- image_read(file.path("book_images", image_names[i]))
  img <- image_resize(img, "224x224!")
  arr <- as.integer(img[[1]])           # [0, 255] integer
  x[i,,,] <- arr / 255                  # [0, 1] float
}
# MobileNetV2 预处理: [0,1] -> [-1,1]
x <- x * 2 - 1
# ── 2. 加载预训练模型 ──
model <- keras$applications$MobileNetV2(
  weights      = "imagenet",
  input_shape  = NULL,
  include_top  = TRUE,
  classes      = 1000L
)
# ── 3. 预测 ──
preds <- predict(model, x)
# ── 4. 解码结果 ──
for(i in seq_len(num_images)) {
  cat(sprintf("[%d/%d] %s\n", i, num_images, image_names[i]))
  top <- keras$applications$mobilenet_v2$decode_predictions(preds, top = 5L)[[i]]
  for(j in seq_along(top)) {
    cat(sprintf("  %2d. %s (%.1f%%)\n", j, top[[j]][[2]], top[[j]][[3]] * 100))
  }
  cat("\n")
}

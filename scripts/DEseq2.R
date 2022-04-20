library(DESeq2)
setwd("/Users/jendawk/Dropbox (MIT)/C Diff Recurrence Paper/Analyses/scripts/")
out_path <- "/Users/jendawk/Dropbox (MIT)/C Diff Recurrence Paper/Analyses/univariate_analysis_control/16s/deseq2/"

files_x <- list.files(path="inputs/DEseq2_covariate", pattern="counts*", full.names=TRUE, recursive=FALSE)
files_y <- list.files(path="inputs/DEseq2_covariate", pattern="targ*", full.names=TRUE, recursive=FALSE)
files_y2 <- list.files(path="inputs/DEseq2_covariate", pattern="drug*", full.names=TRUE, recursive=FALSE)

run_deseq2test <- function(otu_table, target){
  library(DESeq2, quietly = TRUE)
  
  # outcomedf <- data.frame(outcome = factor(outcome))
  row.names(target) <- colnames(otu_table)
  if ("drug" %in% colnames(target)){
    x <- DESeqDataSetFromMatrix(countData = otu_table, colData = target , design = ~ drug + outcome)
  } else {
    x <- DESeqDataSetFromMatrix(countData = otu_table, colData = target , design = ~ outcome)
  }
  gm_mean = function(x, na.rm=TRUE){
    exp(sum(log(x[x > 0]), na.rm=na.rm) / length(x))
  }
  geoMeans = apply(counts(x), 1, gm_mean)
  
  x = estimateSizeFactors(x, geoMeans = geoMeans)
  x <- DESeq(x)
  res <- results(x, contrast <- c("outcome", 1,0))
  # res2 <- results(x)
  # res$padj <- p.adjust(res$pvalue)
  
  output_df <- data.frame(OTU = row.names(res), pval = res$pvalue, 
                          padj = res$padj, log2fold = res$log2FoldChange)
  output_df 
}

for (i in seq(length(files_x))){
  fx = files_x[i]
  fy = files_y[i]
  fy2 = files_y2[i]
  x = read.csv(fx, row.names = 1)
  y = read.csv(fy, row.names =1)
  y2 = read.csv(fy2, row.names = 1)
  colnames(y) = 'outcome'
  colnames(y2) = 'drug'
  y$outcome = as.factor(y$outcome)
  if (grepl('0', fy2) & !grepl('ecurrer', fy2)){
    res <- run_deseq2test(t(x),y)
  } else {
    y2$drug = as.factor(y2$drug)
    target <- t(rbind(t(y), t(y2)))
    res <- run_deseq2test(t(x),target)
  }
  #dds <- DESeqDataSetFromMatrix(countData=t(x), colData = y, design = ~X0)
  #dds <- DESeq(dds)
  #res <- results(dds)
  num <- strsplit(strsplit(fx, "counts_")[[1]][-1], "\\.")[[1]][1]
  write.csv(res, paste(out_path, 'res', num, '.csv', sep=''))
}
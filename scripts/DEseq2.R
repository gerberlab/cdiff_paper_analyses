library(DESeq2)
setwd("/Users/jendawk/Dropbox (MIT)/C Diff Recurrence Paper/Analyses/scripts/")
out_path <- "/Users/jendawk/Dropbox (MIT)/C Diff Recurrence Paper/Analyses/univariate_analysis/16s/deseq2/"

files_x <- list.files(path="inputs/DEseq2", pattern="counts*", full.names=TRUE, recursive=FALSE)
files_y <- list.files(path="inputs/DEseq2", pattern="col*", full.names=TRUE, recursive=FALSE)

run_deseq2test <- function(otu_table, outcome){
  library(DESeq2, quietly = TRUE)
  
  # outcomedf <- data.frame(outcome = factor(outcome))
  row.names(outcome) <- colnames(otu_table)
  x <- DESeqDataSetFromMatrix(countData = otu_table, colData = outcome , design = ~ outcome)
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
  x = read.csv(fx, row.names = 1)
  y = read.csv(fy, row.names =1)
  colnames(y) = 'outcome'
  y$outcome = as.factor(y$outcome)
  res <- run_deseq2test(t(x),y)
  #dds <- DESeqDataSetFromMatrix(countData=t(x), colData = y, design = ~X0)
  #dds <- DESeq(dds)
  #res <- results(dds)
  num <- strsplit(strsplit(fx, "counts_")[[1]][-1], "\\.")[[1]][1]
  write.csv(res, paste(out_path, 'res', num, '.csv', sep=''))
}
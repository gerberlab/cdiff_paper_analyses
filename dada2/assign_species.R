library(phyloseq); packageVersion("phyloseq")
library(Biostrings); packageVersion("Biostrings")
library(ggplot2); packageVersion("ggplot2")
library(plyr); packageVersion("plyr")
library(openxlsx)
library(pracma)

library(dada2); packageVersion("dada2")
library(set);
setwd("/Users/jendawk/Dropbox (MIT)/C Diff Recurrence Paper/Analyses/dada2/raw_data")

temp <- read.xlsx("seqtab-nochim.xlsx", colNames = FALSE, rowNames = FALSE)
otumat = temp[-1,-1]
colnames(otumat) = temp[1,-1]
rownames(otumat) = temp[-1,1]

seqtab.nochim <- otumat

seqs <- rownames(seqtab.nochim)
taxa <- assignTaxonomy(seqs, "training/rdp_train_set_16.fa.gz",multithread=FALSE)
taxa.plus <- addSpecies(taxa, "training/rdp_species_assignment_16.fa.gz")
write.csv(t(taxa.plus), 'dada2-taxonomy-rdp.csv')

taxa_silva <- assignTaxonomy(seqs, "training/silva_nr99_v138_train_set.fa",multithread=FALSE)
taxa_silva.plus <- addSpecies(taxa_silva, "training/silva_species_assignment_v138.fa")
write.csv(t(taxa_silva.plus), 'dada2-taxonomy-silva.csv')

# ============================================================================
# FINANCE APPLICATIONS IN DATA SCIENCE ASSIGNMENT
# Questions 1, 2, and 3: WSJ Articles Analysis
# DATE: March 2026
# DATASET: WSJarticlesFADS.rds (WSJ front-page articles 2016-2025)
# 
# ============================================================================

# Load dplyr for data manipulation (mutate, group_by, summarize, filter)
library(dplyr)
# Load tm (Text Mining) for corpus creation and text cleaning
library(tm)
# Load SnowballC for Porter stemming (wordStem function)
library(SnowballC)
# Load SentimentAnalysis for the analyzeSentiment() function with LM dictionary
library(SentimentAnalysis)
# Load ggplot2 for creating all visualizations
library(ggplot2)
# Load stringr for string utility functions (str_count etc.)
library(stringr)
# Load zoo for rollmean() to compute the 22-day moving average
library(zoo)
# Load wordcloud for generating word cloud visualizations
library(wordcloud)
# Load RColorBrewer for color palettes used in word clouds and plots
library(RColorBrewer)
# Load RWeka for n-gram tokenization (NGramTokenizer)
library(RWeka)
# Load httr for running LLM locally in Ollama
library(httr)
# Load jsonlite for working with JSON data, needed when using API to run prompts in Ollama
library(jsonlite)
# Load lubridate for working with dates, needed for selecting articles in question 6
library(lubridate)
# Load tidyr for cleaning and reshaping data
library(tidyr)
# Load readr to store data in a csv
library(readr)
# Load purrr for map functions
library(purrr)

# ============================================================================
# HOW TO RUN THIS SCRIPT
# ============================================================================
# 1. Set the working directory below to the folder containing:
#      WSJarticlesFADS.rds
#      Scaled_Word_Weights_By_Topic_Phi_tilde.csv
#      Loughran-McDonald_MasterDictionary_1993-2024.csv
# 2. All output files will be saved to the same folder.
# 3. On first run, checkpoint .rds files are created to speed up re-runs.
#    If you change equity synonyms (Q3) or bubble terms (Q4), delete:
#      equity_articles.rds, equity_with_sentiment.rds, equity_with_uncertainty.rds
#    before re-running so the new settings take effect.
# 4. Results are fully reproducible: set.seed(123) is used in Q4 sampling.
# ============================================================================

# CHANGE THIS PATH to the folder where your data files are located:
setwd("/home/discodruif/data science 2/assignment") # set working directory so relative file paths resolve correctly

# ============================================================================
# LOAD DATA
# ============================================================================


wsjdata <- readRDS("WSJarticlesFADS.rds") # load the WSJ front-page articles from the RDS file

# ============================================================================
# QUESTION 1: Summary Statistics and Token-based Filtering
# ============================================================================
cat("\n============================================================\n")
cat("  QUESTION 1: Summary Statistics and Token-based Filtering\n")
cat("============================================================\n")

# gregexpr finds every word boundary match, lengths() counts them, +1 corrects for off-by-one
wsjdata$token_count <- lengths(gregexpr("[A-z]\\W+", wsjdata$Fulltxt)) + 1

# Store total article count BEFORE filtering for Q1 statistic
total_articles <- nrow(wsjdata)
cat("1) Total number of articles:", total_articles, "\n\n")

# Compute average articles per day: group by date, count articles, then average
articles_per_day <- wsjdata %>%
  mutate(publication_date = as.Date(Pdate)) %>%  # convert Pdate string to Date object
  group_by(publication_date) %>%                 # one group per calendar day
  summarize(articles = n(), .groups = "drop")    # count articles per day

avg_articles_per_day <- mean(articles_per_day$articles) # average across all days
cat("2) Average number of articles per day:", round(avg_articles_per_day, 2), "\n\n")

# Average token count before removing short articles
avg_tokens_before <- mean(wsjdata$token_count, na.rm = TRUE)
cat("3) Average tokens per article (before filtering):", round(avg_tokens_before, 2), "\n\n")

# Compute the 5th percentile cutoff for token length
token_5th_percentile <- quantile(wsjdata$token_count, probs = 0.05, na.rm = TRUE)
cat("5th percentile of token count:", token_5th_percentile, "\n")

# Keep only articles at or above the 5th percentile (removes bottom 5% shortest articles)
wsjdata_filtered <- wsjdata[wsjdata$token_count >= token_5th_percentile, ] # base R subset, avoids dplyr scoping issues

articles_removed <- total_articles - nrow(wsjdata_filtered) # number of articles dropped
cat("Articles removed:", articles_removed, "\n")
cat("Articles remaining:", nrow(wsjdata_filtered), "\n\n")

# Average token count after removing short articles
avg_tokens_after <- mean(wsjdata_filtered$token_count, na.rm = TRUE)
cat("4) Average tokens per article (after filtering):", round(avg_tokens_after, 2), "\n\n")


# ============================================================================
# QUESTION 2: Recession Indicator using Dictionary Method
# ============================================================================
cat("\n============================================================\n")
cat("  QUESTION 2: Recession Indicator using Dictionary Method\n")
cat("============================================================\n")


# Load the scaled topic word weights CSV from Bybee et al. (structureofnews.com)
word_weights <- read.csv("Scaled_Word_Weights_By_Topic_Phi_tilde.csv")

# Sort all terms by their Recession topic weight (descending) and keep top 50
recession_words <- word_weights$term[order(word_weights$Recession, decreasing = TRUE)][1:50]

# Helper function: stem a phrase that may contain multiple words
stem_phrase <- function(phrase) {
  words <- unlist(strsplit(tolower(phrase), " ")) # split phrase into individual tokens
  paste(wordStem(words, language = "english"), collapse = " ") # stem each token and rejoin
}

# Apply stemming to every recession dictionary word (keeps multi-word phrases intact)
recession_words_stemmed <- sapply(recession_words, stem_phrase)
recession_words_stemmed <- unique(recession_words_stemmed) # remove duplicate stems

# Split dictionary into single-token terms and multi-token phrases for correct matching
recession_unigrams <- recession_words_stemmed[!grepl(" ", recession_words_stemmed)]
recession_phrases  <- recession_words_stemmed[grepl(" ", recession_words_stemmed)]


# Initialise recession word count column to zero for all articles
wsjdata_filtered$recession_count <- 0

for (r in seq_len(nrow(wsjdata_filtered))) {    # loop over every article
  text <- wsjdata_filtered[r, "Fulltxt"]        # extract the full article text
  text <- gsub("\\d|[[:punct:]]", "", text)    # remove digits and punctuation
  text <- tolower(text)                          # convert to lowercase
  words <- unlist(strsplit(text, "\\s+"))       # split into individual tokens
  words_stemmed <- wordStem(words, language = "english") # stem each token
  text_stemmed <- paste(words_stemmed, collapse = " ") # stemmed full text used for phrase matching
  unigram_count <- sum(words_stemmed %in% recession_unigrams) # token-level dictionary matches
  phrase_count <- 0
  if (length(recession_phrases) > 0) {
    phrase_count <- sum(vapply(recession_phrases, function(phrase) {
      grepl(paste0("(^| )", phrase, "( |$)"), text_stemmed)
    }, logical(1)))
  }
  recession_count <- unigram_count + phrase_count # total includes both unigram and phrase matches
  # Normalise by article length to get the share of recession words (not raw count)
  wsjdata_filtered[r, "recession_count"] <- recession_count / wsjdata_filtered[r, "token_count"]
}

# Aggregate article-level scores to daily level (mean across all articles on that day)
daily_recession <- wsjdata_filtered %>%
  mutate(date = as.Date(Pdate)) %>%             # parse publication date
  group_by(date) %>%                            # one group per calendar day
  summarize(
    recession_index = mean(recession_count, na.rm = TRUE), # daily mean recession share
    n_articles = n(),                           # number of articles that day
    .groups = "drop"
  ) %>%
  arrange(date)                                 # sort chronologically


# Compute 22-day right-aligned moving average to smooth daily noise
daily_recession <- daily_recession %>%
  mutate(recession_ma22 = rollmean(recession_index, k = 22, fill = NA, align = "right"))

# Print summary statistics of the recession index
cat("Recession Index Statistics:\n")
cat("  Mean:", round(mean(daily_recession$recession_index, na.rm = TRUE) * 100, 2), "%\n")
cat("  Max:", round(max(daily_recession$recession_index, na.rm = TRUE) * 100, 2), "% on",
    as.character(daily_recession$date[which.max(daily_recession$recession_index)]), "\n")
cat("  Min:", round(min(daily_recession$recession_index, na.rm = TRUE) * 100, 2), "% on",
    as.character(daily_recession$date[which.min(daily_recession$recession_index)]), "\n\n")

# Find the 5 days with the highest 22-day MA value (peak recession periods)
top_dates <- daily_recession %>%
  arrange(desc(recession_ma22)) %>% # sort by MA descending
  head(5)                           # keep top 5

cat("Top 5 periods with highest recession coverage (22-day MA):\n")
for (i in 1:5) {
  cat("  ", i, ".", as.character(top_dates$date[i]), ":",
      round(top_dates$recession_ma22[i] * 100, 2), "%\n")
}

# Export the daily recession index to CSV for use in the PDF writeup
write.csv(daily_recession, "Q2_recession_index_daily.csv", row.names = FALSE)

# Build the recession indicator time-series plot
plot_recession <- ggplot(daily_recession, aes(x = date)) +
  geom_line(aes(y = recession_index), color = "gray70", alpha = 0.6) +  # light gray = raw daily index
  geom_line(aes(y = recession_ma22), color = "#e74c3c", linewidth = 1) + # red = 22-day MA
  labs(title = "WSJ Recession Indicator (2016-2025)",
       subtitle = "Daily index (gray) and 22-day moving average (red)",
       x = "Date",
       y = "Recession Index (% of tokens)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

print(plot_recession)                                                                    # display in RStudio
ggsave("Q2_recession_indicator_timeseries.png", plot_recession, width = 12, height = 6, dpi = 300) # save to PNG

# Interpretation of Q2 findings is discussed in the PDF writeup.

# ============================================================================
# QUESTION 3: Equity Sentiment & Uncertainty
# ============================================================================
cat("\n============================================================\n")
cat("  QUESTION 3: Equity Sentiment & Uncertainty\n")
cat("============================================================\n")


# Check if equity articles already filtered and sentiment calculated
checkpoint_equity <- "equity_articles.rds"
checkpoint_sentiment <- "equity_with_sentiment.rds"
checkpoint_uncertainty <- "equity_with_uncertainty.rds"

# -------------------------------------------------------------------------
# STEP 1: Filter for equity articles (CHECKPOINT)
# -------------------------------------------------------------------------

if (file.exists(checkpoint_equity)) {
  equity_articles <- readRDS(checkpoint_equity)
} else {
  
  
  # Define equity synonyms
  # Define equity synonyms used to identify equity-related articles
  # "market"/"markets" removed (too broad - matches nearly all financial articles)
  # "shareholder"/"shareholders" added (more specific to equity ownership)
  equity_terms_raw <- c(
    "equity", "equities",   # direct equity references
    "stock", "stocks",      # stock market references
    "share", "shares",      # share references
    "shareholder", "shareholders" # ownership references
  )

  # Stem equity terms so they match the stemmed article text
  equity_terms_stemmed <- unique(wordStem(equity_terms_raw, language = "english"))

  # Initialise flag column: FALSE means no equity term found yet
  wsjdata_filtered$mentions_equity <- FALSE

  for (r in seq_len(nrow(wsjdata_filtered))) {      # loop over every filtered article
    text <- wsjdata_filtered[r, "Fulltxt"]          # get full article text
    text <- gsub("\\d|[[:punct:]]", "", text)      # remove digits and punctuation
    text <- tolower(text)                            # lowercase
    words <- unlist(strsplit(text, "\\s+"))         # tokenize by whitespace
    words_stemmed <- wordStem(words, language = "english") # apply Porter stemming
    if (any(words_stemmed %in% equity_terms_stemmed)) # check for any equity stem match
      wsjdata_filtered[r, "mentions_equity"] <- TRUE  # flag the article
  }

  # Keep only articles that contain at least one equity synonym
  equity_articles <- wsjdata_filtered[wsjdata_filtered$mentions_equity == TRUE, ]

  saveRDS(equity_articles, checkpoint_equity) # save checkpoint to skip this step on re-runs
}

# -------------------------------------------------------------------------
# STEP 2: Calculate sentiment in BATCHES (CHECKPOINT)
# -------------------------------------------------------------------------

if (file.exists(checkpoint_sentiment)) {
  equity_articles <- readRDS(checkpoint_sentiment)
} else {
  

  equity_articles$SentimentLM <- NA  # initialise sentiment column
  batch_size <- 1000                  # process 1000 articles at a time to manage memory
  n_batches <- ceiling(nrow(equity_articles) / batch_size) # number of batches needed

  for (batch in 1:n_batches) {
    start_idx <- (batch - 1) * batch_size + 1          # first article index in this batch
    end_idx   <- min(batch * batch_size, nrow(equity_articles)) # last article index
    batch_data <- equity_articles[start_idx:end_idx, ] # subset to current batch

    # Build VCorpus from raw article text in this batch, then apply cleaning and stemming
    corpus_batch <- VCorpus(VectorSource(batch_data$Fulltxt))
    corpus_batch <- tm_map(corpus_batch, content_transformer(tolower))        # lowercase all text
    corpus_batch <- tm_map(corpus_batch, removeWords, stopwords("english"))   # remove English stop words

    removeSpecialChars <- function(x) {           # helper: remove non-alphanumeric characters
      x = gsub("[^a-zA-Z0-9 ]","",x)            # strip everything except letters, digits, spaces
      x = gsub("\\s+", " ", x)                  # collapse multiple spaces to one
      return(trimws(x))                           # trim leading/trailing whitespace
    }

    corpus_batch <- tm_map(corpus_batch, content_transformer(removeSpecialChars)) # apply cleanup
    corpus_batch <- tm_map(corpus_batch, removeNumbers)   # remove any remaining numbers
    corpus_batch <- tm_map(corpus_batch, stripWhitespace) # final whitespace strip
    corpus_batch <- tm_map(corpus_batch, stemDocument)    # apply Porter stemming

    sentiment_batch <- analyzeSentiment(corpus_batch)  # compute LM sentiment scores
    equity_articles$SentimentLM[start_idx:end_idx] <- sentiment_batch$SentimentLM # store LM net score

    rm(corpus_batch, sentiment_batch, batch_data) # free memory after each batch
    gc(verbose = FALSE)                           # trigger garbage collection
  }

  cat("Sentiment LM statistics:\n")
  print(summary(equity_articles$SentimentLM)) # display distribution of sentiment scores

  saveRDS(equity_articles, checkpoint_sentiment) # save checkpoint to skip this step on re-runs
}

# -------------------------------------------------------------------------
# STEP 3: Calculate uncertainty in BATCHES (CHECKPOINT)
# -------------------------------------------------------------------------

if (file.exists(checkpoint_uncertainty)) {
  equity_articles <- readRDS(checkpoint_uncertainty)
} else {
  

  # Load the Loughran-McDonald Master Dictionary (1993-2024)
  lm_dict <- read.csv("Loughran-McDonald_MasterDictionary_1993-2024.csv",
                      stringsAsFactors = FALSE)

  # Extract words flagged as uncertainty words (Uncertainty column > 0)
  uncertainty_words_raw    <- lm_dict$Word[lm_dict$Uncertainty > 0]
  # Stem the uncertainty words to match the stemmed article text
  uncertainty_words_stemmed <- unique(wordStem(tolower(uncertainty_words_raw), language = "english"))

  equity_articles$uncertainty_count <- NA  # initialise uncertainty column
  batch_size <- 2000                        # larger batch size; simpler loop than sentiment
  n_batches  <- ceiling(nrow(equity_articles) / batch_size)

  for (batch in 1:n_batches) {
    start_idx <- (batch - 1) * batch_size + 1
    end_idx   <- min(batch * batch_size, nrow(equity_articles))
    for (r in start_idx:end_idx) {                            # loop articles within batch
      text <- equity_articles[r, "Fulltxt"]                  # raw article text
      text <- gsub("\\d|[[:punct:]]", "", text)             # remove digits and punctuation
      text <- tolower(text)                                   # lowercase
      words <- unlist(strsplit(text, "\\s+"))                # tokenize
      words_stemmed <- wordStem(words, language = "english") # stem tokens
      uncertainty_count <- sum(words_stemmed %in% uncertainty_words_stemmed) # count matches
      # Normalise by article length to get proportion of uncertainty words
      equity_articles[r, "uncertainty_count"] <- uncertainty_count / equity_articles[r, "token_count"]
    }
  }

  cat("Uncertainty statistics:\n")
  print(summary(equity_articles$uncertainty_count)) # display distribution

  saveRDS(equity_articles, checkpoint_uncertainty) # save checkpoint
}

# -------------------------------------------------------------------------
# STEP 4: Aggregate to daily level with 22-day MA
# -------------------------------------------------------------------------


# Compute daily mean sentiment and uncertainty, then apply 22-day rolling averages
daily_equity <- equity_articles %>%
  mutate(date = as.Date(Pdate)) %>%              # parse publication date
  group_by(date) %>%                            # group by calendar day
  summarize(
    sentiment_lm  = mean(SentimentLM, na.rm = TRUE),      # mean LM sentiment per day
    uncertainty   = mean(uncertainty_count, na.rm = TRUE), # mean uncertainty share per day
    n_articles    = n(),                                   # article count per day
    .groups = "drop"
  ) %>%
  arrange(date) %>%                             # sort chronologically
  mutate(
    sentiment_ma22  = rollmean(sentiment_lm, k = 22, fill = NA, align = "right"), # 22-day MA of sentiment
    uncertainty_ma22 = rollmean(uncertainty, k = 22, fill = NA, align = "right")  # 22-day MA of uncertainty
  )


# -------------------------------------------------------------------------
# STEP 5: Merge with recession indicator and calculate correlations
# -------------------------------------------------------------------------


# Left-join: keep all equity days, attach recession MA where dates overlap
combined <- daily_equity %>%
  left_join(daily_recession, by = "date", suffix = c("_equity", "_recession"))

# Drop rows where any MA series is NA (leading edge of 22-day window)
combined_clean <- combined[!is.na(combined$sentiment_ma22) &
                           !is.na(combined$recession_ma22) &
                           !is.na(combined$uncertainty_ma22), ]


# Pearson correlation between the three 22-day MA series
cor_sentiment_recession  <- cor(combined_clean$sentiment_ma22,  combined_clean$recession_ma22)   
cor_uncertainty_recession <- cor(combined_clean$uncertainty_ma22, combined_clean$recession_ma22) 
cor_sentiment_uncertainty <- cor(combined_clean$sentiment_ma22,  combined_clean$uncertainty_ma22) 

cat("CORRELATION RESULTS (22-day moving averages):\n\n")
cat("1. Sentiment (LM) vs Recession Index:      ", round(cor_sentiment_recession, 4), "\n")
cat("2. Uncertainty vs Recession Index:         ", round(cor_uncertainty_recession, 4), "\n")
cat("3. Sentiment (LM) vs Uncertainty:          ", round(cor_sentiment_uncertainty, 4), "\n\n")

# -------------------------------------------------------------------------
# STEP 6: Save results
# -------------------------------------------------------------------------

write.csv(daily_equity, "Q3_equity_sentiment_daily.csv", row.names = FALSE)  # daily equity sentiment time-series
write.csv(combined,    "Q3_sentiment_recession_combined.csv", row.names = FALSE) # merged equity + recession series

# Interpretation of Q3 findings (correlations, historical context, methodological
# considerations) is discussed in the PDF writeup.

# ============================================================================
# QUESTION 4: Bubble Articles - Salient Unigrams and Bigrams
# ============================================================================
cat("\n============================================================\n")
cat("  QUESTION 4: Bubble Articles - Salient Unigrams and Bigrams\n")
cat("============================================================\n")


# ============================================================================
# STEP 1: Load equity articles from Q3
# ============================================================================

if (!exists("equity_articles")) {                 # only load from disk if not already in memory
  equity_articles <- readRDS("equity_articles.rds") # read checkpoint produced in Q3
}

# ============================================================================
# STEP 2: Filter for bubble mentions (stemmed)
# ============================================================================


# Define bubble terms
bubble_terms <- c("bubble", "bubbles", "froth", "frothy", "overvalued",
                  "overvaluation", "overvalue", "mania", "manias",
                  "euphoria", "euphoric", "irrational exuberance") # vocabulary from bubble literature

# Stem bubble terms, preserving multi-word phrases
bubble_terms_stemmed <- unique(sapply(bubble_terms, stem_phrase)) # stem to match processed article text
bubble_unigrams <- bubble_terms_stemmed[!grepl(" ", bubble_terms_stemmed)]
bubble_phrases  <- bubble_terms_stemmed[grepl(" ", bubble_terms_stemmed)]

# Filter articles mentioning bubble terms (on stemmed text)
equity_articles$mentions_bubble <- FALSE # initialise flag column

for (r in seq_len(nrow(equity_articles))) {
  text <- equity_articles$Fulltxt[r]       # raw article text

  # Clean and stem (matching Tutorial1 pattern)
  text <- gsub("[^a-zA-Z0-9 ]", "", text)  # remove punctuation / special characters
  text <- tolower(text)                     # lowercase
  words <- unlist(strsplit(text, "\\s+"))   # split on whitespace into word tokens
  words_stemmed <- wordStem(words, language = "english") # stem each token
  text_stemmed <- paste(words_stemmed, collapse = " ") # stemmed full text used for phrase matching

  # Check if any bubble unigram or phrase appears
  unigram_match <- any(words_stemmed %in% bubble_unigrams)
  phrase_match <- FALSE
  if (length(bubble_phrases) > 0) {
    phrase_match <- any(vapply(bubble_phrases, function(phrase) {
      grepl(paste0("(^| )", phrase, "( |$)"), text_stemmed)
    }, logical(1)))
  }
  if (unigram_match || phrase_match) {
    equity_articles$mentions_bubble[r] <- TRUE # mark article as bubble-related
  }

}

# Create subsamples
bubble_articles     <- equity_articles[equity_articles$mentions_bubble == TRUE, ]  # articles with bubble term
non_bubble_articles <- equity_articles[equity_articles$mentions_bubble == FALSE, ] # articles without


# ============================================================================
# STEP 3: Verify asset market context
# ============================================================================


asset_keywords <- c("market", "markets", "stock", "stocks", "asset", "assets",
                   "price", "prices", "valuation", "valuations") # words that confirm asset-market context
asset_keywords_stemmed <- unique(wordStem(tolower(asset_keywords), language = "english")) # stem for matching

bubble_articles$has_asset_context <- FALSE # initialise context flag

for (r in seq_len(nrow(bubble_articles))) {
  text <- bubble_articles$Fulltxt[r]
  text <- gsub("[^a-zA-Z0-9 ]", "", text) # remove non-alphanumeric characters
  text <- tolower(text)                    # lowercase
  words <- unlist(strsplit(text, "\\s+"))  # tokenise
  words_stemmed <- wordStem(words, language = "english") # stem

  if (any(words_stemmed %in% asset_keywords_stemmed)) {
    bubble_articles$has_asset_context[r] <- TRUE # article mentions a financial asset keyword
  }
}

articles_with_context <- sum(bubble_articles$has_asset_context)          # count of articles with asset mention

# Keep only bubble articles with asset context
bubble_articles <- bubble_articles[bubble_articles$has_asset_context == TRUE, ] # drop irrelevant bubble mentions


# Sample non-bubble articles to balance dataset and improve speed
# Use 3x bubble sample size for comparison
set.seed(123) # fix random seed for reproducibility
sample_size <- min(nrow(bubble_articles) * 3, nrow(non_bubble_articles)) # 3:1 non-bubble:bubble ratio
non_bubble_articles_sampled <- non_bubble_articles[sample(nrow(non_bubble_articles), sample_size), ] # draw random sample


# ============================================================================
# STEP 4: Create cleaned+stemmed corpora and collapse to single strings
# ============================================================================



corpus_bubble     <- VCorpus(VectorSource(bubble_articles$Fulltxt))              # create volatile corpus for bubble articles
corpus_non_bubble <- VCorpus(VectorSource(non_bubble_articles_sampled$Fulltxt)) # create volatile corpus for non-bubble articles


clean_corpus <- function(corpus) {
  corpus <- tm_map(corpus, content_transformer(tolower))      # lowercase all text
  corpus <- tm_map(corpus, removeWords, stopwords("english")) # remove English stop words

  removeSpecialChars <- function(x) {
    x <- gsub("[^a-zA-Z0-9 ]", "", x) # strip non-alphanumeric characters
    x <- gsub("\\s+", " ", x)         # collapse multiple spaces into one
    return(trimws(x))                  # remove leading / trailing whitespace
  }

  corpus <- tm_map(corpus, content_transformer(removeSpecialChars)) # apply custom cleaner
  corpus <- tm_map(corpus, removeNumbers)                            # remove digit tokens
  corpus <- tm_map(corpus, stripWhitespace)                         # final whitespace normalisation

  return(corpus)
}

# Create cleaned-only corpus (for readable display terms)
corpus_bubble_clean     <- clean_corpus(corpus_bubble)     # cleaned but NOT stemmed — used for display words
corpus_non_bubble_clean <- clean_corpus(corpus_non_bubble) # cleaned but NOT stemmed

# Create cleaned+stemmed corpus (for salience calculation)
corpus_bubble_stemmed     <- tm_map(corpus_bubble_clean,     stemDocument) # stem bubble corpus
corpus_non_bubble_stemmed <- tm_map(corpus_non_bubble_clean, stemDocument) # stem non-bubble corpus


text_bubble_clean     <- paste(sapply(corpus_bubble_clean,     as.character), collapse = " ") # unstemmed bubble string
text_non_bubble_clean <- paste(sapply(corpus_non_bubble_clean, as.character), collapse = " ") # unstemmed non-bubble string
text_bubble     <- paste(sapply(corpus_bubble_stemmed,     as.character), collapse = " ")     # stemmed bubble string
text_non_bubble <- paste(sapply(corpus_non_bubble_stemmed, as.character), collapse = " ")    # stemmed non-bubble string


# ============================================================================
# STEP 5: Extract UNIGRAMS and calculate salience
# ============================================================================


# Extract unigrams (words) from stemmed collapsed strings
unigrams_bubble     <- unlist(strsplit(text_bubble,     "\\s+")) # tokenise stemmed bubble text
unigrams_non_bubble <- unlist(strsplit(text_non_bubble, "\\s+")) # tokenise stemmed non-bubble text


unigrams_bubble_table     <- table(unigrams_bubble)     # term frequencies in bubble corpus
unigrams_non_bubble_table <- table(unigrams_non_bubble) # term frequencies in non-bubble corpus


common_unigrams <- intersect(names(unigrams_bubble_table), names(unigrams_non_bubble_table)) # only terms present in both


# Filter both tables to only common terms (ratio undefined otherwise)
unigrams_bubble_common     <- unigrams_bubble_table[names(unigrams_bubble_table)         %in% common_unigrams]
unigrams_non_bubble_common <- unigrams_non_bubble_table[names(unigrams_non_bubble_table) %in% common_unigrams]


unigrams_bubble_norm     <- unigrams_bubble_common     / sum(unigrams_bubble_common)     # relative frequency in bubble
unigrams_non_bubble_norm <- unigrams_non_bubble_common / sum(unigrams_non_bubble_common) # relative frequency in non-bubble

unigram_ratio        <- unigrams_bubble_norm / unigrams_non_bubble_norm # salience ratio: bubble freq / non-bubble freq
unigram_ratio_sorted <- sort(unigram_ratio, decreasing = TRUE)          # rank by descending salience
salient_unigrams     <- head(unigram_ratio_sorted, 100)                 # keep top-100 salient unigrams

cat("Top 10 salient unigrams (stemmed):\n")
print(head(salient_unigrams, 10))

# Build stem-to-display mapping: for each stem, find the most frequent unstemmed form
unigrams_bubble_clean   <- unlist(strsplit(text_bubble_clean, "\\s+")) # unstemmed tokens (same order as stemmed)
unigrams_bubble_stemmed <- unlist(strsplit(text_bubble,       "\\s+")) # stemmed tokens

# Pair each stemmed token with its corresponding unstemmed token
valid_idx <- nchar(unigrams_bubble_stemmed) > 0 & nchar(unigrams_bubble_clean) > 0 # drop empty strings
stem_display_pairs <- data.frame(
  stemmed = unigrams_bubble_stemmed[valid_idx],
  display = unigrams_bubble_clean[valid_idx],
  stringsAsFactors = FALSE
)

# Count how often each (stem, display) pair occurs, then pick the most frequent display form per stem
stem_display_table <- table(paste(stem_display_pairs$stemmed, stem_display_pairs$display, sep = "|||")) # combined key
stem_display_df <- data.frame(
  pair  = names(stem_display_table),
  count = as.numeric(stem_display_table),
  stringsAsFactors = FALSE
)
stem_display_df$stemmed <- sapply(strsplit(stem_display_df$pair, "|||", fixed = TRUE), `[`, 1) # extract stem
stem_display_df$display <- sapply(strsplit(stem_display_df$pair, "|||", fixed = TRUE), `[`, 2) # extract display word
stem_display_sorted   <- stem_display_df[order(stem_display_df$stemmed, -stem_display_df$count), ] # sort by stem then freq
stem_to_display_unigram <- stem_display_sorted[!duplicated(stem_display_sorted$stemmed), ]          # keep top display per stem

# Replace stemmed names in salient_unigrams with their most frequent unstemmed form
salient_unigrams_display <- salient_unigrams
for (i in seq_along(salient_unigrams)) {
  stem      <- names(salient_unigrams)[i]                              # stemmed term
  match_idx <- which(stem_to_display_unigram$stemmed == stem)[1]       # find its display mapping
  if (!is.na(match_idx)) {
    names(salient_unigrams_display)[i] <- stem_to_display_unigram$display[match_idx] # replace with readable form
  }
}

cat("Top 10 salient unigrams (unstemmed for display):\n")
print(head(salient_unigrams_display, 10))

# ============================================================================
# STEP 6: Extract BIGRAMS and calculate salience
# ============================================================================



BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2)) # RWeka tokenizer: extract consecutive word pairs

# Extract bigrams from stemmed collapsed strings
bigrams_bubble     <- BigramTokenizer(text_bubble)     # all bigrams in bubble corpus
bigrams_non_bubble <- BigramTokenizer(text_non_bubble) # all bigrams in non-bubble corpus


bigrams_bubble_table     <- table(bigrams_bubble)     # bigram frequencies in bubble
bigrams_non_bubble_table <- table(bigrams_non_bubble) # bigram frequencies in non-bubble


common_bigrams <- intersect(names(bigrams_bubble_table), names(bigrams_non_bubble_table)) # bigrams present in both corpora


# Filter both tables to common bigrams
bigrams_bubble_common     <- bigrams_bubble_table[names(bigrams_bubble_table)         %in% common_bigrams]
bigrams_non_bubble_common <- bigrams_non_bubble_table[names(bigrams_non_bubble_table) %in% common_bigrams]


bigrams_bubble_norm     <- bigrams_bubble_common     / sum(bigrams_bubble_common)     # relative frequency in bubble
bigrams_non_bubble_norm <- bigrams_non_bubble_common / sum(bigrams_non_bubble_common) # relative frequency in non-bubble

bigram_ratio        <- bigrams_bubble_norm / bigrams_non_bubble_norm # salience ratio
bigram_ratio_sorted <- sort(bigram_ratio, decreasing = TRUE)         # rank by descending salience
salient_bigrams     <- head(bigram_ratio_sorted, 100)                # top-100 salient bigrams

cat("Top 10 salient bigrams (stemmed):\n")
print(head(salient_bigrams, 10))

# Build stem-to-display mapping for bigrams (same logic as unigrams but per article)
salient_bigrams_display <- salient_bigrams          # will overwrite names with unstemmed forms
needed_stems            <- names(salient_bigrams)   # only look up bigrams in the top-100

bigram_mapping <- list() # final stem -> display word mapping
bigram_counts  <- list() # intermediate: count each (stem, display) pair

for (i in seq_len(nrow(bubble_articles))) {
  text <- bubble_articles$Fulltxt[i]

  # Clean text (same pipeline as corpus cleaning)
  text_clean  <- gsub("[^a-zA-Z0-9 ]", "", text)                       # remove punctuation
  text_clean  <- tolower(text_clean)                                    # lowercase
  words_clean <- unlist(strsplit(text_clean, "\\s+"))                   # tokenise
  words_clean <- words_clean[!words_clean %in% stopwords("english")]   # remove stop words
  words_clean <- words_clean[nchar(words_clean) > 0]                   # remove empty tokens

  words_stemmed <- wordStem(words_clean, language = "english")          # stem

  # Scan consecutive word pairs; only record pairs that are salient
  if (length(words_stemmed) > 1) {
    for (j in 1:(length(words_stemmed) - 1)) {
      bigram_stemmed <- paste(words_stemmed[j], words_stemmed[j + 1])  # stemmed bigram
      if (bigram_stemmed %in% needed_stems) {
        bigram_clean <- paste(words_clean[j], words_clean[j + 1])      # unstemmed display bigram
        key <- paste(bigram_stemmed, bigram_clean, sep = "|||")         # combined key for counting
        if (is.null(bigram_counts[[key]])) {
          bigram_counts[[key]] <- 1
        } else {
          bigram_counts[[key]] <- bigram_counts[[key]] + 1             # increment count
        }
      }
    }
  }

}

# For each stemmed bigram, pick the most frequently observed unstemmed form
for (stem in needed_stems) {
  matching_keys <- names(bigram_counts)[grepl(paste0("^", gsub("([.|()\\^{}+$*?]|\\[|\\])", "\\\\\\1", stem), "\\|\\|\\|"), names(bigram_counts))]
  if (length(matching_keys) > 0) {
    counts   <- sapply(matching_keys, function(k) bigram_counts[[k]]) # frequency of each display form
    best_key <- matching_keys[which.max(counts)]                       # most frequent display form
    bigram_mapping[[stem]] <- strsplit(best_key, "|||", fixed = TRUE)[[1]][2] # store display bigram
  }
}

# Replace stemmed bigram names with readable display forms
for (i in seq_along(salient_bigrams)) {
  stem <- names(salient_bigrams)[i]
  if (!is.null(bigram_mapping[[stem]])) {
    names(salient_bigrams_display)[i] <- bigram_mapping[[stem]] # swap in unstemmed bigram
  }
}

cat("Top 10 salient bigrams (unstemmed for display):\n")
print(head(salient_bigrams_display, 10))

# ============================================================================
# STEP 7: Wordcloud for salient unigrams
# ============================================================================



png("Q4_salient_unigrams_wordcloud.png", width = 800, height = 800, res = 150) # open PNG device
par(mar = c(0, 0, 0, 0))                          # remove all margins so words fill the canvas

wordcloud(words = names(salient_unigrams_display), # display terms (unstemmed)
          freq  = salient_unigrams_display,         # salience ratio used as frequency weight
          scale = c(3.5, 0.5),                      # range of word sizes (largest to smallest)
          min.freq     = 1,                         # include all terms (ratio already filtered)
          max.words    = 100,                       # limit to top-100 for readability
          random.order = FALSE,                     # most salient term drawn first (centre)
          rot.per      = 0.35,                      # 35% of words drawn vertically
          colors       = brewer.pal(8, "Dark2"))    # 8-colour Dark2 palette

dev.off()                                           # close PNG device and write file


# ============================================================================
# STEP 8: Wordcloud for salient bigrams
# ============================================================================


png("Q4_salient_bigrams_wordcloud.png", width = 800, height = 800, res = 150) # open PNG device
par(mar = c(0, 0, 0, 0))                         # remove all margins so words fill the canvas

wordcloud(words = names(salient_bigrams_display), # display bigrams (unstemmed)
          freq  = salient_bigrams_display,         # salience ratio used as frequency weight
          scale = c(3.5, 0.5),                     # range of word sizes
          min.freq     = 1,                        # include all terms
          max.words    = 100,                      # top-100 bigrams
          random.order = FALSE,                    # most salient drawn first
          rot.per      = 0.35,                     # 35% vertical
          colors       = brewer.pal(8, "Dark2"))   # Dark2 colour palette

dev.off()                                          # close PNG device and write file


# ============================================================================
# STEP 9: Save results
# ============================================================================


# Build data frames with display terms and their salience ratios for export
salient_unigrams_df <- data.frame(
  term  = names(salient_unigrams_display),  # readable (unstemmed) unigram
  ratio = as.numeric(salient_unigrams),     # salience ratio (bubble freq / non-bubble freq)
  row.names = NULL
)

salient_bigrams_df <- data.frame(
  term  = names(salient_bigrams_display),   # readable (unstemmed) bigram
  ratio = as.numeric(salient_bigrams),      # salience ratio
  row.names = NULL
)

write.csv(salient_unigrams_df, "Q4_salient_unigrams.csv", row.names = FALSE) # export unigram results
write.csv(salient_bigrams_df,  "Q4_salient_bigrams.csv",  row.names = FALSE) # export bigram results

# ============================================================================
# QUESTION 5: LLM-Based Classification of Equity News
# ============================================================================
cat("\n============================================================\n")
cat("  QUESTION 5: LLM-Based Classification of Equity News\n")
cat("============================================================\n")

# Set random seed to ensure reproducbility
set.seed(42)

# Use raw text data
data <- wsjdata

# ============================================================================
# STEP 1: Prepare article text
# ============================================================================

# Combine title and lead paragraph
data$article_date <- data$Pdate
data$article_text <- paste(data$Title, data$LeadPar, sep = ". ")

# Keep rows with text
valid_rows <- which(!is.na(data$article_text))
cat("Valid articles with usable text:", length(valid_rows), "\n")

# Random sample of 40 articles
sample_idx <- sample(valid_rows, min(40, length(valid_rows)))
sample_articles <- data[sample_idx, ]
article_texts <- sample_articles$article_text

cat("Sample size:", length(article_texts), "\n")
cat("Sample date range:\n")
print(range(sample_articles$article_date, na.rm = TRUE))

# ============================================================================
# STEP 2: Connect to OpenAI ChatGPT
# ============================================================================

# Set your key before running:
# Sys.setenv(OPENAI_API_KEY = "your-api-key-here") in console first, then run MY_KEY
MY_KEY <- Sys.getenv("OPENAI_API_KEY")

chatGPT <- function(prompt,
                    modelName = "gpt-4o",
                    temperature = 0,
                    apiKey = Sys.getenv("OPENAI_API_KEY")) {
  
  if (nchar(apiKey) == 0) {
    stop("OPENAI_API_KEY is empty. Set it first with Sys.setenv(...).")
  }
  
  response <- POST(
    url = "https://api.openai.com/v1/chat/completions",
    add_headers(Authorization = paste("Bearer", apiKey)),
    content_type_json(),
    encode = "json",
    body = list(
      model = modelName,
      temperature = temperature,
      messages = list(
        list(role = "user", content = prompt)
      )
    )
  )
  
  txt <- content(response, as = "text", encoding = "UTF-8")
  cat(txt, "\n\n")   # helpful for debugging
  
  if (status_code(response) >= 300) {
    stop(txt)
  }
  
  # Prevent over-simplification
  parsed <- fromJSON(txt, simplifyVector = FALSE)
  
  parsed$choices[[1]]$message$content
}

# Connection test
test_openai <- chatGPT("Say 'Connection successful' and nothing else.")
cat("OpenAI test:", test_openai, "\n")

# ============================================================================
# STEP 3: Connect to Ollama (local)
# ============================================================================

chatOllama <- function(prompt,
                       model = "gemma3:1b",
                       temperature = 0) {
  
  url <- "http://localhost:11434/api/generate"
  
  body <- list(
    model = model,
    prompt = prompt,
    stream = FALSE,
    options = list(temperature = temperature)
  )
  
  response <- tryCatch({
    POST(
      url,
      body = body,
      encode = "json",
      content_type_json(),
      timeout(120)
    )
  }, error = function(e) {
    warning(paste("Ollama connection error:", e$message))
    return(NULL)
  })
  
  if (is.null(response)) return(NA_character_)
  
  status <- status_code(response)
  txt <- content(response, as = "text", encoding = "UTF-8")
  
  cat("HTTP status:", status, "\n")
  cat("Raw response:\n", txt, "\n")
  
  if (status >= 300) {
    warning(paste("Ollama returned HTTP", status))
    return(NA_character_)
  }
  
  parsed <- fromJSON(txt, simplifyVector = FALSE)
  trimws(parsed$response)
}

# Connection test
test_ollama <- chatOllama("Say 'Connection successful' and nothing else.")
cat("Ollama test:", test_ollama, "\n")

# ============================================================================
# STEP 4: Define 3 prompts
# ============================================================================

# Version 1: minimal prompt
prompt_v1 <- paste(
  "Read the article and answer:",
  "1. Does it contain forward-looking information for equity markets?",
  "2. Is it bullish, bearish, or neutral for equity markets?",
  "3. What broad category does it belong to?",
  "Article:",
  sep = "\n"
)

# Version 2: better guidance + constrained format
prompt_v2 <- paste(
  "You are a financial analyst.",
  "",
  "For the article below, evaluate:",
  "1. forward-looking: Yes or No",
  "   Forward-looking means statements about expected future economic, policy, corporate, or market conditions.",
  "2. sentiment: Bullish, Bearish, or Neutral for broad equity markets.",
  "3. category: choose exactly one from this list:",
  "   Monetary Policy, Earnings & Corporate, Geopolitics & Conflict, Regulation & Policy,",
  "   Macroeconomic Data, Commodity & Energy, Trade & Tariffs, Technology & Innovation,",
  "   Market Technicals & Flows, Other",
  "",
  "Respond exactly in this format:",
  "forward_looking: <Yes or No>",
  "sentiment: <Bullish, Bearish, or Neutral>",
  "category: <one category>",
  "",
  "Article:",
  sep = "\n"
)

# Version 3: strongest prompt
prompt_v3 <- paste(
  "You are an equity market research assistant.",
  "",
  "Task: classify the article for broad equity-market analysis.",
  "",
  "Evaluate these fields:",
  "1. forward_looking: Yes if the article includes forecasts, guidance, expected future policy, expected economic conditions, or expected market effects. Otherwise No.",
  "2. sentiment: Bullish, Bearish, or Neutral for broad equity markets, not for a single stock only.",
  "3. category: choose exactly one from:",
  "   Monetary Policy",
  "   Macroeconomic Data",
  "   Earnings & Corporate",
  "   Geopolitics & Conflict",
  "   Trade & Tariffs",
  "   Regulation & Policy",
  "   Commodity & Energy",
  "   Market Technicals & Flows",
  "   Technology & Innovation",
  "   Other",
  "4. confidence: a number from 0.00 to 1.00 for your sentiment classification.",
  "5. rationale: one short sentence based only on the article.",
  "",
  "Rules:",
  "- Use Neutral if the direction for equities is unclear or mixed.",
  "- Do not add categories outside the list.",
  "- Do not infer facts not supported by the article.",
  "- If the article is too vague, use Neutral and low confidence.",
  "",
  "Respond exactly in this format:",
  "forward_looking: <Yes or No>",
  "sentiment: <Bullish, Bearish, or Neutral>",
  "confidence: <0.00 to 1.00>",
  "category: <one category from list>",
  "rationale: <one short sentence>",
  "",
  "Article:",
  sep = "\n"
)

# ============================================================================
# STEP 5: Function to run prompt over all sampled articles
# ============================================================================

run_prompt_set <- function(article_texts, prompt_template, model_name) {
  
  out <- character(length(article_texts))
  
  for (i in 1:length(article_texts)) {
    cat("Running article", i, "of", length(article_texts), "\n")
    
    full_prompt <- paste(prompt_template, article_texts[i])
    
    out[i] <- chatGPT(
      prompt = full_prompt,
      modelName = model_name,
      temperature = 0
    )
    
    Sys.sleep(1)
  }
  
  return(out)
}

#OLLAMA
run_prompt_set_ollama <- function(article_texts, prompt_template, model_name) {
  
  out <- character(length(article_texts))
  
  for (i in 1:length(article_texts)) {
    cat("Running article", i, "of", length(article_texts), "\n")
    
    full_prompt <- paste(prompt_template, article_texts[i])
    
    out[i] <- chatOllama(
      prompt = full_prompt,
      model = model_name,
      temperature = 0
    )
  }
  
  return(out)
}

# ============================================================================
# STEP 6: Run the three prompts on main model
# ============================================================================

# Models
main_model  <- "gpt-4o"
mini_model  <- "gpt-4o-mini"
other_model <- "gemma3:1b"

#Prompt V1 on main
results_v1_gpt4o <- run_prompt_set(
  article_texts = article_texts,
  prompt_template = prompt_v1,
  model_name = main_model
)

#Prompt V2 on main
results_v2_gpt4o <- run_prompt_set(
  article_texts = article_texts,
  prompt_template = prompt_v2,
  model_name = main_model
)

#Prompt V3 on main
results_v3_gpt4o <- run_prompt_set(
  article_texts = article_texts,
  prompt_template = prompt_v3,
  model_name = main_model
)

# ============================================================================
# STEP 7: Robustness: preferred prompt across two models
# ============================================================================

#Running V3 on GPT Mini
results_v3_gpt4omini <- run_prompt_set(
  article_texts = article_texts,
  prompt_template = prompt_v3,
  model_name = mini_model
)

# Running V3 on OLLAMA
results_v3_ollama <- run_prompt_set_ollama(
  article_texts = article_texts,
  prompt_template = prompt_v3,
  model_name = other_model
)

# ============================================================================
# STEP 8: Parse structured outputs for comparison
# ============================================================================

parse_results <- function(response_text) {
  # forward-looking
  if (grepl("forward_looking:\\s*Yes", response_text, ignore.case = TRUE)) {
    forward_looking <- "Yes"
  } else if (grepl("forward_looking:\\s*No", response_text, ignore.case = TRUE)) {
    forward_looking <- "No"
  } else {
    forward_looking <- NA
  }
  
  # sentiment
  if (grepl("sentiment:\\s*Bullish", response_text, ignore.case = TRUE)) {
    sentiment <- "Bullish"
  } else if (grepl("sentiment:\\s*Bearish", response_text, ignore.case = TRUE)) {
    sentiment <- "Bearish"
  } else if (grepl("sentiment:\\s*Neutral", response_text, ignore.case = TRUE)) {
    sentiment <- "Neutral"
  } else {
    sentiment <- NA
  }
  
  # category
  category <- sub(".*category:\\s*", "", response_text, ignore.case = TRUE)
  category <- sub("\\n.*", "", category)
  category <- trimws(tolower(category))
  
  if (category == "" || identical(category, tolower(response_text))) {
    category <- NA
  }
  
  data.frame(
    forward_looking = forward_looking,
    sentiment = sentiment,
    category = category,
    stringsAsFactors = FALSE
  )
}

# Apply parser to each set of results
parsed_v2 <- do.call(rbind, lapply(results_v2_gpt4o, parse_results))
parsed_v3 <- do.call(rbind, lapply(results_v3_gpt4o, parse_results))
parsed_v3_mini <- do.call(rbind, lapply(results_v3_gpt4omini, parse_results))
parsed_v3_ollama <- do.call(rbind, lapply(results_v3_ollama, parse_results))

# ============================================================================
# STEP 9: Analyze prompt quality
# ============================================================================

parse_rate <- function(x) {
  mean(!is.na(x))
}

#Prompt Quality Comparison
cat("\nExample V1 outputs:\n")
for (i in 1:min(3, length(results_v1_gpt4o))) {
  cat("\nArticle", i, "\n")
  cat(results_v1_gpt4o[i], "\n")
}

cat("\nParse success for V2:\n")
cat("Forward-looking:", round(100 * parse_rate(parsed_v2$forward_looking), 1), "%\n")
cat("Sentiment:", round(100 * parse_rate(parsed_v2$sentiment), 1), "%\n")
cat("Category:", round(100 * parse_rate(parsed_v2$category), 1), "%\n")

cat("\nParse success for V3:\n")
cat("Forward-looking:", round(100 * parse_rate(parsed_v3$forward_looking), 1), "%\n")
cat("Sentiment:", round(100 * parse_rate(parsed_v3$sentiment), 1), "%\n")
cat("Category:", round(100 * parse_rate(parsed_v3$category), 1), "%\n")

cat("\nV2 sentiment:\n")
print(table(parsed_v2$sentiment, useNA = "ifany"))

cat("\nV3 sentiment:\n")
print(table(parsed_v3$sentiment, useNA = "ifany"))

cat("\nV2 forward-looking:\n")
print(table(parsed_v2$forward_looking, useNA = "ifany"))

cat("\nV3 forward-looking:\n")
print(table(parsed_v3$forward_looking, useNA = "ifany"))

cat("\nV2 category:\n")
print(table(parsed_v2$category, useNA = "ifany"))

cat("\nV3 category:\n")
print(table(parsed_v3$category, useNA = "ifany"))

# ============================================================================
# STEP 10: Robustness of prompt V3
# ============================================================================

agree_sent_mini <- mean(parsed_v3$sentiment == parsed_v3_mini$sentiment, na.rm = TRUE)
agree_fl_mini <- mean(parsed_v3$forward_looking == parsed_v3_mini$forward_looking, na.rm = TRUE)
agree_cat_mini <- mean(parsed_v3$category == parsed_v3_mini$category, na.rm = TRUE)

cat("\nGPT-4o vs GPT-4o-mini:\n")
cat("Sentiment agreement:", round(100 * agree_sent_mini, 1), "%\n")
cat("Forward-looking agreement:", round(100 * agree_fl_mini, 1), "%\n")
cat("Category agreement:", round(100 * agree_cat_mini, 1), "%\n")

agree_sent_ollama <- mean(parsed_v3$sentiment == parsed_v3_ollama$sentiment, na.rm = TRUE)
agree_fl_ollama <- mean(parsed_v3$forward_looking == parsed_v3_ollama$forward_looking, na.rm = TRUE)
agree_cat_ollama <- mean(parsed_v3$category == parsed_v3_ollama$category, na.rm = TRUE)

cat("\nGPT-4o vs Ollama:\n")
cat("Sentiment agreement:", round(100 * agree_sent_ollama, 1), "%\n")
cat("Forward-looking agreement:", round(100 * agree_fl_ollama, 1), "%\n")
cat("Category agreement:", round(100 * agree_cat_ollama, 1), "%\n")

# ============================================================================
# STEP 11: Summary tables
# ============================================================================

summary_table <- data.frame(
  setup = c("V2_gpt4o", "V3_gpt4o", "V3_gpt4o_mini", "V3_ollama"),
  
  pct_forward = c(
    mean(parsed_v2$forward_looking == "Yes", na.rm = TRUE),
    mean(parsed_v3$forward_looking == "Yes", na.rm = TRUE),
    mean(parsed_v3_mini$forward_looking == "Yes", na.rm = TRUE),
    mean(parsed_v3_ollama$forward_looking == "Yes", na.rm = TRUE)
  ),
  
  pct_bullish = c(
    mean(parsed_v2$sentiment == "Bullish", na.rm = TRUE),
    mean(parsed_v3$sentiment == "Bullish", na.rm = TRUE),
    mean(parsed_v3_mini$sentiment == "Bullish", na.rm = TRUE),
    mean(parsed_v3_ollama$sentiment == "Bullish", na.rm = TRUE)
  ),
  
  pct_bearish = c(
    mean(parsed_v2$sentiment == "Bearish", na.rm = TRUE),
    mean(parsed_v3$sentiment == "Bearish", na.rm = TRUE),
    mean(parsed_v3_mini$sentiment == "Bearish", na.rm = TRUE),
    mean(parsed_v3_ollama$sentiment == "Bearish", na.rm = TRUE)
  ),
  
  pct_neutral = c(
    mean(parsed_v2$sentiment == "Neutral", na.rm = TRUE),
    mean(parsed_v3$sentiment == "Neutral", na.rm = TRUE),
    mean(parsed_v3_mini$sentiment == "Neutral", na.rm = TRUE),
    mean(parsed_v3_ollama$sentiment == "Neutral", na.rm = TRUE)
  ),
  
  agree_sent_main = c(
    NA,
    NA,
    mean(parsed_v3$sentiment == parsed_v3_mini$sentiment, na.rm = TRUE),
    mean(parsed_v3$sentiment == parsed_v3_ollama$sentiment, na.rm = TRUE)
  ),
  
  agree_forward_main = c(
    NA,
    NA,
    mean(parsed_v3$forward_looking == parsed_v3_mini$forward_looking, na.rm = TRUE),
    mean(parsed_v3$forward_looking == parsed_v3_ollama$forward_looking, na.rm = TRUE)
  )
)

# Side by side comparison 
comparison <- data.frame(
  article_id = seq_along(article_texts),
  date = sample_articles$article_date,
  v2_forward = parsed_v2$forward_looking,
  v2_sentiment = parsed_v2$sentiment,
  v2_category = parsed_v2$category,
  v3_forward = parsed_v3$forward_looking,
  v3_sentiment = parsed_v3$sentiment,
  v3_category = parsed_v3$category,
  mini_forward = parsed_v3_mini$forward_looking,
  mini_sentiment = parsed_v3_mini$sentiment,
  mini_category = parsed_v3_mini$category,
  ollama_forward = parsed_v3_ollama$forward_looking,
  ollama_sentiment = parsed_v3_ollama$sentiment,
  ollama_category = parsed_v3_ollama$category,
  stringsAsFactors = FALSE
)

cat("\nSummary table:\n")
print(summary_table)

# ============================================================================
# STEP 12: Save outputs
# ============================================================================

write.csv(comparison, "LLM_comparison_results.csv", row.names = FALSE)
write.csv(summary_table, "LLM_summary_table.csv", row.names = FALSE)
writeLines(results_v1_gpt4o, "raw_results_v1_gpt4o.txt")
writeLines(results_v2_gpt4o, "raw_results_v2_gpt4o.txt")
writeLines(results_v3_gpt4o, "raw_results_v3_gpt4o.txt")
writeLines(results_v3_gpt4omini, "raw_results_v3_gpt4omini.txt")
writeLines(results_v3_ollama, "raw_results_v3_ollama.txt")

cat("\n========================================\n")
cat("DONE\n")
cat("Saved files:\n")
cat("- LLM_comparison_results.csv\n")
cat("- LLM_summary_table.csv\n")
cat("- raw_results_v1_gpt4o.txt\n")
cat("- raw_results_v2_gpt4o.txt\n")
cat("- raw_results_v3_gpt4o.txt\n")
cat("- raw_results_v3_gpt4omini.txt\n")
cat("- raw_results_v3_ollama.txt\n")
cat("========================================\n")

# ============================================================================
# QUESTION 6: Evaluation of Equity News Using Local LLM
# ============================================================================
cat("\n============================================================\n")
cat("  QUESTION 6: Evaluation of Equity News Using Local LLM\n")
cat("============================================================\n")

# Getting the equity articles set created in question 3
articles <- equity_articles

# ============================================================================
# STEP 1: Sampling 10 articles per week from 2020, not published in weekends
# ============================================================================

# Set random seed to ensure reproducibility
set.seed(42)
# Filter the articles to only get articles from 2020
articles_2020 <- articles %>% filter(year(Pdate) == 2020)
# Filter the articles to only get articles not published on weekend days
articles_2020_weekdays <- articles_2020 %>% filter(!wday(articles_2020$Pdate) %in% c(1,7))
# Add a column containing the weeknumber in which the article was published (january 1 till 7 is week 1, january 8 till 14 is week 2 etc.)
articles_2020_weekdays_2 <- articles_2020_weekdays %>% mutate(week = floor((as.numeric(as.Date(Pdate) - as.Date("2020-01-01"))) / 7) + 1)
# Sample 10 articles from each week, from week 1 till 52 (to get 520 articles)
articles_selected <- articles_2020_weekdays_2 %>% filter(week<=52) %>% group_by(week) %>% slice_sample(n=10)

# ============================================================================
# STEP 2: Testing different local LLMs
# ============================================================================

# Testing 5 different models on output speed using a local Ollama connection
testing_models <- c("qwen3:4b","deepseek-r1:8b","gemma3:12b","gpt-oss:20b","qwen3-vl:30b")

test_model_speed <- function(model){
  
  prompt <- "Write a short paragraph on equity markets"
  url <- "http://localhost:11434/api/generate"
  
  body <- list(
    model = model,
    prompt = prompt,
    stream = FALSE,
    options = list(temperature = 0.0)
  )
  
  response <- POST(
    url = url,
    body = toJSON(body, auto_unbox = TRUE),
    content_type_json(),
    encode = "json"
  )
  
  result <- content(response, as = "text", encoding = "UTF-8")
  parsed <- fromJSON(result)
  
  tokens <- parsed$eval_count  # number of tokens generated
  duration_seconds <- parsed$eval_duration / 1000000000 # duration of output, adjusted from nanoseconds to seconds
  output_speed <- tokens / duration_seconds
  
  return(output_speed)
}

# Get a table with the results of every tested LLM and their output speed
output_speeds <- sapply(testing_models, test_model_speed)
output_speed_table <- data.frame(Model = testing_models, Tokens_per_second = output_speeds)
output_speed_table

# Chosen model
chosen_model <- "deepseek-r1:8b"

# ============================================================================
# STEP 3: Run the prompt with the articles in the chosen LLM
# ============================================================================

# Get the chosen prompt
chosen_prompt <- prompt_v3 

# Run the prompt with each article locally on the chosen model in Ollama
analyze_article <- function(article, model = chosen_model) {
  
  # Build the prompt
  prompt <- paste0(chosen_prompt, article)
  
  # Ollama API endpoint
  url <- "http://localhost:11434/api/generate"
  
  # Prepare the request body
  body <- list(
    model = model,
    prompt = prompt,
    stream = FALSE,
    options = list(temperature = 0.0)
  )
  
  # Send POST request
  response <- httr::POST(
    url = url,
    body = jsonlite::toJSON(body, auto_unbox = TRUE),
    httr::content_type_json(),
    encode = "json"
  )
  
  # Parse the response
  result <- httr::content(response, as = "text", encoding = "UTF-8")
  parsed <- jsonlite::fromJSON(result)
  
  # Return the response text
  return(parsed$response)
}

# Run the function to get the results
answers <- vector("list", nrow(articles_selected))
for (i in 117:nrow(articles_selected)){
  article <- articles_selected$Fulltxt[i]
  answers[i] <- analyze_article(article, chosen_model)
  print(paste("Number of articles processed", i))
}

# Create a dataframe that combines the answers with the weeknumber of the article
df_answers_weeknumber <- data.frame(week = rep(1:52, each=10), answer = sapply(answers, paste, collapse = " "), stringsAsFactors = FALSE)

# Add column forward_looking to the dataframe (yes/no)
df_answers_weeknumber$forward_looking <- sub(".*forward_looking: ([^\n]+).*", "\\1", df_answers_weeknumber$answer)
# Add column sentiment to the dataframe (Bullish, Neutral, Bearish)
df_answers_weeknumber$sentiment <- sub(".*sentiment: ([^\n]+).*", "\\1", df_answers_weeknumber$answer)
# Add column category to the dataframe (one of the categories defined in question 5)
df_answers_weeknumber$category <- sub(".*category: ([^\n]+).*", "\\1", df_answers_weeknumber$answer)

# Clean rows that did not give an interpretable answer
df_answers_weeknumber_cleaned <- df_answers_weeknumber %>% filter(grepl("^forward_looking", trimws(.[[2]])))

# ============================================================================
# STEP 4: Percentage of articles per week containing forward-looking information
# ============================================================================

# Create a dataframe with the weeknumbers and the percentage of forward-looking articles for that week
forward_looking_percentage <- df_answers_weeknumber_cleaned %>% mutate(forward_looking = forward_looking == 'Yes') %>% group_by(week) %>% summarise(percentage = round(mean(forward_looking) * 100,2))

# Get statistics from the dataframe
print(mean(forward_looking_percentage$percentage))
print(sd(forward_looking_percentage$percentage))
print(min(forward_looking_percentage$percentage))
print(max(forward_looking_percentage$percentage))

# Plot the percentages 
ggplot(forward_looking_percentage, aes(x = week, y=percentage)) + geom_line(color='blue') + geom_point(color='blue') + labs(title = "Weekly Percentage of Forward-Looking Equity Articles (2020)", x="Week", y="Percentage of articles (%)") + scale_y_continuous(limits = c(0,100), breaks = seq(0,100,10)) + theme_classic() + theme(plot.title = element_text(size = 14, face = "bold"), axis.title = element_text(size = 12), axis.text = element_text(size = 10))

# ============================================================================
# STEP 5: Weekly sentiment index
# ============================================================================

# Add a column to the answer dataframe for sentiment index: 1 if sentiment is Bullish, 0 if sentiment is Neutral, -1 if sentiment is Bearish
df_answers_weeknumber_cleaned <- df_answers_weeknumber_cleaned %>% mutate(equity_sentiment = case_when(sentiment == 'Bullish' ~ 1, sentiment == 'Neutral' ~ 0, sentiment == 'Bearish' ~ -1), .after = sentiment)
# Create new dataframe consisting of weeknumber and the average sentiment index of that week
sentiment_index <- df_answers_weeknumber_cleaned %>% group_by(week) %>% summarise(sentiment_index = round(mean(equity_sentiment),2))

# Get statistics from the dataframe
print(mean(sentiment_index$sentiment_index))
print(sd(sentiment_index$sentiment_index))
print(min(sentiment_index$sentiment_index))
print(max(sentiment_index$sentiment_index))

# Plot the sentiment index
ggplot(sentiment_index, aes(x = week, y = sentiment_index)) + geom_line(color='blue') + geom_point(color='blue') + labs(title="Weekly sentiment index (2020)", x="Week", y="Sentiment Index") + scale_y_continuous(limits = c(-1,1), breaks = seq(-1,1,0.25))+ theme_classic() + theme(plot.title = element_text(size = 14, face = "bold"), axis.title = element_text(size = 12), axis.text = element_text(size = 10))

# ============================================================================
# STEP 6: Time-varying incidence of the different topics
# ============================================================================

# Create new dataframe consisting of for every topic of every week the number of articles in that category
topics_share <- df_answers_weeknumber_cleaned %>% group_by(week, category) %>% summarise(n = n(),.groups='drop')

# Fill dataframe with the topics that are missing each week with the value of 0 for the frequency (necessary for a plot)
topics_share <- topics_share %>% complete(week, category, fill = list(n=0))
# Add a column share to the dataframe which consists of the share that the category has in each week 
#(not similar to the frequency as not all weeks have 10 articles after the cleaning)
topic_share <- topics_share %>% group_by(week) %>% mutate(share = n / sum(n))

# Plot the different category shares for each week
ggplot(topics_share, aes(x=week, y=n, color=category)) + geom_line() + labs(title="Time-varying incidence of topics", x="Week", y="Number of articles", color="Topic") + theme_classic()

# Get summary statistics for each category
topic_summary <- topic_share %>% group_by(category) %>% summarise(mean_share = mean(share), min_share = min(share), max_share = max(share)) %>% arrange(desc(mean_share))
topic_summary

# =====================================================================================
# QUESTION 7: Exploratory analysis of Investor Profile Effects on LLM-Based Predictions
# =====================================================================================
cat("\n========================================================================================\n")
cat("  QUESTION 7: Exploratory analysis of Investor Profile Effects on LLM-Based Predictions\n")
cat("===========================================================================================\n")

# ============================================================================
# STEP 1: Data preparation and sampling
# ============================================================================

# Run the Ollama API
ollama_url  <- "http://localhost:11434/api/generate"
# Select a model
model_name  <- "llama3.1:8b"

# Load the data and label the title and lead text into a single field
articles <- equity_articles |>
  mutate(
    article_id = row_number(),
    news_text  = paste(as.character(Title), LeadPar),
    NS         = as.character(NS)
  )

# Store column names as variables to be used later
id_col   <- "article_id"
text_col <- "news_text"

# Set random seed to ensure reproducibility
set.seed(42)

# Sample size is set to 15
n_articles <- 15
# Sampel 15 random articles from the equity articles set
articles_sample <- articles |> slice_sample(n = n_articles)

# Show sampled articles
articles_sample |>
  select(article_id, news_text) |>
  mutate(news_text = str_trunc(news_text, 80)) |>
  print()

# ============================================================================
# STEP 2: Create investor profiles
# ============================================================================

# Create investor profiles for 6 dimensions: age, gender, race, income, education, political orientation

profiles <- list(
  
  # --- Age ---
  young_male =
    "You are a 24-year-old male investor with moderate income and a short investment horizon. You prefer growth stocks.",
  
  older_female =
    "You are a 68-year-old female investor focused on capital preservation, dividends, and low-risk bonds.",
  
  # --- Gender ---
  male_midcareer =
    "You are a 45-year-old male investor with high income, a family to support, and a balanced portfolio.",
  
  female_midcareer =
    "You are a 45-year-old female investor with high income, a family to support, and a balanced portfolio.",
  
  # --- Race / Ethnicity ---
  white_investor =
    "You are a white 50-year-old investor from the Midwest with a traditional view of markets and a long-term horizon.",
  
  black_investor =
    "You are a Black 50-year-old investor from the South with awareness of structural economic inequalities and a long-term horizon.",
  
  # --- Income ---
  low_income =
    "You are a 35-year-old low-income investor with limited savings, high sensitivity to inflation and job market news.",
  
  high_income =
    "You are a 35-year-old high-income investor with a large diversified portfolio, mostly insulated from economic shocks.",
  
  # --- Education ---
  college_dropout =
    "You are a 30-year-old investor who did not finish college. You rely on intuition and news headlines rather than financial models.",
  
  phd_economist =
    "You are a 30-year-old investor with a PhD in economics who interprets news through macroeconomic models and data.",
  
  # --- Political orientation ---
  liberal_investor =
    "You are a politically liberal investor who values regulation, climate policy, and social equity when assessing market impact.",
  
  conservative_investor =
    "You are a politically conservative investor who values deregulation, free markets, and fiscal discipline when assessing market impact."
)

# Check total Ollama calls (experiment size) that the sample size and number of investor profiles generates
cat("Sample size:", nrow(articles_sample), "articles\n")
cat("Profiles:   ", length(profiles), "\n")
cat("Total calls:", nrow(articles_sample) * length(profiles), "\n\n")
cat("Selected articles:\n")

# ============================================================================
# STEP 3: Building the prompt
# ============================================================================

# Create LLM prompt
build_prompt <- function(profile_text, news_text) {
  paste(
    profile_text,
    "",
    "You are evaluating the following financial news headline and lead paragraph.",
    "Based on your perspective, will this news have a positive, negative, or neutral impact",
    "on equity markets or affected firms over the next trading day?",
    "",
    'Answer "YES" if a positive impact on stock prices is likely.',
    'Answer "NO" if a negative impact on stock prices is likely.',
    'Answer "UNKNOWN" if the impact is unclear or mixed.',
    "",
    "Instructions:",
    "- First line: write ONLY one of: YES / NO / UNKNOWN",
    "- Second line: one sentence explaining your reasoning from your perspective.",
    "",
    "News:",
    news_text,
    sep = "\n"
  )
}

# ============================================================================
# STEP 3: Building the prompt
# ============================================================================

# Function to sent the prompt to local Ollama model
query_ollama <- function(prompt, model = model_name) {
  body <- list(
    model  = model,
    prompt = prompt,
    stream = FALSE,
    options = list(temperature = 0.2)
  )
  
  resp <- request(ollama_url) |>
    req_headers(`Content-Type` = "application/json") |>
    req_body_json(body, auto_unbox = TRUE) |>
    req_perform()
  
  out <- resp_body_json(resp, simplifyVector = TRUE)
  out$response
}

# ============================================================================
# STEP 4: Parse model output
# ============================================================================

# Turns LLM output into structured data
parse_output <- function(raw_text) {
  lines <- str_split(raw_text, "\n")[[1]] |>
    trimws() |>
    keep(~ .x != "")
  
  first_line  <- ifelse(length(lines) >= 1, toupper(lines[1]), NA_character_)
  explanation <- ifelse(length(lines) >= 2, paste(lines[-1], collapse = " "), NA_character_)
  
  forward <- case_when(
    str_detect(first_line, "^YES")     ~ "Yes",
    str_detect(first_line, "^NO")      ~ "No",
    str_detect(first_line, "^UNKNOWN") ~ "Unknown",
    TRUE ~ NA_character_
  )
  
  sentiment <- case_when(
    forward == "Yes"     ~ "Bullish",
    forward == "No"      ~ "Bearish",
    forward == "Unknown" ~ "Neutral",
    TRUE                 ~ NA_character_
  )
  
  tibble(
    raw_output  = raw_text,
    forward     = forward,
    sentiment   = sentiment,
    explanation = explanation
  )
}

# ============================================================================
# STEP 5: Run the prompt in combination with article and investor profile
# ============================================================================

# Create the results of running the LLM with the prompt, article, and investor profile
results <- map_dfr(names(profiles), function(profile_name) {
  profile_text <- profiles[[profile_name]]
  
  map_dfr(seq_len(nrow(articles_sample)), function(i) {
    article_id <- articles_sample[[id_col]][i]
    news_text  <- articles_sample[[text_col]][i]
    prompt     <- build_prompt(profile_text, news_text)
    
    cat(sprintf("[%s] article %d\n", profile_name, article_id))
    
    raw <- tryCatch(
      query_ollama(prompt),
      error = function(e) paste("ERROR:", e$message)
    )
    
    parsed <- if (str_starts(raw, "ERROR")) {
      tibble(raw_output = raw, forward = NA_character_,
             sentiment = NA_character_, explanation = NA_character_)
    } else {
      parse_output(raw)
    }
    
    bind_cols(
      tibble(article_id = article_id, profile = profile_name, news_text = news_text),
      parsed
    )
  })
})

# Store the results in a csv file
write_csv(results, "ollama_profile_results_long.csv")
cat("\nSaved: ollama_profile_results_long.csv\n")

# Create a table for the result for all investor profiles for each article
results_wide <- results |>
  select(article_id, profile, sentiment) |>
  pivot_wider(names_from = profile, values_from = sentiment)

# Store these results from the table in a csv file
write_csv(results_wide, "ollama_profile_results_wide.csv")


# ============================================================================
# STEP 6: Investor profile behavior
# ============================================================================

# Create the investor profile order that will be used in the table
profile_order <- c("young_male", "older_female","male_midcareer","female_midcareer","white_investor","black_investor","low_income", "high_income","college_dropout","phd_economist","liberal_investor","conservative_investor")

# Create a table with the share of each sentiment classification for each of the investor profiles
profile_behavior <- results |>
  group_by(profile) |>
  summarise(
    bullish = round(mean(sentiment == "Bullish", na.rm = TRUE), 3),
    neutral = round(mean(sentiment == "Neutral", na.rm = TRUE), 3),
    bearish = round(mean(sentiment == "Bearish", na.rm = TRUE), 3),
    .groups = "drop"
  ) |>
  arrange(factor(profile, levels = profile_order))

# Show the table
print(profile_behavior)

# ============================================================================
# STEP 7: General disagreement analysis
# ============================================================================

# Disagreement per article
# Create a table that shows for each article the number of responses for each sentiment and whether there is disagreement
disagreement_per_article <- results |>
  filter(!is.na(forward)) |>
  group_by(article_id, news_text) |>
  summarise(
    n_profiles        = n(),
    n_yes             = sum(forward == "Yes"),
    n_no              = sum(forward == "No"),
    n_unknown         = sum(forward == "Unknown"),
    n_unique_forward  = n_distinct(forward),
    disagreement      = n_unique_forward > 1,
    .groups = "drop"
  ) |>
  arrange(desc(n_unique_forward), desc(n_profiles))

# Store the table information in a csv file
write_csv(disagreement_per_article, "ollama_disagreement_per_article.csv")

# Disagreement rate per profile pair
# Create a variable including all possible profile pairs
profile_pairs <- combn(names(profiles), 2, simplify = FALSE)

# Create a table that shows the disagreement rate for all possible profile pairs
pair_disagreement <- map_dfr(profile_pairs, function(pair) {
  p1 <- pair[1]; p2 <- pair[2]
  
  joined <- results |>
    filter(profile %in% c(p1, p2), !is.na(forward)) |>
    select(article_id, profile, forward) |>
    pivot_wider(names_from = profile, values_from = forward) |>
    filter(!is.na(.data[[p1]]), !is.na(.data[[p2]]))
  
  if (nrow(joined) == 0) return(NULL)
  
  tibble(
    profile_1       = p1,
    profile_2       = p2,
    n_articles      = nrow(joined),
    n_disagree      = sum(joined[[p1]] != joined[[p2]]),
    disagree_rate   = round(n_disagree / n_articles, 3)
  )
}) |>
  arrange(desc(disagree_rate))

# Store the table information in a csv file
write_csv(pair_disagreement, "ollama_pair_disagreement.csv")

# Overall disagreement rate
# Determines the overall disagreement rate for all articles
overall_rate <- mean(disagreement_per_article$disagreement, na.rm = TRUE)
cat(sprintf("\nOverall article-level disagreement rate: %.1f%%\n", overall_rate * 100))

# ============================================================================
# STEP 8: Dimension level disagreement analysis
# ============================================================================

# Tag each profile with its defining dimension
dimension_map <- tribble(
  ~profile,              ~dimension,      ~group,
  "young_male",          "age",           "young",
  "older_female",        "age",           "old",
  "male_midcareer",      "gender",        "male",
  "female_midcareer",    "gender",        "female",
  "white_investor",      "race",          "white",
  "black_investor",      "race",          "black",
  "low_income",          "income",        "low",
  "high_income",         "income",        "high",
  "college_dropout",     "education",     "low_edu",
  "phd_economist",       "education",     "high_edu",
  "liberal_investor",    "politics",      "liberal",
  "conservative_investor","politics",     "conservative"
)

# For each dimension, compute disagreement rate between the two groups
dimension_disagreement <- dimension_map |>
  group_by(dimension) |>
  group_map(function(df, key) {
    grps <- df$group
    p1   <- df$profile[1]
    p2   <- df$profile[2]
    
    joined <- results |>
      filter(profile %in% c(p1, p2), !is.na(forward)) |>
      select(article_id, profile, forward) |>
      pivot_wider(names_from = profile, values_from = forward) |>
      filter(!is.na(.data[[p1]]), !is.na(.data[[p2]]))
    
    if (nrow(joined) == 0) return(NULL)
    
    tibble(
      dimension     = key$dimension,
      profile_A     = p1,
      profile_B     = p2,
      n_articles    = nrow(joined),
      n_disagree    = sum(joined[[p1]] != joined[[p2]]),
      disagree_rate = round(n_disagree / n_articles, 3)
    )
  }) |>
  bind_rows() |>
  arrange(desc(disagree_rate))

# Show disagreement rate by dimension
cat("\n--- Disagreement by dimension ---\n")
print(dimension_disagreement)

# Store the table information in a csv file
write_csv(dimension_disagreement, "ollama_dimension_disagreement.csv")

# ============================================================================
# STEP 9: Interesting cases
# ============================================================================
# Determine the articles where at least one investor profile says Yes and at least one says No
interesting_cases <- results |>
  filter(!is.na(forward)) |>
  group_by(article_id) |>
  filter(any(forward == "Yes") & any(forward == "No")) |>
  ungroup() |>
  select(article_id, news_text, profile, forward, sentiment, explanation) |>
  arrange(article_id, profile)

# Store the information in a csv file
write_csv(interesting_cases, "ollama_interesting_cases.csv")
# Print the number of articles that are considered interesting cases
cat(sprintf("\nArticles with YES vs NO split: %d\n", n_distinct(interesting_cases$article_id)))

# ============================================================================
# STEP 10: Print example cases
# ============================================================================

# Determine the top 3 most contested articles in which at least one profile said yes and one said no
top_contested <- disagreement_per_article |>
  filter(n_yes >= 1, n_no >= 1) |>
  arrange(desc(n_yes + n_no)) |>
  slice_head(n = 3)

# Create a function that loops over contested articles and prints article and each profile's prediction
cat("\n========== TOP CONTESTED ARTICLES ==========\n")
for (i in seq_len(nrow(top_contested))) {
  # Selects the article label and the news
  aid <- top_contested$article_id[i]
  cat(sprintf("\n[Article %d]\n", aid))
  cat(str_wrap(top_contested$news_text[i], width = 80), "\n\n")
  
  # Selects all model responses for the current contested article
  results |>
    filter(article_id == aid, !is.na(forward)) |>
    select(profile, forward, explanation) |>
    arrange(forward) |>
    pwalk(function(profile, forward, explanation) {
      cat(sprintf("  %-25s  %s\n", profile, forward))
      cat(sprintf("    -> %s\n", str_wrap(explanation, width = 70, exdent = 7)))
    })
}

# ============================================================================
# STEP 11: Visualization
# ============================================================================
# Heatmap
# Create sentiment matrix that shows combination of each article and investor profile the sentiment (used for plotting)
sentiment_matrix <- results |>
  filter(!is.na(sentiment)) |>
  mutate(
    article_label = str_trunc(news_text, 40),
    sentiment_num = case_when(
      sentiment == "Bullish" ~ 1,
      sentiment == "Neutral" ~ 0,
      sentiment == "Bearish" ~ -1
    )
  )

# Create a heatmap for each combination of article and investor profile
p_heatmap <- ggplot(sentiment_matrix,
                    aes(x = profile, y = reorder(article_label, article_id),
                        fill = factor(sentiment_num))) +
  geom_tile(color = "white", linewidth = 0.4) +
  # Bearish is red, Neutral is yellow, Bullish is green
  scale_fill_manual(
    values = c("-1" = "#d73027", "0" = "#ffffbf", "1" = "#1a9850"),
    labels = c("-1" = "Bearish", "0" = "Neutral", "1" = "Bullish"),
    name   = "Sentiment"
  ) +
  theme_minimal(base_size = 9) +
  theme(
    axis.text.x  = element_text(angle = 45, hjust = 1),
    axis.text.y  = element_text(size = 7),
    panel.grid   = element_blank()
  ) +
  labs(
    title    = "Investor Profile × Article Sentiment",
    subtitle = "Green = Bullish | Yellow = Neutral | Red = Bearish",
    x = "Investor Profile", y = "Article"
  )

# Save the heatmap as a png
ggsave("ollama_heatmap.png", p_heatmap, width = 12, height = 8, dpi = 150)
cat("\nSaved: ollama_heatmap.png\n")

# Bar chart
# Create a bar chart for the disagreement rate for each dimension
p_dim <- ggplot(dimension_disagreement,
                aes(x = reorder(dimension, disagree_rate), y = disagree_rate)) +
  geom_col(fill = "#7aa6a1") +
  coord_flip() +
  scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
  labs(
    title = "Disagreement Rate by Investor Dimension",
    subtitle = "Share of articles where the two profile variants gave different signals",
    x = "Dimension", y = "Disagreement Rate"
  ) +
  theme_classic()

# Save the bar chart as a png
ggsave("ollama_dimension_disagreement.png", p_dim, width = 7, height = 4, dpi = 150)
cat("Saved: ollama_dimension_disagreement.png\n")

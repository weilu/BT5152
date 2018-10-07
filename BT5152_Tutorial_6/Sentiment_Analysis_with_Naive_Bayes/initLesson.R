DATA_DIR <- file.path("data")
IMDB_URL <- "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_LOCAL_DEST <- file.path(DATA_DIR, "aclImdb_v1.tar.gz")

dir.create(DATA_DIR, showWarnings = FALSE)

if(!file.exists(IMDB_LOCAL_DEST)) {
  download.file(url = IMDB_URL, destfile = IMDB_LOCAL_DEST)
  untar(tarfile = IMDB_LOCAL_DEST, exdir = DATA_DIR)
}
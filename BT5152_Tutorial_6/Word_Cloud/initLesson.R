# Code placed in this file fill be executed every time the
      # lesson is started. Any variables created here will show up in
      # the user's working directory and thus be accessible to them
      # throughout the lesson.

DATA_DIR <- file.path("data")
IMDB_URL <- "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_LOCAL_DEST <- file.path(DATA_DIR, "aclImdb_v1.tar.gz")

dir.create(DATA_DIR, showWarnings = FALSE)

if(!file.exists(IMDB_LOCAL_DEST)) {
  download.file(url = IMDB_URL, destfile = IMDB_LOCAL_DEST)
  untar(tarfile = IMDB_LOCAL_DEST, exdir = DATA_DIR)
}
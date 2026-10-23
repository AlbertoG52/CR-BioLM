
lib <- Sys.getenv("R_LIBS_USER"); .libPaths(c(lib, .libPaths()))
suppressMessages(library(lme4))
d <- read.csv(commandArgs(TRUE)[1], encoding = "UTF-8")
d <- d[d$arm %in% c("D", "F", "A1F", "CF"), ]
d$arm <- factor(d$arm, levels = c("A1F", "CF", "D", "F"))
d$lrec <- log1p(d$n_records)
d$lglob <- log10(d$n_global)
ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
full <- glmer(correct ~ arm * lrec + lglob + (1 | species) + (1 | qtype),
              data = d, family = binomial, control = ctl)
null <- glmer(correct ~ arm + lrec + lglob + (1 | species) + (1 | qtype),
              data = d, family = binomial, control = ctl)
lrt <- anova(null, full)
co <- summary(full)$coefficients
v <- vcov(full)
slope <- function(arm) {
  b <- "lrec"; k <- paste0("arm", arm, ":lrec")
  if (arm == "A1F") return(c(co[b, 1], sqrt(v[b, b])))
  est <- co[b, 1] + co[k, 1]
  se <- sqrt(v[b, b] + v[k, k] + 2 * v[b, k])
  c(est, se)
}
out <- list(n = nrow(d), species = length(unique(d$species)),
            lrt_chisq = lrt$Chisq[2], lrt_df = lrt$Df[2], lrt_p = lrt$`Pr(>Chisq)`[2],
            global_slope = c(co["lglob", 1], co["lglob", 2], co["lglob", 4]),
            slopes = lapply(setNames(c("A1F", "CF", "D", "F"), c("A1F", "CF", "D", "F")), slope))
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 6))

# Originality and similarity self-check

This is a local check, not a similarity-engine report. It covers what
can be verified offline and what actually drives a similarity score.
It makes no claim about what a commercial detector will return.

## 1. Overlap with the round-1 manuscript

- Revised manuscript: **19,106 words**
- Round-1 manuscript: **10,214 words**
- Shared 12-word sequences: **5,058** (**26.6%** of the revision)
- Verbatim runs of 25+ words retained from round 1: **202**

Retained text is concentrated in the introduction, related work and
ethics sections, which the revision did not need to change. This
overlap is with the authors' own unpublished submission. It will
register in a similarity report only if that text is already public,
for example as a preprint or thesis chapter. If it is, disclose it to
the editor rather than rewriting working prose to defeat a matcher.

## 2. Internal duplication

- Repeated 12-word sequences within the revision: **101**

Most frequent:
  - (2x) a graph feature defined as the shortest path to a confirmed illicit
  - (2x) graph feature defined as the shortest path to a confirmed illicit node
  - (2x) feature defined as the shortest path to a confirmed illicit node was
  - (2x) defined as the shortest path to a confirmed illicit node was found
  - (2x) as the shortest path to a confirmed illicit node was found to

Internal repetition inflates a similarity score and usually means
copy-paste during editing. A handful of repeats is normal where a
result is restated in the abstract, results and conclusion.

## 3. Quoted material

- Quoted passages in the source: **1**
  - "DGI pre-training followed by GAT fine-tuning"

## 4. Attribution

- Citation commands: **54** across **308** substantive
  paragraphs
- All 30 bibliography entries are cited; no orphans, no dangling keys
  (verified by `research/src/tex_audit.py`)

## 5. Externally sourced material and how it is handled

Two external sources inform the revision, and neither is reproduced
verbatim.

- **Detection rules** are taken from the Dumplings tool. The
  conditions are restated in the authors' own prose as numeric
  thresholds and script-type constraints, not copied as code or text,
  and the source is named in the methodology and the response letter.
- **Per-output transaction values** are factual blockchain records
  retrieved from public explorers. Facts of this kind are not
  copyrightable and carry no similarity risk; the retrieval procedure
  is described and the explorers are named.

## 6. On AI-detection tools

No attempt has been made, and none should be made, to alter this text
so that it scores differently on an AI-detection tool. Those tools are
unreliable in both directions, and editing prose to defeat one
degrades the writing without changing whether the work is sound.

The defensible position is the one this manuscript is in: every number
comes from a released script, every experiment has a commit hash, the
data are public or described, and the prose states what the
experiments found. If originality is questioned, that record answers
it, and a detector score does not.

## 7. What this check does not cover

- Matching against published literature, subscription databases or the
  open web. Only a similarity service can do that.
- Paraphrase that preserves another source's structure without
  matching its words.

Run the publisher's own similarity check before final acceptance and
read the report, treating self-overlap with prior work separately from
overlap with third-party sources.
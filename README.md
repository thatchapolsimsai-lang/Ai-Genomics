---
title: Ai-Genomics
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: "1.63.0"
app_file: app.py
pinned: false
---

# Ai-Genomics

AI-assisted analysis of DNA/RNA sequencing data (FASTA/FASTQ), with a
persistent-history chat assistant for Open Access biology research
alongside a Biopython-based sequence analyzer.

## Configuration

Set these as Space secrets (Settings → Variables and secrets) rather
than committing them:

- `GOOGLE_API_KEY` — for the Gemini model
- `NINEARM_API_KEY`, `NINEARM_BASE_URL` — for the Claude (reseller) model
- `NCBI_API_KEY` (optional) — raises NCBI E-utilities rate limits
- `NCBI_CONTACT_EMAIL` (optional) — contact address NCBI asks E-utilities callers to identify with

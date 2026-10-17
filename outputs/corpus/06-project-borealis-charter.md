# Project Borealis charter

Document ID: DOC-006
Company: Aster Vale Investments (fictional)
Version: 0.1-draft
Updated: 2026-10-16
Source owner: Risk Analytics (fictional)

All entities and events in this document are fictional training data.

## Purpose

Project Borealis makes internal risk-reporting dependencies understandable. It helps analysts trace a reporting input back through upstream operational systems, identify the owner of that input, and locate supporting review records. This is operational lineage work, not a fund-performance forecasting system.

## Ownership

Sofia Alvarez is the project lead for Project Borealis. Daniel Ortiz is its executive sponsor. Sofia coordinates the lineage inventory and review work; Daniel handles executive escalations.

## Team and tools

Sofia Alvarez and Alex Rivera work on Project Borealis. They use Signal Lineage Board to record reporting dependencies and source lineage. Their project roles do not assign Alex Morgan to Borealis; Alex Morgan and Alex Rivera are distinct people.

## Dependencies

Borealis depends on Project Harbor's access-review records. Harbor Access Register is registered as an upstream operational source for Borealis. DEC-003 documents who approved that source registration. Borealis uses Signal Lineage Board for lineage; Harbor Access Register remains the upstream access-review source, not the lineage-board replacement.

## Scope boundaries

The project tracks internal reporting relationships. It contains no portfolio holdings, client identities, expected fund returns, or guaranteed performance figures. Source registration does not establish a completed production deployment.

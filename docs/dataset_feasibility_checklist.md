# Dataset Feasibility Checklist

> Complete this after downloading and inspecting the dataset.  
> **Gate 0 is not passed until all REQUIRED items are confirmed.**

## Instructions

Run `python scripts/audit_data.py --data-dir data/raw --out reports/audit.json` first,  
then fill in this checklist from `reports/audit_summary.txt` and your own inspection.

---

## A. Access and Legal

- [ ] **REQUIRED** Dataset is downloaded locally
- [ ] **REQUIRED** License/terms reviewed and understood
- [ ] **REQUIRED** Redistribution restriction confirmed — raw files will NOT be committed
- [ ] **REQUIRED** Derived results and code may be published (or restriction is documented)
- [ ] Citation added to `docs/dataset_card.md`

## B. File Inventory

- [ ] All downloaded files listed in `docs/dataset_card.md` → Section 4
- [ ] File formats confirmed (CSV / Parquet / XML / other): ___
- [ ] Total size on disk: ___

## C. Schema

- [ ] **REQUIRED** Patient/subject identifier column identified: ___
- [ ] **REQUIRED** Timestamp column identified: ___
- [ ] **REQUIRED** Glucose value column identified: ___
- [ ] Glucose units confirmed: mg/dL / mmol/L / mixed
- [ ] Optional context columns present and understood: ___
- [ ] Columns assumed but NOT present documented

## D. Temporal Integrity

- [ ] Timestamp format parsed correctly
- [ ] Time zone / UTC offset confirmed: ___
- [ ] Records are sortable by (patient, timestamp)
- [ ] Duplicate timestamps quantified: N = ___
- [ ] Duplicate resolution policy chosen and documented

## E. Sampling Cadence

- [ ] **REQUIRED** Median inter-reading interval confirmed: ___ minutes
- [ ] Actual cadence distribution inspected (not assumed)
- [ ] Number of 5-min intervals in 120-minute horizon: ___ (should be ≈24 if cadence is 5 min)
- [ ] Long gaps (>30 min) quantified: N gaps = ___

## F. Sequence Availability

- [ ] **REQUIRED** Number of patients with usable data: ___
- [ ] **REQUIRED** Minimum sequence length per patient to form one valid window: ___ hours
- [ ] Median sequence length: ___ hours
- [ ] Estimated number of valid 120-min forecast windows: ___
- [ ] Windows are sufficient for train/val/test split: Yes / No

## G. Target Construction

- [ ] **REQUIRED** A glucose reading within ± tolerance of t+120 exists for a meaningful fraction of potential windows
- [ ] Tolerance chosen: ± ___ minutes (must be justified by cadence)
- [ ] Target is never included in the input window (confirmed by code review)
- [ ] Estimated window rejection rate due to missing target: ___

## H. Split Design

- [ ] Chronological split design chosen (earlier → train, later → test)
- [ ] Patient-level split or time-level split — decision documented with justification
- [ ] No patient appears in both train and test time periods in a leakage-creating way
- [ ] Validation set defined separately from test set

## I. Data Quality

- [ ] Missingness rate per column documented
- [ ] Physiologically impossible values: N = ___; disposition documented
- [ ] Unit anomalies (mmol/L mixed with mg/dL): N = ___

## J. Decision

- [ ] **GO** — Dataset supports 120-min forecasting; proceed to Phase 2
- [ ] **NARROW** — Target is supportable with modifications: ___
- [ ] **SWITCH** — Dataset insufficient; switch to: ___
- [ ] **SYNTHETIC** — No suitable real dataset accessible; proceed with clearly labeled synthetic prototype

**Decision made by:** ___  
**Date:** ___  
**Rationale:** ___

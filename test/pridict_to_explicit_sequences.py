#!/usr/bin/env python3
"""
Convert PRIDICT-style encoded prime editing data into explicit, separated
biological sequences.

PRIDICT encodes the pegRNA–target interface as two 99-nt DNA strings
(initial / mutated) plus four location masks.  This script decomposes that
representation into the distinct biological components a researcher
would recognise:

    Target DNA (both strands)
    Spacer / guide RNA
    PAM
    PBS  (on the pegRNA, i.e. reverse-complement of the nicked strand)
    RTT  (on the pegRNA, i.e. reverse-complement of the desired edited strand)
    Full pegRNA  (spacer + scaffold + RTT + PBS)
    Edit description  (type, size, position)

Usage
-----
    python pridict_to_explicit_sequences.py --data <csv> [--rows 0,1,2]

If --rows is omitted the first 5 rows are processed.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

SCAFFOLD = "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC"


def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1]


def dna_to_rna(seq: str) -> str:
    return seq.replace("T", "U").replace("t", "u")


@dataclass
class PrimeEditDesign:
    """Holds the decomposed prime-editing design."""

    # --- raw PRIDICT fields ---
    initial_sequence: str
    mutated_sequence: str
    protospacer_loc: Tuple[int, int]
    pbs_loc: Tuple[int, int]
    rt_init_loc: Tuple[int, int]
    rt_mut_loc: Tuple[int, int]

    # --- derived fields (populated by .decompose()) ---
    target_strand_5to3: str = ""
    nontarget_strand_5to3: str = ""
    nick_position: int = -1
    pam_seq: str = ""
    pam_position: Tuple[int, int] = (0, 0)

    spacer_dna: str = ""
    spacer_rna: str = ""

    pbs_on_nicked_strand: str = ""
    pbs_on_pegrna: str = ""

    rtt_wt: str = ""
    rtt_edited: str = ""
    rtt_on_pegrna: str = ""

    pegrna_sequence: str = ""

    edit_type: str = ""
    edit_size: int = 0
    edit_position_in_rtt: int = -1
    edit_wt_bases: str = ""
    edit_mut_bases: str = ""

    upstream_flank: str = ""
    downstream_flank: str = ""

    def decompose(self) -> "PrimeEditDesign":
        seq = self.initial_sequence
        mut = self.mutated_sequence
        p_s, p_e = self.protospacer_loc
        b_s, b_e = self.pbs_loc
        ri_s, ri_e = self.rt_init_loc
        rm_s, rm_e = self.rt_mut_loc

        # The displayed sequence is the non-target (nicked / PAM) strand, 5'→3'.
        self.nontarget_strand_5to3 = seq
        self.target_strand_5to3 = reverse_complement(seq)

        # Nick site: between last PBS position and first RTT position
        self.nick_position = b_e  # nick is *after* this position on the nicked strand

        # PAM (GG) immediately 3' of protospacer on non-target strand
        self.pam_position = (p_e + 1, p_e + 2)
        self.pam_seq = seq[self.pam_position[0]: self.pam_position[1] + 1]

        # Spacer = protospacer on non-target strand (same sequence as guide RNA)
        self.spacer_dna = seq[p_s: p_e + 1]
        self.spacer_rna = dna_to_rna(self.spacer_dna)

        # PBS on the nicked (non-target) strand — the 3' free end after nicking
        self.pbs_on_nicked_strand = seq[b_s: b_e + 1]
        # PBS on pegRNA = reverse complement (anneals to the 3' free end)
        self.pbs_on_pegrna = reverse_complement(self.pbs_on_nicked_strand)

        # RTT: wild-type region on initial sequence
        self.rtt_wt = seq[ri_s: ri_e + 1]
        # RTT: desired edited region on mutated sequence
        self.rtt_edited = mut[rm_s: rm_e + 1]
        # RTT on pegRNA = reverse complement of the desired edited sequence
        self.rtt_on_pegrna = reverse_complement(self.rtt_edited)

        # Full pegRNA (5'→3'): spacer + scaffold + RTT + PBS
        self.pegrna_sequence = (
            dna_to_rna(self.spacer_dna)
            + dna_to_rna(SCAFFOLD)
            + dna_to_rna(self.rtt_on_pegrna)
            + dna_to_rna(self.pbs_on_pegrna)
        )

        # Flanking context
        self.upstream_flank = seq[:p_s]
        self.downstream_flank = seq[ri_e + 1:]

        # Edit characterisation
        len_init = ri_e - ri_s + 1
        len_mut = rm_e - rm_s + 1
        self.edit_size = len_mut - len_init
        if self.edit_size == 0:
            self.edit_type = "substitution"
        elif self.edit_size > 0:
            self.edit_type = f"insertion (+{self.edit_size} bp)"
        else:
            self.edit_type = f"deletion ({self.edit_size} bp)"

        # Pinpoint the edited bases within the RTT
        self._find_edit_details()

        return self

    def _find_edit_details(self):
        wt = self.rtt_wt
        ed = self.rtt_edited
        # Simple alignment: find first mismatch from the left
        min_len = min(len(wt), len(ed))
        left = 0
        while left < min_len and wt[left] == ed[left]:
            left += 1
        # Find first mismatch from the right
        right_wt = len(wt) - 1
        right_ed = len(ed) - 1
        while right_wt > left and right_ed > left and wt[right_wt] == ed[right_ed]:
            right_wt -= 1
            right_ed -= 1
        self.edit_position_in_rtt = left
        self.edit_wt_bases = wt[left: right_wt + 1] if right_wt >= left else ""
        self.edit_mut_bases = ed[left: right_ed + 1] if right_ed >= left else ""

    def summary(self, row_idx: Optional[int] = None) -> str:
        header = f"{'='*80}"
        if row_idx is not None:
            header += f"\n  Row {row_idx}"
        lines = [
            header,
            "",
            "  TARGET GENOMIC CONTEXT",
            f"    Non-target strand (5'→3'):  {self.nontarget_strand_5to3}",
            f"    Target strand     (3'→5'):  {self.target_strand_5to3}",
            "",
            "  FUNCTIONAL REGIONS  (0-indexed, inclusive, on non-target strand)",
            f"    Protospacer  [{self.protospacer_loc[0]:>2},{self.protospacer_loc[1]:>2}]"
            f"  ({self.protospacer_loc[1]-self.protospacer_loc[0]+1} bp)"
            f"  {self.spacer_dna}",
            f"    PAM          [{self.pam_position[0]:>2},{self.pam_position[1]:>2}]"
            f"           {self.pam_seq}",
            f"    PBS          [{self.pbs_loc[0]:>2},{self.pbs_loc[1]:>2}]"
            f"  ({self.pbs_loc[1]-self.pbs_loc[0]+1} bp)"
            f"  {self.pbs_on_nicked_strand}",
            f"    Nick site       {self.nick_position} | {self.nick_position+1}",
            f"    RTT (wt)     [{self.rt_init_loc[0]:>2},{self.rt_init_loc[1]:>2}]"
            f"  ({self.rt_init_loc[1]-self.rt_init_loc[0]+1} bp)"
            f"  {self.rtt_wt}",
            f"    RTT (edited) [{self.rt_mut_loc[0]:>2},{self.rt_mut_loc[1]:>2}]"
            f"  ({self.rt_mut_loc[1]-self.rt_mut_loc[0]+1} bp)"
            f"  {self.rtt_edited}",
            "",
            "  EDIT",
            f"    Type:      {self.edit_type}",
            f"    Position:  RTT offset {self.edit_position_in_rtt}"
            f"  (genomic pos {self.rt_init_loc[0] + self.edit_position_in_rtt})",
            f"    WT bases:  {self.edit_wt_bases if self.edit_wt_bases else '—'}",
            f"    Mut bases: {self.edit_mut_bases if self.edit_mut_bases else '—'}",
            "",
            "  pegRNA COMPONENTS",
            f"    Spacer (DNA, 5'→3'):  {self.spacer_dna}",
            f"    Spacer (RNA, 5'→3'):  {self.spacer_rna}",
            f"    PBS  on pegRNA (5'→3'):  {self.pbs_on_pegrna}",
            f"    RTT  on pegRNA (5'→3'):  {self.rtt_on_pegrna}",
            "",
            "  FULL pegRNA  (5'→3':  spacer — scaffold — RTT — PBS)",
            f"    {self.pegrna_sequence[:60]}",
        ]
        # Wrap long pegRNA
        peg = self.pegrna_sequence
        if len(peg) > 60:
            for i in range(60, len(peg), 60):
                lines.append(f"    {peg[i:i+60]}")
        lines.append("")
        lines.append(f"    Total length: {len(self.pegrna_sequence)} nt")
        lines.append("")
        return "\n".join(lines)


def parse_location(loc_str: str) -> Tuple[int, int]:
    loc_str = loc_str.strip().strip("[]")
    parts = loc_str.split(",")
    return int(parts[0].strip()), int(parts[1].strip())


def process_row(row: pd.Series) -> PrimeEditDesign:
    design = PrimeEditDesign(
        initial_sequence=row["initial_sequence"],
        mutated_sequence=row["mutated_sequence"],
        protospacer_loc=parse_location(row["protospacer_location"]),
        pbs_loc=parse_location(row["pbs_location"]),
        rt_init_loc=parse_location(row["rt_initial_location"]),
        rt_mut_loc=parse_location(row["rt_mutated_location"]),
    )
    return design.decompose()


def validate_design(d: PrimeEditDesign) -> List[str]:
    """Run sanity checks and return a list of warnings (empty = all good)."""
    warnings = []

    if d.pam_seq[0:2] != "GG" and d.pam_seq[1:3] != "GG":
        warnings.append(f"PAM '{d.pam_seq}' does not contain GG")

    if len(d.spacer_dna) != 20:
        warnings.append(f"Spacer length is {len(d.spacer_dna)}, expected 20")

    # PBS on pegRNA should reverse-complement back to the nicked strand
    rc_pbs = reverse_complement(d.pbs_on_pegrna)
    if rc_pbs != d.pbs_on_nicked_strand:
        warnings.append("PBS reverse-complement mismatch")

    # RTT on pegRNA should reverse-complement to the edited RTT
    rc_rtt = reverse_complement(d.rtt_on_pegrna)
    if rc_rtt != d.rtt_edited:
        warnings.append("RTT reverse-complement mismatch")

    # Downstream of RTT should match (shifted for indels)
    init = d.initial_sequence
    mut = d.mutated_sequence
    ri_e = d.rt_init_loc[1]
    rm_e = d.rt_mut_loc[1]
    remaining = min(len(init) - ri_e - 1, len(mut) - rm_e - 1)
    if remaining > 0:
        init_down = init[ri_e + 1: ri_e + 1 + remaining]
        mut_down = mut[rm_e + 1: rm_e + 1 + remaining]
        if init_down != mut_down:
            warnings.append("Downstream of RTT does not match between initial and mutated")

    return warnings


def main():
    parser = argparse.ArgumentParser(
        description="Decompose PRIDICT-encoded PE data into explicit sequences"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to PRIDICT CSV")
    parser.add_argument(
        "--rows",
        type=str,
        default=None,
        help="Comma-separated row indices to process (default: first 5)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run sanity-check validation on each row",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}\n")

    if args.rows is not None:
        indices = [int(x.strip()) for x in args.rows.split(",")]
    else:
        indices = list(range(min(5, len(df))))

    all_ok = True
    for idx in indices:
        if idx >= len(df):
            print(f"Row {idx} out of range (max {len(df)-1}), skipping.")
            continue

        design = process_row(df.iloc[idx])
        print(design.summary(row_idx=idx))

        if args.validate:
            warns = validate_design(design)
            if warns:
                all_ok = False
                for w in warns:
                    print(f"  ⚠  VALIDATION WARNING: {w}")
                print()
            else:
                print("  ✓  All validation checks passed\n")

    if all_ok:
        print(f"\n{'='*80}")
        print(f"  All {len(indices)} rows passed validation.")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()

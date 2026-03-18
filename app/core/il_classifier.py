"""Incremental-load classifier: decides whether a new file is an IL or OTL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd

from app.core.logger import logger


# ------------------------------------------------------------------ #
#  Data structures                                                     #
# ------------------------------------------------------------------ #

@dataclass
class MatchCandidate:
    """A potential incremental-load target table."""

    table_name: str
    score: float
    source: str  # "manual_inc" | "exact_name" | "milvus" | "column_fallback" | "peer_job"
    peer_job_id: str = ""
    validation_result: Dict = field(default_factory=dict, repr=False)
    is_compatible: bool = field(default=False, repr=False)
    is_additive_evolution: bool = field(default=False, repr=False)
    report: str = field(default="", repr=False)


@dataclass
class ILDecision:
    """Result of incremental-load classification."""

    is_incremental: bool
    candidate: Optional[MatchCandidate] = None
    validation_result: Dict = field(default_factory=dict)
    duplicate_result: Dict = field(default_factory=dict)
    is_compatible: bool = False
    is_additive_evolution: bool = False
    report: str = ""


# ------------------------------------------------------------------ #
#  Classifier                                                          #
# ------------------------------------------------------------------ #

class ILClassifier:
    """Pipeline-based incremental-load classifier.

    Instantiate once per request; the private methods share state through
    *self* so callers stay clean.
    """

    def __init__(self, schema_validator, db_manager, job_manager, settings):
        self._sv = schema_validator
        self._db = db_manager
        self._jm = job_manager
        self._cfg = settings

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def classify(
        self,
        *,
        table_name: str,
        processed_df: "pd.DataFrame",
        column_types: Dict[str, str],
        similar_tables: List[Dict],
        llm_metadata: Dict,
        file_path: str,
        job_id: str,
        sql_mode: str,
        table_name_override: Optional[str],
    ) -> ILDecision:
        """Run the ordered match pipeline and return an IL/OTL decision."""

        # --- Manual OTL ------------------------------------------------
        if sql_mode == "otl" and table_name_override:
            logger.info(
                f"[Job {job_id}] Manual OTL mode selected. "
                f"Bypassing similarity matching and forcing new-table OTL "
                f"for '{table_name_override}'."
            )
            return ILDecision(is_incremental=False)

        # --- Manual INC ------------------------------------------------
        if sql_mode == "inc" and table_name_override:
            target = str(table_name_override).strip().lower().replace(" ", "_")
            logger.info(
                f"[Job {job_id}] Manual INC mode selected. "
                f"Forcing incremental target table: {target}"
            )
            existing_db_types = self._db.get_table_column_types(target)
            if not existing_db_types:
                raise ValueError(
                    f"Manual INC requested for '{target}', "
                    "but target table does not exist."
                )
            candidate = MatchCandidate(
                table_name=target, score=1.0, source="manual_inc",
            )
            self._run_schema_validation(
                candidate, processed_df.columns.tolist(),
                column_types, file_path, job_id,
            )
            return self._finalize(candidate, processed_df, job_id)

        # --- Ordered match pipeline ------------------------------------
        new_columns = processed_df.columns.tolist()

        candidate = (
            self._try_exact_name(table_name, job_id)
            or self._try_milvus(
                similar_tables, table_name, new_columns,
                column_types, llm_metadata, file_path, job_id,
            )
            or self._try_column_fallback(
                table_name, new_columns, column_types,
                llm_metadata, file_path, job_id,
            )
            or self._try_peer_job(
                table_name, new_columns, llm_metadata, job_id,
            )
        )

        if candidate is not None:
            if not candidate.validation_result:
                self._run_schema_validation(
                    candidate, new_columns, column_types, file_path, job_id,
                )
            return self._finalize(candidate, processed_df, job_id)

        logger.info(
            f"[Job {job_id}] No matches found (exact name + Milvus "
            "+ column fallback + peer jobs). "
            "Proceeding with One-Time Load (OTL)."
        )
        return ILDecision(is_incremental=False)

    # ------------------------------------------------------------------ #
    #  Match-source methods  (each returns Optional[MatchCandidate])      #
    # ------------------------------------------------------------------ #

    def _try_exact_name(
        self, table_name: str, job_id: str,
    ) -> Optional[MatchCandidate]:
        """If the LLM-generated name already exists in DB, accept immediately."""
        if not self._db.table_exists(table_name):
            return None
        logger.info(
            f"[Job {job_id}] Exact table name match: '{table_name}' exists "
            "in DB. Bypassing similarity search — routing directly to IL "
            "validation."
        )
        return MatchCandidate(
            table_name=table_name, score=1.0, source="exact_name",
        )

    def _try_milvus(
        self,
        similar_tables: List[Dict],
        table_name: str,
        new_columns: List[str],
        column_types: Dict[str, str],
        llm_metadata: Dict,
        file_path: str,
        job_id: str,
    ) -> Optional[MatchCandidate]:
        """Accept the top Milvus match if it survives schema + LLM guards."""
        if not similar_tables:
            return None

        top = similar_tables[0]
        matched = top["table_name"]
        score = top["similarity_score"]
        logger.info(
            f"[Job {job_id}] Top match: {matched} "
            f"(similarity: {score:.2%})"
        )

        candidate = MatchCandidate(
            table_name=matched, score=score, source="milvus",
        )
        self._run_schema_validation(
            candidate, new_columns, column_types, file_path, job_id,
        )

        match_pct = candidate.validation_result.get("match_percentage", 0.0)
        if match_pct < self._cfg.schema_match_min_percentage:
            logger.warning(
                f"[Job {job_id}] Schema overlap too low "
                f"({match_pct:.1f}% < {self._cfg.schema_match_min_percentage}%) "
                f"for match '{matched}' — treating as new table (OTL)"
            )
            return None

        if not self._verify_with_llm(
            candidate, table_name, new_columns,
            llm_metadata, match_pct, job_id,
        ):
            return None

        return candidate

    def _try_column_fallback(
        self,
        table_name: str,
        new_columns: List[str],
        column_types: Dict[str, str],
        llm_metadata: Dict,
        file_path: str,
        job_id: str,
    ) -> Optional[MatchCandidate]:
        """Iterate IDF-weighted column-overlap candidates until one survives."""
        logger.info(
            f"[Job {job_id}] No Milvus match. Trying column-based fallback."
        )
        fallback_candidates = self._sv.find_similar_table_by_columns(
            new_columns=new_columns,
            min_overlap=self._cfg.column_fallback_min_overlap,
        )

        for idx, (fb_table, fb_score) in enumerate(fallback_candidates):
            logger.info(
                f"[Job {job_id}] Column fallback candidate #{idx + 1}: "
                f"{fb_table} (IDF-weighted overlap: {fb_score:.2%})"
            )
            candidate = MatchCandidate(
                table_name=fb_table, score=fb_score,
                source="column_fallback",
            )
            self._run_schema_validation(
                candidate, new_columns, column_types, file_path, job_id,
            )

            match_pct = candidate.validation_result.get("match_percentage", 0.0)
            if match_pct < self._cfg.schema_match_min_percentage:
                logger.warning(
                    f"[Job {job_id}] Column fallback schema overlap too low "
                    f"({match_pct:.1f}% < "
                    f"{self._cfg.schema_match_min_percentage}%) "
                    f"for '{fb_table}' — trying next candidate"
                )
                continue

            if not self._verify_with_llm(
                candidate, table_name, new_columns,
                llm_metadata, match_pct, job_id,
            ):
                continue

            return candidate

        logger.info(
            f"[Job {job_id}] All column fallback candidates rejected by "
            "guards. Trying peer-job matching."
        )
        return None

    def _try_peer_job(
        self,
        table_name: str,
        new_columns: List[str],
        llm_metadata: Dict,
        job_id: str,
    ) -> Optional[MatchCandidate]:
        """Match against other preprocessed jobs (batch detection)."""
        logger.info(
            f"[Job {job_id}] Trying peer-job match (batch detection)."
        )
        peer_result = self._jm.find_peer_job_by_columns(
            current_job_id=job_id,
            new_columns=new_columns,
            min_overlap=self._cfg.column_fallback_min_overlap,
        )
        if not peer_result:
            return None

        peer_jid, peer_table, peer_score = peer_result
        logger.info(
            f"[Job {job_id}] Peer-job match found: job={peer_jid}, "
            f"table={peer_table}, overlap={peer_score:.2%}"
        )

        candidate = MatchCandidate(
            table_name=peer_table, score=peer_score,
            source="peer_job", peer_job_id=peer_jid,
        )

        peer_overlap_pct = peer_score * 100.0
        if not self._verify_with_llm(
            candidate, table_name, new_columns,
            llm_metadata, peer_overlap_pct, job_id,
        ):
            logger.info(
                f"[Job {job_id}] Peer-job match rejected. "
                "Proceeding with OTL."
            )
            return None

        # Simplified validation — target table doesn't exist in DB yet
        peer_job = self._jm.get_job(peer_jid)
        peer_cols = (
            peer_job.processed_df.columns.tolist()
            if peer_job and peer_job.processed_df is not None
            else []
        )
        new_lower = {c.lower() for c in new_columns}
        peer_lower = {c.lower() for c in peer_cols}
        overlap_cols = new_lower & peer_lower
        match_pct = (
            (len(overlap_cols) / len(peer_lower) * 100) if peer_lower else 0
        )
        missing = [c for c in peer_cols if c.lower() not in new_lower]
        extra = [c for c in new_columns if c.lower() not in peer_lower]

        candidate.validation_result = {
            "is_compatible": new_lower == peer_lower,
            "match_percentage": match_pct,
            "matching_columns": list(overlap_cols),
            "missing_columns": missing,
            "extra_columns": extra,
            "is_additive_evolution": len(extra) > 0 and len(missing) == 0,
            "match_source": "peer_job",
        }
        candidate.is_compatible = candidate.validation_result["is_compatible"]
        candidate.is_additive_evolution = candidate.validation_result.get(
            "is_additive_evolution", False,
        )

        logger.info(
            f"[Job {job_id}] Peer-job incremental load detected. "
            f"Target table: {peer_table} (from job {peer_jid}). "
            f"Schema match: {match_pct:.1f}%"
        )
        return candidate

    # ------------------------------------------------------------------ #
    #  Unified LLM verification                                           #
    # ------------------------------------------------------------------ #

    def _verify_with_llm(
        self,
        candidate: MatchCandidate,
        table_name: str,
        new_columns: List[str],
        llm_metadata: Dict,
        match_pct: float,
        job_id: str,
    ) -> bool:
        """Run LLM semantic verification with structural override.

        Returns True to keep the match.
        """
        if candidate.source == "exact_name":
            return True

        logger.info(
            f"[Job {job_id}] Running LLM semantic verification for "
            f"{candidate.source} match '{candidate.table_name}'"
        )

        if candidate.source == "peer_job":
            peer_job = self._jm.get_job(candidate.peer_job_id)
            peer_meta = (peer_job.llm_metadata or {}) if peer_job else {}
            matched_metadata = {
                "description": peer_meta.get("description", ""),
                "data_domain": peer_meta.get("data_domain", ""),
            }
        else:
            matched_metadata = (
                self._sv.fetch_table_metadata(candidate.table_name) or {}
            )

        semantic = self._sv.verify_semantic_match(
            matched_table_name=candidate.table_name,
            matched_table_metadata=matched_metadata,
            new_table_name=table_name,
            new_columns=new_columns,
            new_llm_metadata=llm_metadata,
            similarity_score=candidate.score,
            column_overlap_pct=match_pct,
        )
        logger.info(
            f"[Job {job_id}] Semantic verification: "
            f"is_related={semantic['is_related']}, "
            f"confidence={semantic['confidence']:.2f}, "
            f"reasoning='{semantic['reasoning']}'"
        )

        if semantic["is_related"]:
            return True

        if self._should_override_rejection(
            semantic, match_pct, table_name, candidate.table_name,
        ):
            logger.warning(
                f"[Job {job_id}] LLM rejected {candidate.source} match "
                f"'{candidate.table_name}' "
                f"(confidence={semantic['confidence']:.2f}) but structural "
                f"override triggered (overlap={match_pct:.1f}%) "
                "— keeping IL match"
            )
            return True

        logger.warning(
            f"[Job {job_id}] LLM rejected {candidate.source} match "
            f"'{candidate.table_name}' "
            f"(confidence={semantic['confidence']:.2f}) — discarding"
        )
        return False

    # ------------------------------------------------------------------ #
    #  Schema validation + finalization                                    #
    # ------------------------------------------------------------------ #

    def _run_schema_validation(
        self,
        candidate: MatchCandidate,
        new_columns: List[str],
        column_types: Dict[str, str],
        file_path: str,
        job_id: str,
    ) -> None:
        """Run validate_incremental_load and cache results on *candidate*."""
        logger.info(
            f"[Job {job_id}] Validating schema compatibility"
        )
        existing_db_types = self._db.get_table_column_types(
            candidate.table_name,
        )
        new_types = {k.lower(): v for k, v in column_types.items()}

        is_compatible, validation_result, report = (
            self._sv.validate_incremental_load(
                table_name=candidate.table_name,
                new_columns=new_columns,
                new_file_name=Path(file_path).name,
                new_types=new_types,
                existing_types=existing_db_types,
            )
        )
        candidate.validation_result = validation_result
        candidate.is_compatible = is_compatible
        candidate.is_additive_evolution = validation_result.get(
            "is_additive_evolution", False,
        )
        candidate.report = report
        logger.info(
            f"[Job {job_id}] Schema validation for '{candidate.table_name}': "
            f"compatible={is_compatible}, "
            f"additive_evolution={candidate.is_additive_evolution}"
        )

    def _finalize(
        self,
        candidate: MatchCandidate,
        processed_df: "pd.DataFrame",
        job_id: str,
    ) -> ILDecision:
        """Run duplicate detection (DB-backed tables) and build ILDecision."""
        duplicate_result: Dict = {}

        if candidate.source != "peer_job":
            logger.info(f"[Job {job_id}] Checking for duplicate data")
            duplicate_result = self._sv.detect_duplicate_data(
                table_name=candidate.table_name,
                new_df=processed_df,
            )
            logger.info(
                f"[Job {job_id}] Duplicate detection: "
                f"{duplicate_result['status']}"
            )
            logger.info(f"[Job {job_id}] {duplicate_result['message']}")

        logger.info(f"[Job {job_id}] Validation report:\n{candidate.report}")

        return ILDecision(
            is_incremental=True,
            candidate=candidate,
            validation_result=candidate.validation_result,
            duplicate_result=duplicate_result,
            is_compatible=candidate.is_compatible,
            is_additive_evolution=candidate.is_additive_evolution,
            report=candidate.report,
        )

    # ------------------------------------------------------------------ #
    #  Structural override                                                 #
    # ------------------------------------------------------------------ #

    def _should_override_rejection(
        self,
        semantic_result: dict,
        overlap_pct: float,
        table_name_a: str,
        table_name_b: str,
    ) -> bool:
        """Return True when structural evidence outweighs an LLM rejection."""
        confidence = semantic_result.get("confidence", 1.0)
        name_sim = self._sv.compute_table_name_similarity(
            table_name_a, table_name_b,
        )

        if (
            overlap_pct >= self._cfg.structural_override_min_overlap
            and confidence < self._cfg.semantic_rejection_min_confidence
            and name_sim >= self._cfg.table_name_override_min_similarity
        ):
            return True

        if (
            overlap_pct >= 95.0
            and confidence < self._cfg.semantic_rejection_min_confidence
        ):
            return True

        return False

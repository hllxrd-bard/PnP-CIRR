from __future__ import annotations

from collections import defaultdict

from .config import AppConfig
from .reranker import RankedCandidate
from .schemas import DeduplicationOverrides
from .utils import coerce_timestamp


class CandidateDeduplicator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.cfg = config.section("deduplication")
        self.fields = config.get("milvus.fields", {})

    def apply(
        self,
        ranked: list[RankedCandidate],
        overrides: DeduplicationOverrides,
        top_k: int,
    ) -> list[RankedCandidate]:
        enabled = self.cfg.get("enabled", True) if overrides.enabled is None else overrides.enabled
        if not enabled:
            return ranked[:top_k]

        window = (
            float(self.cfg.get("timestamp_window_seconds", 1.5))
            if overrides.timestamp_window_seconds is None
            else float(overrides.timestamp_window_seconds)
        )
        max_per_video = (
            int(self.cfg.get("max_frames_per_video", 5))
            if overrides.max_frames_per_video is None
            else int(overrides.max_frames_per_video)
        )
        configured_cluster_limit = self.cfg.get("max_frames_per_cluster")
        max_per_cluster = (
            int(configured_cluster_limit)
            if overrides.max_frames_per_cluster is None and configured_cluster_limit is not None
            else (
                int(overrides.max_frames_per_cluster)
                if overrides.max_frames_per_cluster is not None
                else None
            )
        )

        video_field = self.fields["video_name"]
        timestamp_field = self.fields["timestamp"]
        cluster_field = self.fields.get("cluster_id")

        video_counts: defaultdict[str, int] = defaultdict(int)
        cluster_counts: defaultdict[str, int] = defaultdict(int)
        accepted_timestamps: defaultdict[str, list[float]] = defaultdict(list)
        selected: list[RankedCandidate] = []

        for candidate in ranked:
            entity = candidate.entity
            video_name = str(entity.get(video_field, ""))
            if video_counts[video_name] >= max_per_video:
                continue

            cluster_value = entity.get(cluster_field) if cluster_field else None
            cluster_key = ""
            if cluster_value is not None:
                try:
                    if float(cluster_value) >= 0:
                        cluster_key = str(cluster_value)
                except (TypeError, ValueError):
                    cluster_key = str(cluster_value).strip()
            if (
                cluster_key
                and max_per_cluster is not None
                and cluster_counts[cluster_key] >= max_per_cluster
            ):
                continue

            timestamp = coerce_timestamp(entity.get(timestamp_field))
            if timestamp is not None and window > 0:
                if any(abs(timestamp - previous) < window for previous in accepted_timestamps[video_name]):
                    continue

            selected.append(candidate)
            video_counts[video_name] += 1
            if cluster_key:
                cluster_counts[cluster_key] += 1
            if timestamp is not None:
                accepted_timestamps[video_name].append(timestamp)
            if len(selected) >= top_k:
                break
        return selected

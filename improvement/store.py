# -*- coding: utf-8 -*-
"""
improvement.store —— 提案 / 门禁事件 / 生效档案 三张表的读写
================================================================================

复用 prompt_manager 的 MySQL 配置与连接池模式（pymysql + dbutils PooledDB），
零新增依赖。任何失败都降级为「不可用」，绝不拖垮主流程。

表职责
------
improvement_proposals  一次提案的完整生命周期（含 before/after 快照，回滚的唯一依据）
improvement_gate_events 每次门禁判定的留痕（出事后复盘：到底卡在哪一道）
proposal_archive       已合并、正在生效的变更（archive 的「活着的部分」）

注意：archive 里**保留 rejected / rolled_back 的记录**，这是 DGM archive 的核心语义——
成绩不好的分支也要留着，它们可能是后续突破的起点。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .schema import (ImprovementProposal, GateEvent, ST_MERGED,
                     ST_ROLLED_BACK, ST_REJECTED, ROLLBACK_BLACKLIST_THRESHOLD)


class ImprovementStore:
    """三张表的读写封装。available=False 时所有写操作静默失败。"""

    def __init__(self, verbose: bool = True):
        self._pool = None
        self.available = False
        self.verbose = verbose
        self._init_db()

    # ========================================================================
    # 连接
    # ========================================================================

    def _init_db(self):
        try:
            import pymysql
            from dbutils.pooled_db import PooledDB
            from prompt_manager import (MYSQL_HOST, MYSQL_PORT, MYSQL_USER,
                                        MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_CHARSET)

            self._pool = PooledDB(
                creator=pymysql,
                maxconnections=3, mincached=1, maxcached=2,
                host=MYSQL_HOST, port=MYSQL_PORT,
                user=MYSQL_USER, password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE, charset=MYSQL_CHARSET,
                autocommit=True,
            )
            self._create_tables()
            self.available = True
            if self.verbose:
                print("  [ImprovementStore] 已连接")
        except Exception as e:
            if self.verbose:
                print(f"  [ImprovementStore] 不可用（降级）: {type(e).__name__}: {e}")
            self._pool = None

    def _conn(self):
        return self._pool.connection() if self._pool else None

    def _create_tables(self):
        conn = self._conn()
        if conn is None:
            return
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS improvement_proposals (
                id              BIGINT AUTO_INCREMENT PRIMARY KEY,
                proposal_id     VARCHAR(64)  NOT NULL UNIQUE,
                parent_id       VARCHAR(64)  DEFAULT NULL,
                root_cause      VARCHAR(8)   NOT NULL,
                change_layer    VARCHAR(8)   NOT NULL,
                change_target   VARCHAR(128) NOT NULL,
                payload_before  LONGTEXT,
                payload_after   LONGTEXT,
                rationale       TEXT,
                status          VARCHAR(16)  NOT NULL DEFAULT 'draft',
                dev_score       JSON,
                holdout_score   JSON,
                baseline_run_id VARCHAR(64),
                variant_run_id  VARCHAR(64),
                fingerprint     VARCHAR(16),
                created_at      DATETIME     NOT NULL,
                decided_at      DATETIME,
                decided_by      VARCHAR(32),
                INDEX idx_status (status),
                INDEX idx_cause  (root_cause),
                INDEX idx_parent (parent_id),
                INDEX idx_fp     (fingerprint)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS improvement_gate_events (
                id          BIGINT AUTO_INCREMENT PRIMARY KEY,
                proposal_id VARCHAR(64) NOT NULL,
                gate        VARCHAR(32) NOT NULL,
                passed      TINYINT(1)  NOT NULL,
                detail      TEXT,
                created_at  DATETIME    NOT NULL,
                INDEX idx_proposal (proposal_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS proposal_archive (
                id            BIGINT AUTO_INCREMENT PRIMARY KEY,
                proposal_id   VARCHAR(64) NOT NULL,
                change_layer  VARCHAR(8)  NOT NULL,
                change_target VARCHAR(128) NOT NULL,
                payload_after LONGTEXT,
                status        VARCHAR(16) NOT NULL,
                merged_at     DATETIME,
                rollbacks     INT DEFAULT 0,
                is_active     TINYINT(1) DEFAULT 1,
                INDEX idx_target (change_target),
                INDEX idx_proposal (proposal_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cur.close()
        conn.close()

    # ========================================================================
    # 提案
    # ========================================================================

    def save_proposal(self, p: ImprovementProposal) -> bool:
        """落盘一条提案。payload_before 为空直接拒（没有 before 就没有回滚）。"""
        if not self.available:
            return False
        if not p.payload_before:
            raise ValueError("payload_before 为空：没有变更前快照就无法回滚")
        conn = self._conn()
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO improvement_proposals
                  (proposal_id, parent_id, root_cause, change_layer, change_target,
                   payload_before, payload_after, rationale, status,
                   dev_score, holdout_score, baseline_run_id, variant_run_id,
                   fingerprint, created_at, decided_at, decided_by)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                   status=VALUES(status), dev_score=VALUES(dev_score),
                   holdout_score=VALUES(holdout_score),
                   baseline_run_id=VALUES(baseline_run_id),
                   variant_run_id=VALUES(variant_run_id),
                   decided_at=VALUES(decided_at), decided_by=VALUES(decided_by)
            """, (
                p.proposal_id, p.parent_id, p.root_cause, p.change_layer,
                p.change_target, p.payload_before, p.payload_after, p.rationale,
                p.status, json.dumps(p.dev_score, ensure_ascii=False),
                json.dumps(p.holdout_score, ensure_ascii=False),
                p.baseline_run_id, p.variant_run_id, p.fingerprint(),
                _dt(p.created_at), _dt(p.decided_at), p.decided_by,
            ))
            cur.close()
            conn.close()
            return True
        except Exception as e:
            if self.verbose:
                print(f"  [ImprovementStore] save_proposal 失败: {e}")
            return False

    def load_proposal(self, proposal_id: str) -> Optional[ImprovementProposal]:
        if not self.available:
            return None
        conn = self._conn()
        if conn is None:
            return None
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT proposal_id, parent_id, root_cause, change_layer, change_target,"
                " payload_before, payload_after, rationale, status, dev_score,"
                " holdout_score, baseline_run_id, variant_run_id, created_at,"
                " decided_at, decided_by FROM improvement_proposals WHERE proposal_id=%s",
                (proposal_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if not row:
                return None
            return ImprovementProposal(
                proposal_id=row[0], parent_id=row[1], root_cause=row[2],
                change_layer=row[3], change_target=row[4],
                payload_before=row[5] or "", payload_after=row[6] or "",
                rationale=row[7] or "", status=row[8] or "draft",
                dev_score=_json(row[9]), holdout_score=_json(row[10]),
                baseline_run_id=row[11] or "", variant_run_id=row[12] or "",
                created_at=_ts(row[13]), decided_at=_ts(row[14]) if row[14] else None,
                decided_by=row[15] or "",
            )
        except Exception as e:
            if self.verbose:
                print(f"  [ImprovementStore] load_proposal 失败: {e}")
            return None

    def list_proposals(self, status: Optional[str] = None,
                       limit: int = 50) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        conn = self._conn()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            if status:
                cur.execute(
                    "SELECT proposal_id, root_cause, change_layer, change_target,"
                    " status, created_at, decided_by FROM improvement_proposals"
                    " WHERE status=%s ORDER BY id DESC LIMIT %s", (status, limit))
            else:
                cur.execute(
                    "SELECT proposal_id, root_cause, change_layer, change_target,"
                    " status, created_at, decided_by FROM improvement_proposals"
                    " ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [{
                "proposal_id": r[0], "root_cause": r[1], "change_layer": r[2],
                "change_target": r[3], "status": r[4],
                "created_at": str(r[5]), "decided_by": r[6] or "",
            } for r in rows]
        except Exception:
            return []

    def has_fingerprint(self, fp: str) -> bool:
        """去重：同一个 target + 同一个 after 已经有了就不再提。"""
        if not self.available:
            return False
        conn = self._conn()
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM improvement_proposals WHERE fingerprint=%s LIMIT 1", (fp,))
            hit = cur.fetchone() is not None
            cur.close()
            conn.close()
            return hit
        except Exception:
            return False

    # ========================================================================
    # 门禁事件
    # ========================================================================

    def add_gate_event(self, ev: GateEvent) -> bool:
        if not self.available:
            return False
        conn = self._conn()
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO improvement_gate_events"
                " (proposal_id, gate, passed, detail, created_at) VALUES (%s,%s,%s,%s,%s)",
                (ev.proposal_id, ev.gate, 1 if ev.passed else 0,
                 ev.detail[:2000], _dt(ev.created_at)))
            cur.close()
            conn.close()
            return True
        except Exception:
            return False

    def list_gate_events(self, proposal_id: str) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        conn = self._conn()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT gate, passed, detail FROM improvement_gate_events"
                " WHERE proposal_id=%s ORDER BY id", (proposal_id,))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [{"gate": r[0], "passed": bool(r[1]), "detail": r[2] or ""}
                    for r in rows]
        except Exception:
            return []

    # ========================================================================
    # 生效档案
    # ========================================================================

    def archive(self, p: ImprovementProposal, active: bool = True) -> bool:
        """
        写档案。**rejected / rolled_back 也写**——保留失败分支，
        它们是后续代次的父代候选（DGM archive 语义）。
        """
        if not self.available:
            return False
        conn = self._conn()
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM proposal_archive WHERE proposal_id=%s LIMIT 1",
                (p.proposal_id,))
            exists = cur.fetchone() is not None
            if exists:
                cur.execute(
                    "UPDATE proposal_archive SET status=%s, is_active=%s,"
                    " merged_at=%s WHERE proposal_id=%s",
                    (p.status, 1 if active else 0,
                     _dt(time.time()) if p.status == ST_MERGED else None,
                     p.proposal_id))
            else:
                cur.execute(
                    "INSERT INTO proposal_archive"
                    " (proposal_id, change_layer, change_target, payload_after,"
                    "  status, merged_at, rollbacks, is_active) VALUES (%s,%s,%s,%s,%s,%s,0,%s)",
                    (p.proposal_id, p.change_layer, p.change_target,
                     p.payload_after, p.status,
                     _dt(time.time()) if p.status == ST_MERGED else None,
                     1 if active else 0))
            cur.close()
            conn.close()
            return True
        except Exception as e:
            if self.verbose:
                print(f"  [ImprovementStore] archive 失败: {e}")
            return False

    def bump_rollback(self, proposal_id: str) -> int:
        """回滚计数 +1，返回累加后的次数。"""
        if not self.available:
            return 0
        conn = self._conn()
        if conn is None:
            return 0
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE proposal_archive SET rollbacks=rollbacks+1, is_active=0"
                " WHERE proposal_id=%s", (proposal_id,))
            cur.execute(
                "SELECT rollbacks FROM proposal_archive WHERE proposal_id=%s",
                (proposal_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def rollback_count(self, change_target: str) -> int:
        """某个目标历史上被回滚过几次（用于黑名单判定）。"""
        if not self.available:
            return 0
        conn = self._conn()
        if conn is None:
            return 0
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT COALESCE(SUM(rollbacks),0) FROM proposal_archive"
                " WHERE change_target=%s", (change_target,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def is_blacklisted(self, change_target: str) -> bool:
        return self.rollback_count(change_target) >= ROLLBACK_BLACKLIST_THRESHOLD

    def active_changes(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        conn = self._conn()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT proposal_id, change_layer, change_target, status, merged_at"
                " FROM proposal_archive WHERE is_active=1 ORDER BY id DESC LIMIT %s",
                (limit,))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [{"proposal_id": r[0], "change_layer": r[1],
                     "change_target": r[2], "status": r[3],
                     "merged_at": str(r[4]) if r[4] else ""} for r in rows]
        except Exception:
            return []

    def today_auto_merge_count(self) -> int:
        if not self.available:
            return 0
        conn = self._conn()
        if conn is None:
            return 0
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM improvement_proposals"
                " WHERE decided_by='auto' AND status=%s AND DATE(decided_at)=CURDATE()",
                (ST_MERGED,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            return int(row[0]) if row else 0
        except Exception:
            return 0


# ============================================================================
# 小工具
# ============================================================================

def _dt(ts: Optional[float]):
    if not ts:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _ts(v) -> float:
    if v is None:
        return time.time()
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return time.mktime(time.strptime(str(v), "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return time.time()


def _json(v) -> Dict[str, Any]:
    if not v:
        return {}
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v)
    except Exception:
        return {}

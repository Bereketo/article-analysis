"""
Simple Database Service for CDD Reports
=======================================

Minimal PostgreSQL integration that reuses existing pipeline code.
"""

import psycopg2
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


class SimpleReportDB:
    """Simple database service for CDD report tracking"""
    
    def __init__(self):
        # Simple connection configuration
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5433'),  # Use PostgreSQL 16 cluster
            'database': os.getenv('DB_NAME', 'article_analysis'),
            'user': os.getenv('DB_USER', 'article_user'),
            'password': os.getenv('DB_PASSWORD', 'article_pass')
        }
    
    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    def create_report(self, report_name: str, company_name: str, alias_list: List[str], 
                     notification_emails: List[str], created_by: str, 
                     start_date: Optional[str] = None, end_date: Optional[str] = None,
                     keyword_groups: Optional[Dict[str, List[str]]] = None) -> str:
        """Step 1: Create new report and return UUID"""
        report_uuid = str(uuid.uuid4())
        
        initial_status = {
            "status": "SEARCH_IN_PROGRESS",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": "api"
        }
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO cdd_report_master (
                        report_uuid, report_name, company_name, alias_list, 
                        notification_emails, created_by, updated_by, 
                        current_status, status_history, start_date, end_date, keyword_groups
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    report_uuid, report_name, company_name, 
                    json.dumps(alias_list), notification_emails, 
                    created_by, created_by, "SEARCH_IN_PROGRESS", 
                    json.dumps([initial_status]), start_date, end_date,
                    json.dumps(keyword_groups) if keyword_groups else None
                ))
        
        logger.info(f"✅ Created report {report_uuid} for {company_name}")
        return report_uuid
    
    def update_database_fields(self, report_uuid: str, fields_to_update: Dict[str, Any]) -> bool:
        """Update specific database fields for a report"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Build dynamic update query
                    set_clauses = []
                    params = []
                    
                    for field, value in fields_to_update.items():
                        if field in ['start_date', 'end_date', 'keyword_groups', 'notification_emails', 'alias_list']:
                            set_clauses.append(f"{field} = %s")
                            if field == 'keyword_groups' or field == 'alias_list':
                                params.append(json.dumps(value) if value else None)
                            elif field == 'notification_emails':
                                params.append(value if value else [])
                            else:
                                params.append(value)
                    
                    if set_clauses:
                        set_clauses.append("updated_by = %s")
                        params.append("system")
                        
                        query = f"UPDATE cdd_report_master SET {', '.join(set_clauses)} WHERE report_uuid = %s"
                        params.append(report_uuid)
                        
                        cur.execute(query, params)
                        logger.info(f"✅ Updated database fields for report {report_uuid}: {list(fields_to_update.keys())}")
                        return True
            
            return False
        except Exception as e:
            logger.error(f"❌ Failed to update database fields: {e}")
            return False
    
    def update_status(self, report_uuid: str, new_status: str, actor: str, 
                     details: Optional[Dict] = None, file_path: Optional[str] = None,
                     file_type: Optional[str] = None, file_paths: Optional[Dict[str, str]] = None) -> bool:
        """Update report status and history
        
        Args:
            report_uuid: UUID of the report
            new_status: New status to set
            actor: Actor performing the update
            details: Additional details for status entry
            file_path: Single file path (legacy support)
            file_type: Type of single file path (legacy support)
            file_paths: Dictionary of multiple file paths {file_type: path}
                       Supported keys: 'article_list', 'article_details', 'report', 'pdf_report'
        """
        try:
            status_entry = {
                "status": new_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "actor": actor
            }
            if details:
                status_entry["details"] = details
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Get current status history
                    cur.execute(
                        "SELECT status_history FROM cdd_report_master WHERE report_uuid = %s",
                        (report_uuid,)
                    )
                    row = cur.fetchone()
                    if not row:
                        return False
                    
                    # Update status and history
                    current_history = row[0] or []
                    current_history.append(status_entry)
                    
                    update_query = """
                        UPDATE cdd_report_master 
                        SET current_status = %s, status_history = %s, updated_by = %s
                    """
                    params = [new_status, json.dumps(current_history), actor]
                    
                    # Support multiple file paths (new way)
                    if file_paths:
                        logger.info(f"📁 Updating file paths for report {report_uuid}: {file_paths}")
                        for ftype, fpath in file_paths.items():
                            if fpath:  # Only update if path is not None/empty
                                if ftype == "article_list":
                                    update_query += ", article_list_file_path = %s"
                                    logger.debug(f"   Adding article_list_file_path: {fpath}")
                                elif ftype == "article_details":
                                    update_query += ", article_details_file_path = %s"
                                    logger.debug(f"   Adding article_details_file_path: {fpath}")
                                elif ftype == "report":
                                    update_query += ", report_file_path = %s"
                                    logger.debug(f"   Adding report_file_path: {fpath}")
                                elif ftype == "pdf_report":
                                    update_query += ", pdf_report_file_path = %s"
                                    logger.debug(f"   Adding pdf_report_file_path: {fpath}")
                                params.append(fpath)
                    # Legacy support: single file path
                    elif file_path and file_type:
                        if file_type == "article_list":
                            update_query += ", article_list_file_path = %s"
                        elif file_type == "article_details":
                            update_query += ", article_details_file_path = %s"
                        elif file_type == "report":
                            update_query += ", report_file_path = %s"
                        elif file_type == "pdf_report":
                            update_query += ", pdf_report_file_path = %s"
                        params.append(file_path)
                    
                    update_query += " WHERE report_uuid = %s"
                    params.append(report_uuid)
                    
                    cur.execute(update_query, params)
            
            logger.info(f"✅ Updated report {report_uuid} to {new_status}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update status: {e}")
            return False
    
    def get_report(self, report_uuid: str) -> Optional[Dict]:
        """Get report by UUID"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT report_uuid, company_name, current_status, 
                               article_list_file_path, article_details_file_path, 
                               report_file_path, pdf_report_file_path, status_history, updated_at,
                               start_date, end_date, keyword_groups
                        FROM cdd_report_master 
                        WHERE report_uuid = %s
                    """, (report_uuid,))
                    
                    row = cur.fetchone()
                    if not row:
                        return None
                    
                    return {
                        'report_uuid': str(row[0]),
                        'company_name': row[1],
                        'current_status': row[2],
                        'article_list_file_path': row[3],
                        'article_details_file_path': row[4],
                        'report_file_path': row[5],
                        'pdf_report_file_path': row[6],
                        'status_history': row[7],
                        'updated_at': row[8].isoformat() if row[8] else None,
                        'start_date': row[9],
                        'end_date': row[10],
                        'keyword_groups': row[11]
                    }
        except Exception as e:
            logger.error(f"❌ Failed to get report: {e}")
            return None


# Simple convenience functions that reuse your existing pipeline
def create_and_run_pipeline(report_name: str, company_name: str, alias_list: List[str], 
                           notification_emails: List[str], created_by: str,
                           start_date: Optional[str] = None, end_date: Optional[str] = None,
                           keyword_groups: Optional[Dict[str, List[str]]] = None) -> str:
    """Create report and run the complete pipeline"""
    db = SimpleReportDB()
    
    # Step 1: Create report
    report_uuid = db.create_report(report_name, company_name, alias_list, 
                                  notification_emails, created_by, start_date, end_date, keyword_groups)
    
    try:
        # Step 2: Run existing search (reuse your existing search logic)
        print(f"🔍 Running search for {company_name}...")
        # TODO: Call your existing search code here
        
        # Simulate search completion
        db.update_status(report_uuid, "SEARCH_COMPLETED", "search-worker", 
                        {"found": 100, "deduped": 75}, 
                        f"search-results/{company_name}_search.xlsx", "article_list")
        
        # Step 3 & 4: Run existing extraction (reuse extract_content_from_excel.py)
        db.update_status(report_uuid, "EXTRACTION_IN_PROGRESS", "extraction-worker")
        print(f"📄 Running content extraction...")
        # TODO: Call your existing extraction code here
        
        db.update_status(report_uuid, "EXTRACTION_COMPLETED", "extraction-worker",
                        {"extracted": 75, "successful": 65, "failed": 10},
                        f"extracted_content/{company_name}_content.json", "article_details")
        
        # Step 5 & 6: Run existing analysis (reuse analyze_extracted_content.py)
        db.update_status(report_uuid, "ANALYSIS_IN_PROGRESS", "analysis-worker")
        print(f"🧠 Running analysis...")
        # TODO: Call your existing analysis code here
        
        db.update_status(report_uuid, "ANALYSIS_COMPLETED", "analysis-worker",
                        {"analyzed": 65, "adverse": 11, "risk_breakdown": {"Legal": 5}},
                        f"llm-analysis/{company_name}_report.xlsx", "report")
        
        print(f"✅ Pipeline completed for report {report_uuid}")
        return report_uuid
        
    except Exception as e:
        db.update_status(report_uuid, "SEARCH_FAILED", "system", {"error": str(e)})
        logger.error(f"❌ Pipeline failed: {e}")
        raise


def get_report_status(report_uuid: str) -> Optional[Dict]:
    """Get current report status"""
    db = SimpleReportDB()
    return db.get_report(report_uuid)


# Database setup SQL (simplified)
SETUP_SQL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS cdd_report_master (
    id                      BIGSERIAL PRIMARY KEY,
    report_uuid             UUID NOT NULL DEFAULT uuid_generate_v4() UNIQUE,
    report_name             TEXT NOT NULL,
    company_name            TEXT NOT NULL,
    alias_list              JSONB NOT NULL DEFAULT '[]',
    notification_emails     TEXT[] NOT NULL DEFAULT '{}',
    created_by              TEXT NOT NULL,
    updated_by              TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    article_list_file_path      TEXT,
    article_details_file_path   TEXT,
    report_file_path            TEXT,
    pdf_report_file_path        TEXT,
    current_status          TEXT NOT NULL DEFAULT 'NEW',
    status_history          JSONB NOT NULL DEFAULT '[]',
    start_date              DATE,
    end_date                DATE,
    keyword_groups          JSONB
);

CREATE INDEX IF NOT EXISTS idx_report_uuid ON cdd_report_master(report_uuid);
CREATE INDEX IF NOT EXISTS idx_current_status ON cdd_report_master(current_status);
"""

def setup_database():
    """Simple database setup"""
    db = SimpleReportDB()
    try:
        with db._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SETUP_SQL)
        print("✅ Database setup completed")
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

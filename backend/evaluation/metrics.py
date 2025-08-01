import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
import statistics

logger = logging.getLogger(__name__)

class MetricsLogger:
    """Handles logging and metrics collection for queries and responses"""
    
    def __init__(self, db_path: str = "logs/metrics.db"):
        self.db_path = db_path
        self._init_database()
        logger.info(f"Metrics logger initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    sources TEXT,
                    query_type TEXT DEFAULT 'rag',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def log_query(self, query: str, response: str, model_used: str, 
                  response_time: float, session_id: str, sources: Optional[List[str]] = None,
                  query_type: str = "rag"):
        """Log a query and its response"""
        try:
            sources_json = json.dumps(sources) if sources else None
            timestamp = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO query_logs 
                    (timestamp, session_id, query, response, model_used, response_time, sources, query_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, session_id, query, response, model_used, response_time, sources_json, query_type))
                conn.commit()
            
            logger.debug(f"Logged query for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error logging query: {e}")
    
    def log_system_metric(self, metric_name: str, metric_value: float, metadata: Optional[Dict] = None):
        """Log a system metric"""
        try:
            timestamp = datetime.now().isoformat()
            metadata_json = json.dumps(metadata) if metadata else None
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO system_metrics (timestamp, metric_name, metric_value, metadata)
                    VALUES (?, ?, ?, ?)
                """, (timestamp, metric_name, metric_value, metadata_json))
                conn.commit()
            
            logger.debug(f"Logged system metric: {metric_name} = {metric_value}")
            
        except Exception as e:
            logger.error(f"Error logging system metric: {e}")
    
    def get_logs(self, limit: int = 100, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get query logs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if session_id:
                    cursor = conn.execute("""
                        SELECT * FROM query_logs 
                        WHERE session_id = ?
                        ORDER BY created_at DESC 
                        LIMIT ?
                    """, (session_id, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM query_logs 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    """, (limit,))
                
                logs = []
                for row in cursor.fetchall():
                    log_entry = dict(row)
                    # Parse sources JSON
                    if log_entry['sources']:
                        try:
                            log_entry['sources'] = json.loads(log_entry['sources'])
                        except json.JSONDecodeError:
                            log_entry['sources'] = []
                    else:
                        log_entry['sources'] = []
                    logs.append(log_entry)
                
                return logs
                
        except Exception as e:
            logger.error(f"Error retrieving logs: {e}")
            return []
    
    def get_query_count(self, session_id: Optional[str] = None) -> int:
        """Get total number of queries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if session_id:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM query_logs WHERE session_id = ?", 
                        (session_id,)
                    )
                else:
                    cursor = conn.execute("SELECT COUNT(*) FROM query_logs")
                
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Error getting query count: {e}")
            return 0
    
    def get_average_response_time(self, model_name: Optional[str] = None, 
                                  hours: Optional[int] = None) -> float:
        """Get average response time, optionally filtered by model and time"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT AVG(response_time) FROM query_logs WHERE 1=1"
                params = []
                
                if model_name:
                    query += " AND model_used = ?"
                    params.append(model_name)
                
                if hours:
                    query += " AND created_at >= datetime('now', '-{} hours')".format(hours)
                
                cursor = conn.execute(query, params)
                result = cursor.fetchone()[0]
                return result if result is not None else 0.0
                
        except Exception as e:
            logger.error(f"Error getting average response time: {e}")
            return 0.0
    
    def get_model_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics by model"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT model_used, COUNT(*) 
                    FROM query_logs 
                    GROUP BY model_used
                """)
                
                return dict(cursor.fetchall())
                
        except Exception as e:
            logger.error(f"Error getting model usage stats: {e}")
            return {}
    
    def get_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the last N hours"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Query metrics for the specified time period
                cursor = conn.execute("""
                    SELECT response_time, model_used, query_type
                    FROM query_logs 
                    WHERE created_at >= datetime('now', '-{} hours')
                """.format(hours), )
                
                results = cursor.fetchall()
                
                if not results:
                    return {
                        "total_queries": 0,
                        "average_response_time": 0.0,
                        "median_response_time": 0.0,
                        "min_response_time": 0.0,
                        "max_response_time": 0.0,
                        "model_stats": {},
                        "query_type_stats": {}
                    }
                
                response_times = [row[0] for row in results]
                models = [row[1] for row in results]
                query_types = [row[2] for row in results]
                
                # Calculate statistics
                model_stats = {}
                for model in set(models):
                    model_times = [rt for rt, m in zip(response_times, models) if m == model]
                    model_stats[model] = {
                        "count": len(model_times),
                        "avg_response_time": statistics.mean(model_times),
                        "median_response_time": statistics.median(model_times)
                    }
                
                query_type_stats = {}
                for qtype in set(query_types):
                    type_times = [rt for rt, qt in zip(response_times, query_types) if qt == qtype]
                    query_type_stats[qtype] = {
                        "count": len(type_times),
                        "avg_response_time": statistics.mean(type_times)
                    }
                
                return {
                    "total_queries": len(results),
                    "average_response_time": statistics.mean(response_times),
                    "median_response_time": statistics.median(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "model_stats": model_stats,
                    "query_type_stats": query_type_stats
                }
                
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def export_logs(self, filename: str, format: str = "json"):
        """Export logs to a file"""
        try:
            logs = self.get_logs(limit=10000)  # Get more logs for export
            
            if format.lower() == "json":
                with open(filename, 'w') as f:
                    json.dump(logs, f, indent=2, default=str)
            elif format.lower() == "csv":
                import csv
                if logs:
                    with open(filename, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                        writer.writeheader()
                        writer.writerows(logs)
            
            logger.info(f"Exported {len(logs)} logs to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting logs: {e}")
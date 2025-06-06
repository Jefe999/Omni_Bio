"""
Report generation modules for biomarker discovery
Task #14: Comprehensive report generator for lab documentation
"""

from .report_generator import (
    generate_comprehensive_report,
    generate_executive_summary,
    ReportConfig,
    ReportSection
)

__all__ = [
    'generate_comprehensive_report',
    'generate_executive_summary', 
    'ReportConfig',
    'ReportSection'
] 
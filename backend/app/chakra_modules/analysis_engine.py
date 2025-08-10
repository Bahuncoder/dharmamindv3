"""
Analysis Engine - System Analysis and Validation
==============================================

This module provides comprehensive analysis capabilities for the DharmaMind system,
including system health monitoring, performance analysis, and dharmic compliance assessment.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import inspect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of analysis available"""
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    DHARMIC_COMPLIANCE = "dharmic_compliance"
    MODULE_INTEGRITY = "module_integrity"
    INTEGRATION_STATUS = "integration_status"
    SECURITY_ASSESSMENT = "security_assessment"
    CONSCIOUSNESS_ANALYSIS = "consciousness_analysis"

class AnalysisLevel(Enum):
    """Analysis depth levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP_SCAN = "deep_scan"

class ComponentStatus(Enum):
    """Component status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class AnalysisRequest:
    """Request for system analysis"""
    analysis_type: AnalysisType
    level: AnalysisLevel = AnalysisLevel.STANDARD
    components: List[str] = field(default_factory=list)
    include_metrics: bool = True
    include_recommendations: bool = True
    save_report: bool = False
    report_path: Optional[str] = None

@dataclass
class ComponentAnalysis:
    """Analysis results for a single component"""
    component_name: str
    status: ComponentStatus
    health_score: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    performance_data: Dict[str, Any] = field(default_factory=dict)
    last_checked: datetime = field(default_factory=datetime.now)

@dataclass
class SystemAnalysisReport:
    """Comprehensive system analysis report"""
    analysis_id: str
    timestamp: datetime
    analysis_type: AnalysisType
    level: AnalysisLevel
    overall_health_score: float
    system_status: ComponentStatus
    component_analyses: List[ComponentAnalysis] = field(default_factory=list)
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    integration_status: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    issues_summary: Dict[str, int] = field(default_factory=dict)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    dharmic_compliance: Dict[str, Any] = field(default_factory=dict)

class AnalysisEngine:
    """
    Advanced System Analysis Engine
    
    This engine provides comprehensive analysis capabilities for the DharmaMind system,
    monitoring system health, performance, dharmic compliance, and component integrity.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Analysis configuration
        self.analysis_config = {
            "health_check_interval": 300,  # 5 minutes
            "performance_window": 3600,    # 1 hour
            "metric_retention": 86400,     # 24 hours
            "alert_thresholds": {
                "health_score": 0.7,
                "response_time": 5.0,
                "error_rate": 0.05
            }
        }
        
        # Component registry
        self.components = {}
        self.metrics_history = {}
        self.analysis_history = []
        
        # Analysis modules
        self.dharmic_validator = None
        self.consciousness_analyzer = None
        self.performance_monitor = None
        
        # Initialize async
        asyncio.create_task(self._initialize_components())
        
        self.logger.info("🔍 Analysis Engine initialized")
    
    async def _initialize_components(self):
        """Initialize analysis engine components"""
        
        try:
            # Initialize component connections
            await self._connect_to_modules()
            
            # Start background monitoring
            asyncio.create_task(self._background_monitoring())
            
            self.logger.info("✅ Analysis Engine components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing Analysis Engine: {str(e)}")
    
    async def _connect_to_modules(self):
        """Connect to other system modules"""
        
        try:
            # Connect to dharma engine for compliance analysis
            from .dharma_engine import get_dharma_engine
            self.dharmic_validator = get_dharma_engine()
            
            # Connect to consciousness core for consciousness analysis
            from .consciousness_core import get_consciousness_core
            self.consciousness_analyzer = get_consciousness_core()
            
            # Register known components
            self.components = {
                "consciousness_core": {"status": ComponentStatus.UNKNOWN, "last_check": None},
                "knowledge_base": {"status": ComponentStatus.UNKNOWN, "last_check": None},
                "emotional_intelligence": {"status": ComponentStatus.UNKNOWN, "last_check": None},
                "dharma_engine": {"status": ComponentStatus.UNKNOWN, "last_check": None},
                "ai_core": {"status": ComponentStatus.UNKNOWN, "last_check": None},
                "security_protection": {"status": ComponentStatus.UNKNOWN, "last_check": None},
                "system_orchestrator": {"status": ComponentStatus.UNKNOWN, "last_check": None},
                "llm_engine": {"status": ComponentStatus.UNKNOWN, "last_check": None}
            }
            
            self.logger.info("🔗 Connected to system modules")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not connect to all modules: {str(e)}")
    
    async def analyze_system(self, request: AnalysisRequest) -> SystemAnalysisReport:
        """
        Perform comprehensive system analysis
        
        Args:
            request: Analysis request with parameters
            
        Returns:
            SystemAnalysisReport with complete analysis results
        """
        
        start_time = datetime.now()
        analysis_id = f"analysis_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info(f"🔄 Starting {request.analysis_type.value} analysis...")
            
            # Initialize report
            report = SystemAnalysisReport(
                analysis_id=analysis_id,
                timestamp=start_time,
                analysis_type=request.analysis_type,
                level=request.level,
                overall_health_score=0.0,
                system_status=ComponentStatus.UNKNOWN
            )
            
            # Perform analysis based on type
            if request.analysis_type == AnalysisType.SYSTEM_HEALTH:
                await self._analyze_system_health(report, request)
            elif request.analysis_type == AnalysisType.PERFORMANCE:
                await self._analyze_performance(report, request)
            elif request.analysis_type == AnalysisType.DHARMIC_COMPLIANCE:
                await self._analyze_dharmic_compliance(report, request)
            elif request.analysis_type == AnalysisType.MODULE_INTEGRITY:
                await self._analyze_module_integrity(report, request)
            elif request.analysis_type == AnalysisType.INTEGRATION_STATUS:
                await self._analyze_integration_status(report, request)
            elif request.analysis_type == AnalysisType.SECURITY_ASSESSMENT:
                await self._analyze_security(report, request)
            elif request.analysis_type == AnalysisType.CONSCIOUSNESS_ANALYSIS:
                await self._analyze_consciousness(report, request)
            
            # Calculate overall health score
            report.overall_health_score = self._calculate_overall_health(report)
            report.system_status = self._determine_system_status(report.overall_health_score)
            
            # Summarize issues
            report.issues_summary = {
                status.value: sum(1 for a in report.component_analyses if a.status == status)
                for status in ComponentStatus
            }
            
            # Generate recommendations
            if request.include_recommendations:
                report.recommendations = await self._generate_recommendations(report)
            
            # Save report if requested
            if request.save_report:
                await self._save_report(report, request.report_path)
            
            # Add to history only if successful
            self.analysis_history.append(report)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"✅ Analysis completed in {processing_time:.2f}s")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error during analysis: {str(e)}")
            # Return error report with minimal info
            return SystemAnalysisReport(
                analysis_id=analysis_id,
                timestamp=start_time,
                analysis_type=request.analysis_type,
                level=request.level,
                overall_health_score=0.0,
                system_status=ComponentStatus.ERROR,
                recommendations=[f"Analysis failed: {str(e)}"],
                issues_summary={"error": 1}
            )
    
    async def _analyze_system_health(self, report: SystemAnalysisReport, request: AnalysisRequest):
        """Analyze overall system health"""
        
        try:
            # Analyze each component
            components_to_check = request.components if request.components else list(self.components.keys())
            
            for component_name in components_to_check:
                component_analysis = await self._analyze_component(component_name)
                report.component_analyses.append(component_analysis)
            
            # Collect system metrics
            if request.include_metrics:
                report.system_metrics = await self._collect_system_metrics()
            
            # Check integration points
            report.integration_status = await self._check_integration_points()
            
            self.logger.debug("🏥 System health analysis completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in system health analysis: {str(e)}")
    
    async def _analyze_component(self, component_name: str) -> ComponentAnalysis:
        """Analyze individual component"""
        
        try:
            self.logger.debug(f"🔍 Analyzing component: {component_name}")
            
            # Initialize component analysis
            analysis = ComponentAnalysis(
                component_name=component_name,
                status=ComponentStatus.UNKNOWN,
                health_score=0.0
            )
            
            # Get component instance
            component = await self._get_component_instance(component_name)
            
            if component is None:
                analysis.status = ComponentStatus.ERROR
                analysis.issues.append("Component not available")
                analysis.health_score = 0.0
                return analysis
            
            # Check component health
            health_score = await self._check_component_health(component)
            analysis.health_score = health_score
            
            # Collect metrics
            analysis.metrics = await self._collect_component_metrics(component)
            
            # Check for issues
            analysis.issues = await self._identify_component_issues(component)
            
            # Determine status
            if health_score >= 0.9:
                analysis.status = ComponentStatus.HEALTHY
            elif health_score >= 0.7:
                analysis.status = ComponentStatus.WARNING
            elif health_score >= 0.5:
                analysis.status = ComponentStatus.ERROR
            else:
                analysis.status = ComponentStatus.CRITICAL
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing component {component_name}: {str(e)}")
            return ComponentAnalysis(
                component_name=component_name,
                status=ComponentStatus.ERROR,
                health_score=0.0,
                issues=[f"Analysis error: {str(e)}"]
            )
    
    async def _get_component_instance(self, component_name: str):
        """Get instance of a component"""
        
        try:
            if component_name == "consciousness_core":
                from .consciousness_core import get_consciousness_core
                return get_consciousness_core()
            elif component_name == "knowledge_base":
                from .knowledge_base import get_knowledge_base
                return get_knowledge_base()
            elif component_name == "emotional_intelligence":
                from .emotional_intelligence import get_emotional_intelligence
                return get_emotional_intelligence()
            elif component_name == "dharma_engine":
                from .dharma_engine import get_dharma_engine
                return get_dharma_engine()
            elif component_name == "ai_core":
                from .ai_core import get_ai_core
                return get_ai_core()
            elif component_name == "security_protection":
                from .security_protection import get_protection_layer
                return get_protection_layer()
            elif component_name == "system_orchestrator":
                from .system_orchestrator import get_system_orchestrator
                return get_system_orchestrator()
            elif component_name == "llm_engine":
                from .llm_engine import get_llm_engine
                return get_llm_engine()
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting component {component_name}: {str(e)}")
            return None
    
    async def _check_component_health(self, component) -> float:
        """Check health of a component"""
        
        try:
            # Check if component has status method
            if hasattr(component, 'get_status'):
                status = component.get_status()
                
                # Check if it's a coroutine that needs to be awaited
                import inspect
                if inspect.iscoroutine(status):
                    status = await status
                
                if isinstance(status, dict):
                    return status.get('health_score', 0.8)
                else:
                    return 0.8
            
            # Basic health check - component exists and has methods
            method_count = len([m for m in dir(component) if not m.startswith('_')])
            
            if method_count > 10:
                return 0.9  # Well-implemented component
            elif method_count > 5:
                return 0.7  # Basic implementation
            else:
                return 0.5  # Minimal implementation
            
        except Exception as e:
            self.logger.error(f"❌ Error checking component health: {str(e)}")
            return 0.0
    
    async def _collect_component_metrics(self, component) -> Dict[str, Any]:
        """Collect metrics from a component"""
        
        metrics = {}
        
        try:
            # Check for metrics method
            if hasattr(component, 'get_metrics'):
                metrics = component.get_metrics()
            elif hasattr(component, 'get_stats'):
                metrics = component.get_stats()
            else:
                # Basic metrics
                metrics = {
                    "type": type(component).__name__,
                    "methods": len([m for m in dir(component) if not m.startswith('_')]),
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting component metrics: {str(e)}")
            metrics = {"error": str(e)}
        
        return metrics
    
    async def _identify_component_issues(self, component) -> List[str]:
        """Identify issues with a component"""
        
        issues = []
        
        try:
            # Check for issues method
            if hasattr(component, 'get_issues'):
                issues = component.get_issues()
            elif hasattr(component, 'check_health'):
                health_result = component.check_health()
                if hasattr(health_result, 'issues'):
                    issues = health_result.issues
            
            # Basic issue detection
            if not hasattr(component, '__init__'):
                issues.append("Component missing __init__ method")
            
            # Check for async methods if needed
            async_methods = [m for m in dir(component) if asyncio.iscoroutinefunction(getattr(component, m))]
            if len(async_methods) == 0:
                issues.append("No async methods found (may not support async operations)")
            
        except Exception as e:
            issues.append(f"Error during issue detection: {str(e)}")
        
        return issues
    
    async def _analyze_performance(self, report: SystemAnalysisReport, request: AnalysisRequest):
        """Analyze system performance"""
        
        try:
            # Collect performance metrics
            performance_data = {
                "response_times": await self._measure_response_times(),
                "throughput": await self._measure_throughput(),
                "resource_usage": await self._measure_resource_usage(),
                "error_rates": await self._calculate_error_rates()
            }
            
            report.performance_summary = performance_data
            
            self.logger.debug("📊 Performance analysis completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in performance analysis: {str(e)}")
    
    async def _analyze_dharmic_compliance(self, report: SystemAnalysisReport, request: AnalysisRequest):
        """Analyze dharmic compliance of the system"""
        
        try:
            if self.dharmic_validator:
                # Test system responses for dharmic compliance
                test_inputs = [
                    "What is the nature of truth?",
                    "How should we treat others?",
                    "What is the purpose of life?",
                    "How can we achieve peace?"
                ]
                
                compliance_results = []
                for test_input in test_inputs:
                    # This would test the system's responses
                    result = await self.dharmic_validator.assess_dharma_compliance(test_input)
                    compliance_results.append({
                        "input": test_input,
                        "compliance_score": result.dharma_score if hasattr(result, 'dharma_score') else 0.8,
                        "level": result.overall_level.value if hasattr(result, 'overall_level') else "positive"
                    })
                
                report.dharmic_compliance = {
                    "overall_score": sum(r["compliance_score"] for r in compliance_results) / len(compliance_results),
                    "test_results": compliance_results
                }
            
            self.logger.debug("🕉️ Dharmic compliance analysis completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in dharmic compliance analysis: {str(e)}")
    
    async def _analyze_module_integrity(self, report: SystemAnalysisReport, request: AnalysisRequest):
        """Analyze module integrity and completeness"""
        
        try:
            # Check module imports and dependencies
            integrity_issues = []
            
            for component_name in self.components.keys():
                try:
                    component = await self._get_component_instance(component_name)
                    if component is None:
                        integrity_issues.append(f"Cannot instantiate {component_name}")
                    else:
                        # Check for required methods
                        required_methods = ["__init__"]
                        for method in required_methods:
                            if not hasattr(component, method):
                                integrity_issues.append(f"{component_name} missing {method}")
                
                except Exception as e:
                    integrity_issues.append(f"Error checking {component_name}: {str(e)}")
            
            report.system_metrics["integrity_issues"] = integrity_issues
            
            self.logger.debug("🔧 Module integrity analysis completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in module integrity analysis: {str(e)}")
    
    async def _analyze_integration_status(self, report: SystemAnalysisReport, request: AnalysisRequest):
        """Analyze integration status between components"""
        
        try:
            integration_matrix = {}
            
            # Test basic integration between components
            for comp1 in self.components.keys():
                integration_matrix[comp1] = {}
                for comp2 in self.components.keys():
                    if comp1 != comp2:
                        # Basic integration test
                        integration_matrix[comp1][comp2] = await self._test_component_integration(comp1, comp2)
            
            report.integration_status = integration_matrix
            
            self.logger.debug("🔄 Integration status analysis completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in integration analysis: {str(e)}")
    
    async def _test_component_integration(self, comp1_name: str, comp2_name: str) -> str:
        """Test integration between two components"""
        
        try:
            comp1 = await self._get_component_instance(comp1_name)
            comp2 = await self._get_component_instance(comp2_name)
            
            if comp1 is None or comp2 is None:
                return "unavailable"
            
            # Basic integration test - both components exist
            return "connected"
            
        except Exception as e:
            return f"error: {str(e)}"
    
    async def _analyze_security(self, report: SystemAnalysisReport, request: AnalysisRequest):
        """Analyze security aspects of the system"""
        
        try:
            # Security analysis would go here
            security_status = {
                "protection_layer_active": False,
                "threat_detection": False,
                "access_controls": False
            }
            
            # Check if security protection is available
            try:
                from .security_protection import get_protection_layer
                security_component = get_protection_layer()
                if security_component:
                    security_status["protection_layer_active"] = True
                    
                    if hasattr(security_component, 'detect_threats'):
                        security_status["threat_detection"] = True
                    
                    if hasattr(security_component, 'validate_access'):
                        security_status["access_controls"] = True
            
            except Exception:
                pass
            
            report.system_metrics["security_status"] = security_status
            
            self.logger.debug("🔒 Security analysis completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in security analysis: {str(e)}")
    
    async def _analyze_consciousness(self, report: SystemAnalysisReport, request: AnalysisRequest):
        """Analyze consciousness processing capabilities"""
        
        try:
            if self.consciousness_analyzer:
                # Test consciousness processing
                test_input = "What is the nature of awareness?"
                
                try:
                    result = await self.consciousness_analyzer.process_input(test_input, "analysis_test")
                    
                    consciousness_metrics = {
                        "processing_available": True,
                        "awareness_streams": hasattr(self.consciousness_analyzer, 'awareness_streams'),
                        "insight_generation": hasattr(result, 'insights') if result else False
                    }
                    
                except Exception as e:
                    consciousness_metrics = {
                        "processing_available": False,
                        "error": str(e)
                    }
                
                report.system_metrics["consciousness_analysis"] = consciousness_metrics
            
            self.logger.debug("🧠 Consciousness analysis completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in consciousness analysis: {str(e)}")
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect overall system metrics"""
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "components_available": len([c for c in self.components.values() if c["status"] != ComponentStatus.ERROR]),
            "total_components": len(self.components),
            "uptime": "unknown",
            "memory_usage": "unknown",
            "cpu_usage": "unknown"
        }
        
        return metrics
    
    async def _check_integration_points(self) -> Dict[str, Any]:
        """Check integration points between components"""
        
        integration_status = {
            "orchestrator_available": False,
            "api_endpoints": 0,
            "message_queues": 0,
            "shared_resources": []
        }
        
        # Check if system orchestrator is available
        try:
            from .system_orchestrator import get_system_orchestrator
            orchestrator = get_system_orchestrator()
            if orchestrator:
                integration_status["orchestrator_available"] = True
        except Exception:
            pass
        
        return integration_status
    
    async def _measure_response_times(self) -> Dict[str, float]:
        """Measure response times for key operations"""
        
        response_times = {}
        
        # Test each component
        for component_name in self.components.keys():
            try:
                start_time = datetime.now()
                component = await self._get_component_instance(component_name)
                end_time = datetime.now()
                
                if component:
                    response_times[component_name] = (end_time - start_time).total_seconds()
                else:
                    response_times[component_name] = -1  # Failed
                    
            except Exception:
                response_times[component_name] = -1
        
        return response_times
    
    async def _measure_throughput(self) -> Dict[str, Any]:
        """Measure system throughput"""
        
        # This is a placeholder - real implementation would measure actual throughput
        return {
            "requests_per_second": 0,
            "processing_rate": 0,
            "queue_length": 0
        }
    
    async def _measure_resource_usage(self) -> Dict[str, Any]:
        """Measure resource usage"""
        
        # This is a placeholder - real implementation would measure actual resources
        return {
            "memory_mb": 0,
            "cpu_percent": 0,
            "disk_usage_mb": 0
        }
    
    async def _calculate_error_rates(self) -> Dict[str, float]:
        """Calculate error rates for components"""
        
        # This is a placeholder - real implementation would track actual errors
        error_rates = {}
        
        for component_name in self.components.keys():
            error_rates[component_name] = 0.0  # Assume no errors for now
        
        return error_rates
    
    def _calculate_overall_health(self, report: SystemAnalysisReport) -> float:
        """Calculate overall system health score"""
        
        if not report.component_analyses:
            return 0.0
        
        # Average health scores of all components
        total_score = sum(analysis.health_score for analysis in report.component_analyses)
        return total_score / len(report.component_analyses)
    
    def _determine_system_status(self, health_score: float) -> ComponentStatus:
        """Determine overall system status based on health score"""
        
        if health_score >= 0.9:
            return ComponentStatus.HEALTHY
        elif health_score >= 0.7:
            return ComponentStatus.WARNING
        elif health_score >= 0.5:
            return ComponentStatus.ERROR
        else:
            return ComponentStatus.CRITICAL
    
    async def _generate_recommendations(self, report: SystemAnalysisReport) -> List[str]:
        """Generate recommendations based on analysis results"""
        
        recommendations = []
        
        # Analyze component issues
        for analysis in report.component_analyses:
            if analysis.status == ComponentStatus.ERROR:
                recommendations.append(f"Fix critical issues in {analysis.component_name}")
            elif analysis.status == ComponentStatus.WARNING:
                recommendations.append(f"Address warnings in {analysis.component_name}")
            
            if analysis.health_score < 0.7:
                recommendations.append(f"Improve health of {analysis.component_name}")
        
        # System-level recommendations
        if report.overall_health_score < 0.8:
            recommendations.append("Overall system health needs improvement")
        
        if report.analysis_type == AnalysisType.PERFORMANCE:
            if "response_times" in report.performance_summary:
                slow_components = [k for k, v in report.performance_summary["response_times"].items() if v > 1.0]
                for component in slow_components:
                    recommendations.append(f"Optimize response time for {component}")
        
        return recommendations
    
    async def _save_report(self, report: SystemAnalysisReport, file_path: Optional[str]):
        """Save analysis report to file"""
        
        try:
            if file_path is None:
                file_path = f"analysis_report_{report.analysis_id}.json"
            
            # Convert report to dict for JSON serialization
            report_dict = {
                "analysis_id": report.analysis_id,
                "timestamp": report.timestamp.isoformat(),
                "analysis_type": report.analysis_type.value,
                "level": report.level.value,
                "overall_health_score": report.overall_health_score,
                "system_status": report.system_status.value,
                "component_analyses": [
                    {
                        "component_name": analysis.component_name,
                        "status": analysis.status.value,
                        "health_score": analysis.health_score,
                        "metrics": analysis.metrics,
                        "issues": analysis.issues,
                        "recommendations": analysis.recommendations
                    }
                    for analysis in report.component_analyses
                ],
                "system_metrics": report.system_metrics,
                "integration_status": report.integration_status,
                "recommendations": report.recommendations,
                "performance_summary": report.performance_summary,
                "dharmic_compliance": report.dharmic_compliance
            }
            
            with open(file_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            self.logger.info(f"📄 Report saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving report: {str(e)}")
    
    async def _background_monitoring(self):
        """Background monitoring task"""
        
        while True:
            try:
                # Perform regular health checks
                request = AnalysisRequest(
                    analysis_type=AnalysisType.SYSTEM_HEALTH,
                    level=AnalysisLevel.BASIC,
                    include_recommendations=False
                )
                
                await self.analyze_system(request)
                
                # Wait for next check
                await asyncio.sleep(self.analysis_config["health_check_interval"])
                
            except Exception as e:
                self.logger.error(f"❌ Error in background monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis engine status"""
        
        return {
            "engine": "AnalysisEngine",
            "components_tracked": len(self.components),
            "analysis_history": len(self.analysis_history),
            "last_analysis": self.analysis_history[-1].timestamp.isoformat() if self.analysis_history else None,
            "monitoring_active": True,
            "dharmic_validator": self.dharmic_validator is not None,
            "consciousness_analyzer": self.consciousness_analyzer is not None,
            "configuration": self.analysis_config
        }

# Global analysis engine instance
_analysis_engine = None

def get_analysis_engine() -> AnalysisEngine:
    """Get global analysis engine instance"""
    global _analysis_engine
    if _analysis_engine is None:
        _analysis_engine = AnalysisEngine()
    return _analysis_engine

# Export main classes
__all__ = [
    "AnalysisEngine",
    "get_analysis_engine",
    "AnalysisRequest",
    "SystemAnalysisReport",
    "ComponentAnalysis",
    "AnalysisType",
    "AnalysisLevel",
    "ComponentStatus"
]

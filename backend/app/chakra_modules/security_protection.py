"""
Security Protection Layer - Comprehensive Security System
=====================================================

This module provides comprehensive security protection for the DharmaMind system,
including dharmic compliance, content filtering, access control, and spiritual safeguards.
"""

import logging
import asyncio
import hashlib
import hmac
import time
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import secrets
import ipaddress

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProtectionType(Enum):
    """Types of protection"""
    DHARMIC_COMPLIANCE = "dharmic_compliance"
    CONTENT_FILTERING = "content_filtering"
    ACCESS_CONTROL = "access_control"
    RATE_LIMITING = "rate_limiting"
    INPUT_VALIDATION = "input_validation"
    OUTPUT_SANITIZATION = "output_sanitization"
    SPIRITUAL_SAFEGUARDS = "spiritual_safeguards"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"

class SecurityEvent(Enum):
    """Security event types"""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_INPUT = "malicious_input"
    DHARMIC_VIOLATION = "dharmic_violation"
    CONTENT_POLICY_VIOLATION = "content_policy_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SYSTEM_INTRUSION = "system_intrusion"

@dataclass
class SecurityIncident:
    """Security incident record"""
    incident_id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    description: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    actions_taken: List[str] = field(default_factory=list)

@dataclass
class ProtectionRule:
    """Protection rule definition"""
    rule_id: str
    name: str
    description: str
    protection_type: ProtectionType
    pattern: str
    action: str
    severity: ThreatLevel
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

class ProtectionLayer:
    """
    Comprehensive Protection Layer for DharmaMind
    
    This layer provides multi-layered security including dharmic compliance,
    content filtering, access control, and spiritual safeguards.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Protection components
        self.dharmic_validator = None
        self.content_filter = None
        self.access_controller = None
        self.rate_limiter = None
        
        # Security state
        self.security_incidents: List[SecurityIncident] = []
        self.blocked_ips: Set[str] = set()
        self.trusted_sources: Set[str] = set()
        self.suspicious_patterns: Dict[str, int] = {}
        
        # Protection rules
        self.protection_rules: Dict[str, ProtectionRule] = {}
        
        # Rate limiting state
        self.rate_limit_state: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            "max_requests_per_minute": 60,
            "max_requests_per_hour": 1000,
            "auto_block_threshold": 10,
            "dharmic_compliance_required": True,
            "content_filtering_enabled": True,
            "spiritual_safeguards_enabled": True,
            "cultural_sensitivity_required": True,
            "input_validation_enabled": True,
            "output_sanitization_enabled": True,
            "advanced_threat_detection": True
        }
        
        # Initialize protection components
        self._initialize_protection_sync()
        
        self.logger.info("🛡️ Protection Layer initialized with comprehensive security")
    
    def _initialize_protection_sync(self):
        """Initialize protection components synchronously"""
        
        try:
            # Initialize dharmic validator
            self.dharmic_validator = DharmicValidator()
            
            # Initialize content filter
            self.content_filter = ContentFilter()
            
            # Initialize access controller
            self.access_controller = AccessController()
            
            # Initialize rate limiter
            self.rate_limiter = RateLimiter(self.config)
            
            # Load protection rules
            self._load_protection_rules()
            
            # Initialize trusted sources
            self._initialize_trusted_sources()
            
            self.logger.info("✅ Protection components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing protection: {str(e)}")
    
    def _load_protection_rules(self):
        """Load comprehensive protection rules"""
        
        # Dharmic compliance rules
        dharmic_rules = [
            ProtectionRule(
                rule_id="dharmic_001",
                name="Ahimsa Violation Prevention",
                description="Block content promoting violence or harm (ahimsa principle)",
                protection_type=ProtectionType.DHARMIC_COMPLIANCE,
                pattern=r"\b(kill|murder|harm|violence|attack|destroy|hurt|damage)\b",
                action="block",
                severity=ThreatLevel.HIGH
            ),
            ProtectionRule(
                rule_id="dharmic_002",
                name="Hatred and Discrimination Prevention",
                description="Block content promoting hatred or discrimination",
                protection_type=ProtectionType.DHARMIC_COMPLIANCE,
                pattern=r"\b(hate|hatred|discriminate|racist|bigot|prejudice)\b",
                action="block",
                severity=ThreatLevel.HIGH
            ),
            ProtectionRule(
                rule_id="dharmic_003",
                name="Truthfulness Verification",
                description="Monitor content for truthfulness and accuracy (satya principle)",
                protection_type=ProtectionType.DHARMIC_COMPLIANCE,
                pattern=r"\b(lie|false|deceive|mislead|cheat|fraud)\b",
                action="warn",
                severity=ThreatLevel.MEDIUM
            ),
            ProtectionRule(
                rule_id="dharmic_004",
                name="Sacred Disrespect Prevention",
                description="Prevent disrespectful treatment of sacred concepts",
                protection_type=ProtectionType.DHARMIC_COMPLIANCE,
                pattern=r"\b(mock|ridicule|insult|blaspheme)\b.*\b(god|divine|sacred|holy)\b",
                action="block",
                severity=ThreatLevel.CRITICAL
            )
        ]
        
        # Content filtering rules
        content_rules = [
            ProtectionRule(
                rule_id="content_001",
                name="Inappropriate Content Filter",
                description="Block inappropriate or offensive content",
                protection_type=ProtectionType.CONTENT_FILTERING,
                pattern=r"\b(inappropriate|offensive|vulgar|obscene|profanity)\b",
                action="block",
                severity=ThreatLevel.MEDIUM
            ),
            ProtectionRule(
                rule_id="content_002",
                name="Spam Detection",
                description="Detect and block spam content",
                protection_type=ProtectionType.CONTENT_FILTERING,
                pattern=r"\b(spam|promotional|advertisement|sell|buy now|click here)\b",
                action="warn",
                severity=ThreatLevel.LOW
            ),
            ProtectionRule(
                rule_id="content_003",
                name="Malicious Content Detection",
                description="Detect potentially malicious content",
                protection_type=ProtectionType.CONTENT_FILTERING,
                pattern=r"\b(hack|crack|exploit|malware|virus|trojan)\b",
                action="block",
                severity=ThreatLevel.HIGH
            )
        ]
        
        # Spiritual safeguard rules
        spiritual_rules = [
            ProtectionRule(
                rule_id="spiritual_001",
                name="Sacred Text Respect",
                description="Ensure respectful treatment of sacred texts",
                protection_type=ProtectionType.SPIRITUAL_SAFEGUARDS,
                pattern=r"\b(mock|ridicule|disrespect|joke about)\b.*\b(scripture|gita|veda|upanishad|bible|quran|torah)\b",
                action="block",
                severity=ThreatLevel.HIGH
            ),
            ProtectionRule(
                rule_id="spiritual_002",
                name="Deity Respect Protection",
                description="Ensure respectful treatment of deities across traditions",
                protection_type=ProtectionType.SPIRITUAL_SAFEGUARDS,
                pattern=r"\b(mock|ridicule|insult|blaspheme)\b.*\b(krishna|rama|shiva|vishnu|brahma|devi|ganesha|buddha|christ|allah)\b",
                action="block",
                severity=ThreatLevel.CRITICAL
            ),
            ProtectionRule(
                rule_id="spiritual_003",
                name="Guru and Teacher Respect",
                description="Ensure respectful treatment of spiritual teachers",
                protection_type=ProtectionType.SPIRITUAL_SAFEGUARDS,
                pattern=r"\b(mock|ridicule|insult)\b.*\b(guru|teacher|sage|saint|master|swami)\b",
                action="block",
                severity=ThreatLevel.HIGH
            ),
            ProtectionRule(
                rule_id="spiritual_004",
                name="Ritual and Practice Respect",
                description="Ensure respectful treatment of spiritual practices",
                protection_type=ProtectionType.SPIRITUAL_SAFEGUARDS,
                pattern=r"\b(mock|ridicule|joke)\b.*\b(puja|aarti|meditation|prayer|ritual|ceremony)\b",
                action="warn",
                severity=ThreatLevel.MEDIUM
            )
        ]
        
        # Input validation rules
        validation_rules = [
            ProtectionRule(
                rule_id="validation_001",
                name="SQL Injection Prevention",
                description="Prevent SQL injection attacks",
                protection_type=ProtectionType.INPUT_VALIDATION,
                pattern=r"\b(select|insert|update|delete|drop|union|exec)\b.*\b(from|into|where|table)\b",
                action="block",
                severity=ThreatLevel.CRITICAL
            ),
            ProtectionRule(
                rule_id="validation_002",
                name="Script Injection Prevention",
                description="Prevent script injection attacks",
                protection_type=ProtectionType.INPUT_VALIDATION,
                pattern=r"<script|javascript:|vbscript:|onload=|onerror=|eval\(|document\.",
                action="block",
                severity=ThreatLevel.HIGH
            ),
            ProtectionRule(
                rule_id="validation_003",
                name="Command Injection Prevention",
                description="Prevent command injection attacks",
                protection_type=ProtectionType.INPUT_VALIDATION,
                pattern=r"\b(exec|eval|system|shell|cmd|powershell)\b\s*[\(\[]",
                action="block",
                severity=ThreatLevel.CRITICAL
            )
        ]
        
        # Cultural sensitivity rules
        cultural_rules = [
            ProtectionRule(
                rule_id="cultural_001",
                name="Cultural Stereotype Prevention",
                description="Prevent cultural stereotyping",
                protection_type=ProtectionType.CULTURAL_SENSITIVITY,
                pattern=r"\b(all (hindus|muslims|christians|buddhists|indians))\b.*\b(are|always|never)\b",
                action="warn",
                severity=ThreatLevel.MEDIUM
            ),
            ProtectionRule(
                rule_id="cultural_002",
                name="Religious Oversimplification Prevention",
                description="Prevent oversimplification of religious concepts",
                protection_type=ProtectionType.CULTURAL_SENSITIVITY,
                pattern=r"\b(just|merely|simply|only)\b.*\b(ritual|tradition|belief|practice)\b",
                action="warn",
                severity=ThreatLevel.LOW
            )
        ]
        
        # Combine all rules
        all_rules = dharmic_rules + content_rules + spiritual_rules + validation_rules + cultural_rules
        
        for rule in all_rules:
            self.protection_rules[rule.rule_id] = rule
        
        self.logger.info(f"📋 Loaded {len(self.protection_rules)} comprehensive protection rules")
    
    def _initialize_trusted_sources(self):
        """Initialize trusted sources"""
        
        self.trusted_sources.update([
            "127.0.0.1",
            "localhost",
            "::1",
            "0.0.0.0"
        ])
        
        self.logger.info(f"🔒 Initialized {len(self.trusted_sources)} trusted sources")
    
    async def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive request validation for security and dharmic compliance
        
        Args:
            request_data: Request data to validate
            
        Returns:
            Dict containing detailed validation results
        """
        
        try:
            validation_result = {
                "valid": True,
                "threats_detected": [],
                "actions_taken": [],
                "warnings": [],
                "dharmic_compliance": True,
                "cultural_sensitivity": True,
                "spiritual_safety": True,
                "protection_level": "secure",
                "confidence_score": 1.0,
                "processing_time": 0.0
            }
            
            start_time = time.time()
            
            # Extract request information
            content = request_data.get("content", "")
            source_ip = request_data.get("source_ip", "unknown")
            user_id = request_data.get("user_id")
            request_type = request_data.get("type", "general")
            context = request_data.get("context", {})
            
            self.logger.debug(f"🔍 Validating request from {source_ip}: {content[:50]}...")
            
            # 1. Rate limiting check
            if source_ip not in self.trusted_sources:
                rate_check = await self.rate_limiter.check_rate_limit(source_ip, user_id)
                if not rate_check["allowed"]:
                    validation_result["valid"] = False
                    validation_result["threats_detected"].append("rate_limit_exceeded")
                    validation_result["actions_taken"].append("request_blocked")
                    
                    await self._log_security_event(
                        SecurityEvent.RATE_LIMIT_EXCEEDED,
                        source_ip,
                        user_id,
                        {"reason": rate_check["reason"]}
                    )
                    
                    return validation_result
            
            # 2. IP blocking check
            if source_ip in self.blocked_ips:
                validation_result["valid"] = False
                validation_result["threats_detected"].append("blocked_ip")
                validation_result["actions_taken"].append("request_blocked")
                
                await self._log_security_event(
                    SecurityEvent.UNAUTHORIZED_ACCESS,
                    source_ip,
                    user_id,
                    {"reason": "blocked_ip"}
                )
                
                return validation_result
            
            # 3. Input validation for security threats
            if self.config["input_validation_enabled"]:
                input_result = await self._validate_input_security(content)
                if not input_result["valid"]:
                    validation_result["valid"] = False
                    validation_result["threats_detected"].extend(input_result["threats"])
                    validation_result["actions_taken"].append("input_blocked")
                    
                    await self._log_security_event(
                        SecurityEvent.MALICIOUS_INPUT,
                        source_ip,
                        user_id,
                        input_result
                    )
                    
                    return validation_result
            
            # 4. Dharmic compliance validation
            if self.config["dharmic_compliance_required"]:
                dharmic_result = await self.dharmic_validator.validate(content)
                
                validation_result["dharmic_compliance"] = dharmic_result["compliant"]
                
                if not dharmic_result["compliant"]:
                    validation_result["threats_detected"].extend(dharmic_result["violations"])
                    
                    if dharmic_result["severity"] in ["high", "critical"]:
                        validation_result["valid"] = False
                        validation_result["actions_taken"].append("dharmic_violation_blocked")
                        
                        await self._log_security_event(
                            SecurityEvent.DHARMIC_VIOLATION,
                            source_ip,
                            user_id,
                            dharmic_result
                        )
                    else:
                        validation_result["warnings"].extend(dharmic_result["warnings"])
            
            # 5. Content filtering
            if self.config["content_filtering_enabled"]:
                content_result = await self.content_filter.filter_content(content)
                
                if content_result["filtered"]:
                    validation_result["threats_detected"].extend(content_result["violations"])
                    
                    if content_result["severity"] in ["high", "critical"]:
                        validation_result["valid"] = False
                        validation_result["actions_taken"].append("content_filtered")
                        
                        await self._log_security_event(
                            SecurityEvent.CONTENT_POLICY_VIOLATION,
                            source_ip,
                            user_id,
                            content_result
                        )
                    else:
                        validation_result["warnings"].extend(content_result["warnings"])
            
            # 6. Spiritual safeguards
            if self.config["spiritual_safeguards_enabled"]:
                spiritual_result = await self._check_spiritual_safeguards(content)
                
                validation_result["spiritual_safety"] = spiritual_result["safe"]
                
                if not spiritual_result["safe"]:
                    validation_result["threats_detected"].extend(spiritual_result["violations"])
                    
                    if spiritual_result["severity"] in ["high", "critical"]:
                        validation_result["valid"] = False
                        validation_result["actions_taken"].append("spiritual_safeguard_triggered")
                        
                        await self._log_security_event(
                            SecurityEvent.DHARMIC_VIOLATION,
                            source_ip,
                            user_id,
                            spiritual_result
                        )
            
            # 7. Cultural sensitivity check
            if self.config["cultural_sensitivity_required"]:
                cultural_result = await self._check_cultural_sensitivity(content)
                
                validation_result["cultural_sensitivity"] = cultural_result["sensitive"]
                
                if not cultural_result["sensitive"]:
                    validation_result["warnings"].extend(cultural_result["concerns"])
            
            # 8. Advanced threat detection
            if self.config["advanced_threat_detection"]:
                threat_result = await self._advanced_threat_detection(content, source_ip)
                
                if threat_result["threats_found"]:
                    validation_result["threats_detected"].extend(threat_result["threats"])
                    validation_result["actions_taken"].extend(threat_result["actions"])
            
            # 9. Pattern analysis for suspicious behavior
            await self._analyze_suspicious_patterns(content, source_ip, user_id)
            
            # Calculate confidence score
            validation_result["confidence_score"] = self._calculate_confidence_score(validation_result)
            
            # Set protection level
            validation_result["protection_level"] = self._determine_protection_level(validation_result)
            
            # Processing time
            validation_result["processing_time"] = time.time() - start_time
            
            # Log validation if needed
            if validation_result["threats_detected"] or not validation_result["valid"]:
                self.logger.warning(f"⚠️ Validation issues detected: {validation_result['threats_detected']}")
            else:
                self.logger.debug(f"✅ Request validation passed in {validation_result['processing_time']:.3f}s")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Error validating request: {str(e)}")
            
            return {
                "valid": False,
                "error": str(e),
                "actions_taken": ["error_blocked"],
                "protection_level": "error",
                "confidence_score": 0.0
            }
    
    async def _validate_input_security(self, content: str) -> Dict[str, Any]:
        """Validate input for security threats"""
        
        result = {
            "valid": True,
            "threats": [],
            "severity": "low"
        }
        
        # Check against validation rules
        for rule_id, rule in self.protection_rules.items():
            if rule.protection_type == ProtectionType.INPUT_VALIDATION and rule.enabled:
                if re.search(rule.pattern, content, re.IGNORECASE):
                    result["valid"] = False
                    result["threats"].append(rule.name)
                    
                    if rule.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                        result["severity"] = rule.severity.value
        
        return result
    
    async def _check_spiritual_safeguards(self, content: str) -> Dict[str, Any]:
        """Comprehensive spiritual safeguards checking"""
        
        result = {
            "safe": True,
            "violations": [],
            "severity": "low",
            "details": []
        }
        
        content_lower = content.lower()
        
        # Check against spiritual safeguard rules
        for rule_id, rule in self.protection_rules.items():
            if rule.protection_type == ProtectionType.SPIRITUAL_SAFEGUARDS and rule.enabled:
                if re.search(rule.pattern, content_lower, re.IGNORECASE):
                    result["safe"] = False
                    result["violations"].append(rule.name)
                    result["details"].append({
                        "rule": rule.name,
                        "description": rule.description,
                        "severity": rule.severity.value
                    })
                    
                    if rule.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                        result["severity"] = rule.severity.value
        
        # Additional spiritual protection checks
        sacred_contexts = [
            ("om_misuse", r"\bom\b.*\b(joke|funny|casual|random|whatever)\b"),
            ("mantra_trivialization", r"\bmantra\b.*\b(fun|game|casual|entertainment)\b"),
            ("meditation_mockery", r"\bmeditation\b.*\b(stupid|waste|useless|nonsense)\b"),
            ("prayer_disrespect", r"\bprayer\b.*\b(pointless|useless|stupid|waste)\b")
        ]
        
        for violation_type, pattern in sacred_contexts:
            if re.search(pattern, content_lower, re.IGNORECASE):
                result["safe"] = False
                result["violations"].append(violation_type)
                result["severity"] = "high"
        
        return result
    
    async def _check_cultural_sensitivity(self, content: str) -> Dict[str, Any]:
        """Comprehensive cultural sensitivity checking"""
        
        result = {
            "sensitive": True,
            "concerns": [],
            "suggestions": []
        }
        
        content_lower = content.lower()
        
        # Check against cultural sensitivity rules
        for rule_id, rule in self.protection_rules.items():
            if rule.protection_type == ProtectionType.CULTURAL_SENSITIVITY and rule.enabled:
                if re.search(rule.pattern, content_lower, re.IGNORECASE):
                    result["sensitive"] = False
                    result["concerns"].append(rule.name)
                    result["suggestions"].append(f"Consider: {rule.description}")
        
        # Additional cultural sensitivity checks
        sensitivity_patterns = [
            ("appropriation_risk", r"\b(taking|using|borrowing|adopting)\b.*\b(tradition|ritual|practice|belief)\b"),
            ("cultural_hierarchy", r"\b(advanced|primitive|better|worse)\b.*\b(culture|tradition|religion)\b"),
            ("religious_competition", r"\b(best|superior|perfect)\b.*\b(religion|faith|tradition)\b"),
            ("practice_trivialization", r"\b(easy|simple|basic)\b.*\b(enlightenment|moksha|nirvana|salvation)\b")
        ]
        
        for concern_type, pattern in sensitivity_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                result["sensitive"] = False
                result["concerns"].append(concern_type)
        
        return result
    
    async def _advanced_threat_detection(self, content: str, source_ip: str) -> Dict[str, Any]:
        """Advanced threat detection using pattern analysis"""
        
        result = {
            "threats_found": False,
            "threats": [],
            "actions": [],
            "confidence": 0.0
        }
        
        # Behavioral analysis patterns
        suspicious_patterns = [
            ("rapid_fire_queries", r"(\w+\s+){1,3}\?\s*(\w+\s+){1,3}\?\s*(\w+\s+){1,3}\?"),
            ("system_probing", r"\b(admin|root|password|login|auth|token|key|secret)\b"),
            ("information_harvesting", r"\b(database|server|system|config|settings|internal)\b"),
            ("social_engineering", r"\b(urgent|immediate|emergency|help|assist|please)\b.*\b(access|permission|override)\b")
        ]
        
        threat_score = 0.0
        
        for threat_type, pattern in suspicious_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                result["threats_found"] = True
                result["threats"].append(threat_type)
                threat_score += len(matches) * 0.1
        
        # IP reputation check
        if source_ip not in self.trusted_sources:
            ip_incidents = [i for i in self.security_incidents if i.source_ip == source_ip]
            if len(ip_incidents) > 3:
                threat_score += 0.3
                result["threats_found"] = True
                result["threats"].append("high_incident_ip")
        
        # Content length analysis
        if len(content) > 5000:
            threat_score += 0.1
            result["threats"].append("excessive_content_length")
        
        result["confidence"] = min(1.0, threat_score)
        
        # Determine actions based on threat level
        if threat_score > 0.7:
            result["actions"].append("high_priority_monitoring")
        elif threat_score > 0.4:
            result["actions"].append("increased_monitoring")
        
        return result
    
    async def _analyze_suspicious_patterns(self, content: str, source_ip: str, user_id: Optional[str]):
        """Analyze patterns for suspicious behavior"""
        
        # Track pattern frequency
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        pattern_key = f"{source_ip}:{content_hash}"
        
        if pattern_key not in self.suspicious_patterns:
            self.suspicious_patterns[pattern_key] = 0
        
        self.suspicious_patterns[pattern_key] += 1
        
        # Alert on repeated patterns
        if self.suspicious_patterns[pattern_key] > 5:
            await self._log_security_event(
                SecurityEvent.SUSPICIOUS_ACTIVITY,
                source_ip,
                user_id,
                {"pattern": "repeated_content", "count": self.suspicious_patterns[pattern_key]}
            )
    
    def _calculate_confidence_score(self, validation_result: Dict[str, Any]) -> float:
        """Calculate confidence score for validation"""
        
        base_score = 1.0
        
        # Reduce score for each threat
        threat_count = len(validation_result["threats_detected"])
        base_score -= threat_count * 0.1
        
        # Reduce score for compliance issues
        if not validation_result["dharmic_compliance"]:
            base_score -= 0.2
        
        if not validation_result["cultural_sensitivity"]:
            base_score -= 0.1
        
        if not validation_result["spiritual_safety"]:
            base_score -= 0.2
        
        return max(0.0, min(1.0, base_score))
    
    def _determine_protection_level(self, validation_result: Dict[str, Any]) -> str:
        """Determine protection level based on validation result"""
        
        if not validation_result["valid"]:
            return "blocked"
        elif validation_result["threats_detected"]:
            return "monitored"
        elif validation_result["warnings"]:
            return "cautioned"
        else:
            return "secure"
    
    async def sanitize_output(self, output: str, context: Optional[Dict] = None) -> str:
        """Comprehensive output sanitization"""
        
        try:
            if not self.config["output_sanitization_enabled"]:
                return output
            
            sanitized = output
            
            # Remove potential HTML/script tags
            sanitized = re.sub(r'<[^>]*>', '', sanitized)
            
            # Escape special characters
            sanitized = sanitized.replace('&', '&amp;')
            sanitized = sanitized.replace('<', '&lt;')
            sanitized = sanitized.replace('>', '&gt;')
            sanitized = sanitized.replace('"', '&quot;')
            sanitized = sanitized.replace("'", '&#x27;')
            
            # Remove potential code injection attempts
            sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
            
            # Clean excessive whitespace
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()
            
            # Ensure dharmic compliance in output
            if context and context.get("require_dharmic_check", True):
                dharmic_result = await self.dharmic_validator.validate(sanitized)
                if not dharmic_result["compliant"]:
                    self.logger.warning("⚠️ Output failed dharmic compliance, applying corrections")
                    sanitized = await self._apply_dharmic_corrections(sanitized, dharmic_result)
            
            return sanitized
            
        except Exception as e:
            self.logger.error(f"❌ Error sanitizing output: {str(e)}")
            return "[Content sanitization error - output blocked for safety]"
    
    async def _apply_dharmic_corrections(self, content: str, dharmic_result: Dict) -> str:
        """Apply dharmic corrections to content"""
        
        corrected = content
        
        # Replace problematic words with dharmic alternatives
        dharmic_replacements = {
            "kill": "transform",
            "destroy": "transcend",
            "hate": "understand",
            "attack": "address",
            "violence": "peaceful action",
            "harm": "heal",
            "enemy": "teacher",
            "revenge": "forgiveness"
        }
        
        for problematic, dharmic in dharmic_replacements.items():
            corrected = re.sub(rf'\b{problematic}\b', dharmic, corrected, flags=re.IGNORECASE)
        
        return corrected
    
    async def _log_security_event(
        self,
        event_type: SecurityEvent,
        source_ip: str,
        user_id: Optional[str] = None,
        details: Dict[str, Any] = None
    ):
        """Log comprehensive security event"""
        
        incident_id = secrets.token_hex(8)
        
        if details is None:
            details = {}
        
        # Determine threat level based on event type
        threat_level_mapping = {
            SecurityEvent.SYSTEM_INTRUSION: ThreatLevel.CRITICAL,
            SecurityEvent.MALICIOUS_INPUT: ThreatLevel.CRITICAL,
            SecurityEvent.DHARMIC_VIOLATION: ThreatLevel.HIGH,
            SecurityEvent.UNAUTHORIZED_ACCESS: ThreatLevel.HIGH,
            SecurityEvent.CONTENT_POLICY_VIOLATION: ThreatLevel.MEDIUM,
            SecurityEvent.RATE_LIMIT_EXCEEDED: ThreatLevel.MEDIUM,
            SecurityEvent.SUSPICIOUS_ACTIVITY: ThreatLevel.LOW
        }
        
        threat_level = threat_level_mapping.get(event_type, ThreatLevel.MEDIUM)
        
        incident = SecurityIncident(
            incident_id=incident_id,
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            description=f"Security event: {event_type.value}",
            details=details,
            timestamp=datetime.now()
        )
        
        self.security_incidents.append(incident)
        
        # Auto-block IP if threshold exceeded
        ip_incidents = [i for i in self.security_incidents if i.source_ip == source_ip]
        if len(ip_incidents) >= self.config["auto_block_threshold"]:
            self.blocked_ips.add(source_ip)
            incident.actions_taken.append("ip_auto_blocked")
            self.logger.warning(f"🚫 Auto-blocked IP {source_ip} after {len(ip_incidents)} incidents")
        
        # Log with appropriate level
        if threat_level == ThreatLevel.CRITICAL:
            self.logger.critical(f"🚨 CRITICAL: {incident_id} - {event_type.value} from {source_ip}")
        elif threat_level == ThreatLevel.HIGH:
            self.logger.error(f"⚠️ HIGH: {incident_id} - {event_type.value} from {source_ip}")
        else:
            self.logger.warning(f"⚡ {threat_level.value.upper()}: {incident_id} - {event_type.value} from {source_ip}")

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""      
        recent_incidents = [
            i for i in self.security_incidents
            if (datetime.now() - i.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        return {
            "protection_layer": "active",
            "total_incidents": len(self.security_incidents),
            "recent_incidents": len(recent_incidents),
            "blocked_ips": len(self.blocked_ips),
            "trusted_sources": len(self.trusted_sources),
            "protection_rules": len(self.protection_rules),
            "suspicious_patterns": len(self.suspicious_patterns),
            "threat_levels": {
                level.value: len([i for i in recent_incidents if i.threat_level == level])
                for level in ThreatLevel
            },
            "protection_types": {
                ptype.value: len([r for r in self.protection_rules.values() 
                                if r.protection_type == ptype and r.enabled])
                for ptype in ProtectionType
            },
            "configuration": {
                "dharmic_compliance": self.config["dharmic_compliance_required"],
                "content_filtering": self.config["content_filtering_enabled"],
                "spiritual_safeguards": self.config["spiritual_safeguards_enabled"],
                "cultural_sensitivity": self.config["cultural_sensitivity_required"],
                "input_validation": self.config["input_validation_enabled"],
                "output_sanitization": self.config["output_sanitization_enabled"],
                "advanced_threat_detection": self.config["advanced_threat_detection"]
            },
            "performance": {
                "auto_block_threshold": self.config["auto_block_threshold"],
                "rate_limit_per_minute": self.config["max_requests_per_minute"],
                "rate_limit_per_hour": self.config["max_requests_per_hour"]
            }
        }
    
    def get_recent_incidents(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent security incidents with full details"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_incidents = [
            i for i in self.security_incidents
            if i.timestamp >= cutoff_time
        ]
        
        return [
            {
                "incident_id": i.incident_id,
                "event_type": i.event_type.value,
                "threat_level": i.threat_level.value,
                "source_ip": i.source_ip,
                "user_id": i.user_id,
                "description": i.description,
                "details": i.details,
                "timestamp": i.timestamp.isoformat(),
                "resolved": i.resolved,
                "actions_taken": i.actions_taken
            }
            for i in recent_incidents
        ]
    
    async def whitelist_ip(self, ip_address: str, reason: str = "manual_whitelist"):
        """Add IP to trusted sources"""
        
        try:
            # Validate IP address
            ipaddress.ip_address(ip_address)
            
            self.trusted_sources.add(ip_address)
            
            # Remove from blocked IPs if present
            if ip_address in self.blocked_ips:
                self.blocked_ips.remove(ip_address)
            
            self.logger.info(f"✅ IP {ip_address} added to trusted sources: {reason}")
            
        except ValueError:
            self.logger.error(f"❌ Invalid IP address: {ip_address}")
    
    async def block_ip(self, ip_address: str, reason: str = "manual_block"):
        """Block IP address"""
        
        try:
            # Validate IP address
            ipaddress.ip_address(ip_address)
            
            self.blocked_ips.add(ip_address)
            
            # Remove from trusted sources if present
            if ip_address in self.trusted_sources:
                self.trusted_sources.remove(ip_address)
            
            self.logger.warning(f"🚫 IP {ip_address} blocked: {reason}")
            
        except ValueError:
            self.logger.error(f"❌ Invalid IP address: {ip_address}")
    
    def get_protection_metrics(self) -> Dict[str, Any]:
        """Get detailed protection metrics"""
        
        total_rules = len(self.protection_rules)
        enabled_rules = len([r for r in self.protection_rules.values() if r.enabled])
        
        rule_distribution = {}
        for ptype in ProtectionType:
            rule_distribution[ptype.value] = len([
                r for r in self.protection_rules.values() 
                if r.protection_type == ptype
            ])
        
        incident_distribution = {}
        for event_type in SecurityEvent:
            incident_distribution[event_type.value] = len([
                i for i in self.security_incidents 
                if i.event_type == event_type
            ])
        
        return {
            "rules": {
                "total": total_rules,
                "enabled": enabled_rules,
                "disabled": total_rules - enabled_rules,
                "distribution": rule_distribution
            },
            "incidents": {
                "total": len(self.security_incidents),
                "distribution": incident_distribution,
                "resolved": len([i for i in self.security_incidents if i.resolved]),
                "unresolved": len([i for i in self.security_incidents if not i.resolved])
            },
            "blocking": {
                "blocked_ips": len(self.blocked_ips),
                "trusted_sources": len(self.trusted_sources),
                "auto_block_threshold": self.config["auto_block_threshold"]
            },
            "detection": {
                "suspicious_patterns": len(self.suspicious_patterns),
                "advanced_detection": self.config["advanced_threat_detection"]
            }
        }

# Supporting Security Classes
class DharmicValidator:
    """Advanced dharmic compliance validator"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dharmic_principles = self._load_dharmic_principles()
    
    def _load_dharmic_principles(self) -> Dict[str, Any]:
        """Load comprehensive dharmic principles"""
        
        return {
            "ahimsa": {
                "description": "Non-violence in thought, word, and deed",
                "violations": [
                    r"\b(kill|murder|harm|violence|attack|destroy|hurt|injure|wound)\b",
                    r"\b(fight|battle|war|combat|assault|abuse)\b"
                ],
                "severity": "high"
            },
            "satya": {
                "description": "Truthfulness and honesty",
                "violations": [
                    r"\b(lie|false|deceive|mislead|cheat|fraud|deception)\b",
                    r"\b(fake|fabricate|distort|manipulate)\b"
                ],
                "severity": "medium"
            },
            "asteya": {
                "description": "Non-stealing",
                "violations": [
                    r"\b(steal|theft|rob|piracy|plagiarism|copyright infringement)\b",
                    r"\b(unauthorized|illegal download|crack|hack)\b"
                ],
                "severity": "medium"
            },
            "brahmacharya": {
                "description": "Energy conservation and purity",
                "violations": [
                    r"\b(inappropriate|vulgar|crude|obscene|lewd)\b",
                    r"\b(sexual misconduct|inappropriate behavior)\b"
                ],
                "severity": "medium"
            },
            "aparigraha": {
                "description": "Non-possessiveness",
                "violations": [
                    r"\b(greed|hoarding|excessive|materialistic)\b",
                    r"\b(exploitation|selfish|accumulation)\b"
                ],
                "severity": "low"
            }
        }
    
    async def validate(self, content: str) -> Dict[str, Any]:
        """Comprehensive dharmic validation"""
        
        result = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "severity": "low",
            "principles_violated": [],
            "dharmic_score": 1.0
        }
        
        content_lower = content.lower()
        violation_count = 0
        
        # Check each dharmic principle
        for principle, config in self.dharmic_principles.items():
            for pattern in config["violations"]:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    result["compliant"] = False
                    result["violations"].append(f"{principle}_violation")
                    result["principles_violated"].append(principle)
                    violation_count += 1
                    
                    if config["severity"] == "high":
                        result["severity"] = "high"
                    elif config["severity"] == "medium" and result["severity"] == "low":
                        result["severity"] = "medium"
        
        # Calculate dharmic score
        result["dharmic_score"] = max(0.0, 1.0 - (violation_count * 0.2))
        
        # Add warnings for borderline content
        warning_patterns = [
            r"\b(anger|frustration|annoyance|irritation)\b",
            r"\b(competition|comparison|jealousy|envy)\b",
            r"\b(material|wealth|money|possession)\b.*\b(important|priority|focus)\b"
        ]
        
        for pattern in warning_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                result["warnings"].append("potential_dharmic_concern")
        
        return result

class ContentFilter:
    """Advanced content filtering system"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.filter_categories = self._load_filter_categories()
    
    def _load_filter_categories(self) -> Dict[str, Any]:
        """Load content filter categories"""
        
        return {
            "inappropriate": {
                "patterns": [
                    r"\b(inappropriate|offensive|vulgar|obscene|profanity)\b",
                    r"\b(nsfw|adult|explicit|mature)\b"
                ],
                "severity": "medium"
            },
            "spam": {
                "patterns": [
                    r"\b(spam|promotional|advertisement|marketing)\b",
                    r"\b(buy now|click here|limited time|special offer|deal)\b",
                    r"\b(make money|get rich|earn cash|free money)\b"
                ],
                "severity": "low"
            },
            "malicious": {
                "patterns": [
                    r"\b(hack|crack|exploit|malware|virus|trojan)\b",
                    r"\b(phishing|scam|fraud|identity theft)\b",
                    r"\b(illegal|criminal|unlawful|prohibited)\b"
                ],
                "severity": "high"
            },
            "misinformation": {
                "patterns": [
                    r"\b(conspiracy|fake news|hoax|rumor)\b",
                    r"\b(unverified|unconfirmed|alleged|supposedly)\b"
                ],
                "severity": "medium"
            }
        }
    
    async def filter_content(self, content: str) -> Dict[str, Any]:
        """Advanced content filtering"""
        
        result = {
            "filtered": False,
            "violations": [],
            "warnings": [],
            "severity": "low",
            "categories_triggered": [],
            "confidence": 0.0
        }
        
        content_lower = content.lower()
        violation_score = 0.0
        
        # Check each filter category
        for category, config in self.filter_categories.items():
            category_triggered = False
            
            for pattern in config["patterns"]:
                matches = re.findall(pattern, content_lower, re.IGNORECASE)
                if matches:
                    result["filtered"] = True
                    result["violations"].append(f"{category}_content")
                    result["categories_triggered"].append(category)
                    category_triggered = True
                    violation_score += len(matches) * 0.1
                    
                    if config["severity"] == "high":
                        result["severity"] = "high"
                    elif config["severity"] == "medium" and result["severity"] == "low":
                        result["severity"] = "medium"
            
            if category_triggered:
                break  # Stop at first category match
        
        result["confidence"] = min(1.0, violation_score)
        
        return result

class AccessController:
    """Advanced access control system"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.permissions = {}
        self.sessions = {}
        self.roles = self._initialize_roles()
    
    def _initialize_roles(self) -> Dict[str, List[str]]:
        """Initialize role-based permissions"""
        
        return {
            "guest": ["read_public"],
            "user": ["read_public", "create_content", "interact"],
            "trusted_user": ["read_public", "create_content", "interact", "moderate_content"],
            "moderator": ["read_public", "create_content", "interact", "moderate_content", "ban_users"],
            "admin": ["*"]  # All permissions
        }
    
    async def check_permission(self, user_id: str, resource: str, action: str = "access") -> Dict[str, Any]:
        """Advanced permission checking"""
        
        result = {
            "allowed": False,
            "reason": "permission_denied",
            "user_role": "guest",
            "required_permission": f"{action}_{resource}"
        }
        
        # Get user role
        user_role = self.permissions.get(user_id, {}).get("role", "guest")
        result["user_role"] = user_role
        
        # Check permissions
        role_permissions = self.roles.get(user_role, [])
        required_permission = f"{action}_{resource}"
        
        if "*" in role_permissions or required_permission in role_permissions:
            result["allowed"] = True
            result["reason"] = "permission_granted"
        
        return result

class RateLimiter:
    """Advanced rate limiting system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.request_counts = {}
        self.penalty_box = {}  # For temporary bans
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def check_rate_limit(self, source_ip: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Advanced rate limiting with adaptive thresholds"""
        
        identifier = user_id or source_ip
        current_time = time.time()
        
        # Check penalty box
        if identifier in self.penalty_box:
            penalty_until = self.penalty_box[identifier]
            if current_time < penalty_until:
                return {
                    "allowed": False,
                    "reason": "penalty_box",
                    "retry_after": penalty_until - current_time
                }
            else:
                del self.penalty_box[identifier]
        
        # Initialize tracking
        if identifier not in self.request_counts:
            self.request_counts[identifier] = {
                "minute": [],
                "hour": [],
                "day": []
            }
        
        counts = self.request_counts[identifier]
        
        # Clean old entries
        counts["minute"] = [t for t in counts["minute"] if current_time - t < 60]
        counts["hour"] = [t for t in counts["hour"] if current_time - t < 3600]
        counts["day"] = [t for t in counts["day"] if current_time - t < 86400]
        
        # Check limits
        minute_limit = self.config["max_requests_per_minute"]
        hour_limit = self.config["max_requests_per_hour"]
        day_limit = self.config.get("max_requests_per_day", 10000)
        
        if len(counts["minute"]) >= minute_limit:
            # Add to penalty box for 1 minute
            self.penalty_box[identifier] = current_time + 60
            return {"allowed": False, "reason": "minute_limit_exceeded", "retry_after": 60}
        
        if len(counts["hour"]) >= hour_limit:
            # Add to penalty box for 1 hour
            self.penalty_box[identifier] = current_time + 3600
            return {"allowed": False, "reason": "hour_limit_exceeded", "retry_after": 3600}
        
        if len(counts["day"]) >= day_limit:
            # Add to penalty box for 1 day
            self.penalty_box[identifier] = current_time + 86400
            return {"allowed": False, "reason": "day_limit_exceeded", "retry_after": 86400}
        
        # Add current request
        counts["minute"].append(current_time)
        counts["hour"].append(current_time)
        counts["day"].append(current_time)
        
        return {
            "allowed": True,
            "remaining": {
                "minute": minute_limit - len(counts["minute"]),
                "hour": hour_limit - len(counts["hour"]),
                "day": day_limit - len(counts["day"])
            }
        }

# Global protection layer instance
_protection_layer = None

def get_protection_layer() -> ProtectionLayer:
    """Get global protection layer instance"""
    global _protection_layer
    if _protection_layer is None:
        _protection_layer = ProtectionLayer()
    return _protection_layer

# Export main classes
__all__ = [
    "ProtectionLayer",
    "get_protection_layer",
    "SecurityIncident",
    "ProtectionRule",
    "ThreatLevel",
    "ProtectionType",
    "SecurityEvent"
]

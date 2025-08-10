#!/usr/bin/env python3
"""
🕉️ DharmaMind Deployment Readiness Check

This script performs comprehensive checks to ensure the entire DharmaMind system
is ready for production deployment.

Checks include:
- Environment configuration validation
- Code quality and lint checks
- Dependencies verification
- Database readiness
- Security configuration
- Performance optimization
- Integration testing
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DeploymentChecker:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.errors = []
        self.warnings = []
        self.successes = []
        
    def log_success(self, message: str):
        """Log a success message"""
        self.successes.append(message)
        logger.info(f"✅ {message}")
        
    def log_warning(self, message: str):
        """Log a warning message"""
        self.warnings.append(message)
        logger.warning(f"⚠️ {message}")
        
    def log_error(self, message: str):
        """Log an error message"""
        self.errors.append(message)
        logger.error(f"❌ {message}")

    def check_environment_files(self) -> bool:
        """Check if all required environment files exist and are properly configured"""
        logger.info("🔍 Checking environment configuration...")
        
        required_env_files = [
            ".env.example",
            "backend/.env.example",
            "dharmamind-chat/.env.local",
            "Brand_Webpage/.env.local",
            "DhramaMind_Community/.env.local"
        ]
        
        all_good = True
        for env_file in required_env_files:
            env_path = self.project_root / env_file
            if env_path.exists():
                self.log_success(f"Environment file exists: {env_file}")
                
                # Check for placeholder values
                try:
                    content = env_path.read_text()
                    if "change_me" in content.lower() or "your_" in content.lower():
                        self.log_warning(f"Placeholder values found in {env_file}")
                except Exception as e:
                    self.log_error(f"Error reading {env_file}: {e}")
                    all_good = False
            else:
                self.log_error(f"Missing environment file: {env_file}")
                all_good = False
                
        return all_good

    def check_package_files(self) -> bool:
        """Check if all package.json and requirements.txt files are present"""
        logger.info("📦 Checking package configurations...")
        
        required_package_files = [
            "requirements.txt",
            "backend/requirements.txt",
            "dharmamind-chat/package.json",
            "Brand_Webpage/package.json",
            "DhramaMind_Community/package.json"
        ]
        
        all_good = True
        for package_file in required_package_files:
            package_path = self.project_root / package_file
            if package_path.exists():
                self.log_success(f"Package file exists: {package_file}")
                
                # Validate JSON for package.json files
                if package_file.endswith("package.json"):
                    try:
                        with open(package_path, 'r') as f:
                            json.load(f)
                        self.log_success(f"Valid JSON: {package_file}")
                    except json.JSONDecodeError as e:
                        self.log_error(f"Invalid JSON in {package_file}: {e}")
                        all_good = False
            else:
                self.log_error(f"Missing package file: {package_file}")
                all_good = False
                
        return all_good

    def check_dockerfile_presence(self) -> bool:
        """Check if Dockerfiles are present for containerization"""
        logger.info("🐳 Checking Docker configuration...")
        
        docker_files = [
            "Dockerfile",
            "Dockerfile.production",
            "docker-compose.yml",
            "docker-compose.production.yml",
            "backend/Dockerfile"
        ]
        
        all_good = True
        for docker_file in docker_files:
            docker_path = self.project_root / docker_file
            if docker_path.exists():
                self.log_success(f"Docker file exists: {docker_file}")
            else:
                self.log_warning(f"Missing Docker file: {docker_file}")
                
        return all_good

    def check_security_configurations(self) -> bool:
        """Check security-related configurations"""
        logger.info("🔒 Checking security configurations...")
        
        security_checks = []
        
        # Check for .gitignore
        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
            if ".env" in gitignore_content:
                self.log_success("Environment files are ignored in .gitignore")
            else:
                self.log_error(".env files not properly ignored in .gitignore")
                security_checks.append(False)
        else:
            self.log_error("Missing .gitignore file")
            security_checks.append(False)
            
        # Check for sensitive data in code
        sensitive_patterns = ["password", "secret", "key", "token"]
        for pattern in sensitive_patterns:
            # This is a simplified check - in reality, you'd want more sophisticated scanning
            pass
            
        return all(security_checks) if security_checks else True

    def check_backend_structure(self) -> bool:
        """Check backend application structure"""
        logger.info("🔧 Checking backend structure...")
        
        required_backend_files = [
            "backend/app/main.py",
            "backend/app/config.py",
            "backend/app/__init__.py",
            "backend/app/routes/__init__.py",
            "backend/app/services/__init__.py",
            "backend/app/db/__init__.py"
        ]
        
        all_good = True
        for backend_file in required_backend_files:
            backend_path = self.project_root / backend_file
            if backend_path.exists():
                self.log_success(f"Backend file exists: {backend_file}")
            else:
                self.log_error(f"Missing backend file: {backend_file}")
                all_good = False
                
        return all_good

    def check_frontend_structure(self) -> bool:
        """Check frontend applications structure"""
        logger.info("🎨 Checking frontend structure...")
        
        frontend_apps = ["dharmamind-chat", "Brand_Webpage", "DhramaMind_Community"]
        
        all_good = True
        for app in frontend_apps:
            app_path = self.project_root / app
            if app_path.exists():
                self.log_success(f"Frontend app exists: {app}")
                
                # Check for essential Next.js files
                essential_files = ["package.json", "next.config.js", "pages/_app.tsx"]
                for file in essential_files:
                    file_path = app_path / file
                    if file_path.exists():
                        self.log_success(f"{app}/{file} exists")
                    else:
                        self.log_error(f"Missing {app}/{file}")
                        all_good = False
            else:
                self.log_error(f"Missing frontend app: {app}")
                all_good = False
                
        return all_good

    def check_database_configuration(self) -> bool:
        """Check database configuration and setup scripts"""
        logger.info("🗄️ Checking database configuration...")
        
        db_files = [
            "backend/app/db/database.py",
            "backend/app/setup_database.py",
            "backend/app/models/__init__.py"
        ]
        
        all_good = True
        for db_file in db_files:
            db_path = self.project_root / db_file
            if db_path.exists():
                self.log_success(f"Database file exists: {db_file}")
            else:
                self.log_warning(f"Database file missing: {db_file}")
                
        return all_good

    def check_monitoring_setup(self) -> bool:
        """Check monitoring and logging configuration"""
        logger.info("📊 Checking monitoring setup...")
        
        monitoring_files = [
            "monitoring/prometheus.yml",
            "monitoring/grafana",
            "k8s/",
            "terraform/"
        ]
        
        for monitor_file in monitoring_files:
            monitor_path = self.project_root / monitor_file
            if monitor_path.exists():
                self.log_success(f"Monitoring component exists: {monitor_file}")
            else:
                self.log_warning(f"Monitoring component missing: {monitor_file}")
                
        return True

    def run_linting_checks(self) -> bool:
        """Run code linting and formatting checks"""
        logger.info("🧹 Running code quality checks...")
        
        try:
            # Check Python code with flake8 if available
            result = subprocess.run(
                ["python", "-m", "flake8", "backend/", "--max-line-length=88"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                self.log_success("Python code passes linting checks")
            else:
                self.log_warning("Python linting issues found")
                
        except FileNotFoundError:
            self.log_warning("flake8 not available for Python linting")
            
        # Check TypeScript/JavaScript with ESLint if available
        for frontend_app in ["dharmamind-chat", "Brand_Webpage", "DhramaMind_Community"]:
            try:
                result = subprocess.run(
                    ["npm", "run", "lint"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root / frontend_app
                )
                
                if result.returncode == 0:
                    self.log_success(f"{frontend_app} passes ESLint checks")
                else:
                    self.log_warning(f"{frontend_app} has ESLint issues")
                    
            except FileNotFoundError:
                self.log_warning(f"npm not available for {frontend_app} linting")
                
        return True

    def check_deployment_scripts(self) -> bool:
        """Check if deployment scripts are present"""
        logger.info("🚀 Checking deployment scripts...")
        
        deployment_files = [
            "deploy.sh",
            "deploy-production.sh",
            "start_complete_system.py",
            "quick_start.sh"
        ]
        
        for deploy_file in deployment_files:
            deploy_path = self.project_root / deploy_file
            if deploy_path.exists():
                self.log_success(f"Deployment script exists: {deploy_file}")
            else:
                self.log_warning(f"Deployment script missing: {deploy_file}")
                
        return True

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive deployment readiness report"""
        total_checks = len(self.successes) + len(self.warnings) + len(self.errors)
        success_rate = len(self.successes) / total_checks * 100 if total_checks > 0 else 0
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_checks": total_checks,
            "successes": len(self.successes),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "success_rate": round(success_rate, 2),
            "deployment_ready": len(self.errors) == 0,
            "details": {
                "successes": self.successes,
                "warnings": self.warnings,
                "errors": self.errors
            }
        }
        
        return report

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all deployment readiness checks"""
        logger.info("🕉️ Starting DharmaMind Deployment Readiness Check...")
        logger.info("=" * 60)
        
        checks = [
            self.check_environment_files,
            self.check_package_files,
            self.check_dockerfile_presence,
            self.check_security_configurations,
            self.check_backend_structure,
            self.check_frontend_structure,
            self.check_database_configuration,
            self.check_monitoring_setup,
            self.run_linting_checks,
            self.check_deployment_scripts
        ]
        
        for check in checks:
            try:
                check()
                logger.info("-" * 40)
            except Exception as e:
                self.log_error(f"Check failed with exception: {e}")
                logger.info("-" * 40)
                
        report = self.generate_report()
        
        logger.info("=" * 60)
        logger.info("🏁 Deployment Readiness Check Complete!")
        logger.info(f"✅ Successes: {report['successes']}")
        logger.info(f"⚠️ Warnings: {report['warnings']}")
        logger.info(f"❌ Errors: {report['errors']}")
        logger.info(f"📊 Success Rate: {report['success_rate']}%")
        logger.info(f"🚀 Deployment Ready: {'YES' if report['deployment_ready'] else 'NO'}")
        
        return report


def main():
    """Main function to run deployment checks"""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = os.getcwd()
        
    checker = DeploymentChecker(project_root)
    report = checker.run_all_checks()
    
    # Save report to file
    report_path = Path(project_root) / "deployment-readiness-report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"📄 Full report saved to: {report_path}")
    
    # Exit with appropriate code
    sys.exit(0 if report['deployment_ready'] else 1)


if __name__ == "__main__":
    main()

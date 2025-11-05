"""
Integration Service Controller 🔗
Manages communication between DevOps Interface and Agent Orchestrator
"""

import os
import sys
import json
import time
import threading
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess
from dataclasses import dataclass

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class ServiceStatus:
    """Service status information"""
    name: str
    status: str  # running, stopped, error
    url: str
    last_check: datetime
    error_message: Optional[str] = None

class IntegrationServiceController:
    """
    Controls and monitors both DevOps Interface and Agent Orchestrator services
    Ensures they work together seamlessly
    """
    
    def __init__(self):
        self.devops_interface_url = "http://localhost:5000"
        self.orchestrator_api_url = "http://localhost:8001"
        self.services = {
            'devops_interface': ServiceStatus(
                name='DevOps Interface',
                status='stopped',
                url=self.devops_interface_url,
                last_check=datetime.utcnow()
            ),
            'orchestrator_api': ServiceStatus(
                name='Agent Orchestrator API',
                status='stopped', 
                url=self.orchestrator_api_url,
                last_check=datetime.utcnow()
            )
        }
        
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        try:
            service = self.services.get(service_name)
            if not service:
                return False
            
            if service_name == 'devops_interface':
                # Check DevOps interface health
                response = requests.get(f"{service.url}/api/orchestrator/health", timeout=10)
                healthy = response.status_code == 200
            
            elif service_name == 'orchestrator_api':
                # Check orchestrator API health
                response = requests.get(f"{service.url}/api/orchestrator/health", timeout=10)
                result = response.json()
                healthy = result.get('status') == 'healthy'
            
            else:
                return False
            
            # Update service status
            service.status = 'running' if healthy else 'error'
            service.last_check = datetime.utcnow()
            service.error_message = None if healthy else 'Health check failed'
            
            return healthy
            
        except requests.exceptions.RequestException as e:
            service = self.services.get(service_name)
            if service:
                service.status = 'error'
                service.last_check = datetime.utcnow()
                service.error_message = f"Connection error: {str(e)}"
            return False
        
        except Exception as e:
            service = self.services.get(service_name)
            if service:
                service.status = 'error'
                service.last_check = datetime.utcnow()
                service.error_message = f"Health check error: {str(e)}"
            return False
    
    def start_orchestrator_api(self) -> bool:
        """Start the orchestrator API service"""
        try:
            print("🚀 Starting Agent Orchestrator API...")
            
            # Check if already running
            if self.check_service_health('orchestrator_api'):
                print("✅ Orchestrator API is already running")
                return True
            
            # Start the orchestrator API
            orchestrator_script = os.path.join(
                os.path.dirname(__file__), 
                'orchestrator_api.py'
            )
            
            if not os.path.exists(orchestrator_script):
                print(f"❌ Orchestrator API script not found: {orchestrator_script}")
                return False
            
            # Start in background
            def run_orchestrator():
                try:
                    subprocess.run([
                        sys.executable, 
                        orchestrator_script
                    ], cwd=os.path.dirname(orchestrator_script))
                except Exception as e:
                    print(f"❌ Error running orchestrator API: {e}")
            
            thread = threading.Thread(target=run_orchestrator)
            thread.daemon = True
            thread.start()
            
            # Wait for service to start
            max_wait = 30  # seconds
            wait_interval = 2
            waited = 0
            
            while waited < max_wait:
                time.sleep(wait_interval)
                waited += wait_interval
                
                if self.check_service_health('orchestrator_api'):
                    print("✅ Orchestrator API started successfully")
                    return True
                
                print(f"⏳ Waiting for orchestrator API to start... ({waited}s)")
            
            print("❌ Orchestrator API failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"❌ Error starting orchestrator API: {e}")
            return False
    
    def start_devops_interface(self) -> bool:
        """Start the DevOps Interface service"""
        try:
            print("🚀 Starting DevOps Interface...")
            
            # Check if already running
            if self.check_service_health('devops_interface'):
                print("✅ DevOps Interface is already running")
                return True
            
            # Start the DevOps interface
            devops_script = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'devops_interface', 
                'app.py'
            )
            
            if not os.path.exists(devops_script):
                print(f"❌ DevOps Interface script not found: {devops_script}")
                return False
            
            # Start in background
            def run_devops():
                try:
                    subprocess.run([
                        sys.executable, 
                        devops_script
                    ], cwd=os.path.dirname(devops_script))
                except Exception as e:
                    print(f"❌ Error running DevOps Interface: {e}")
            
            thread = threading.Thread(target=run_devops)
            thread.daemon = True
            thread.start()
            
            # Wait for service to start
            max_wait = 30  # seconds
            wait_interval = 2
            waited = 0
            
            while waited < max_wait:
                time.sleep(wait_interval)
                waited += wait_interval
                
                if self.check_service_health('devops_interface'):
                    print("✅ DevOps Interface started successfully")
                    return True
                
                print(f"⏳ Waiting for DevOps Interface to start... ({waited}s)")
            
            print("❌ DevOps Interface failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"❌ Error starting DevOps Interface: {e}")
            return False
    
    def start_all_services(self) -> bool:
        """Start both services in the correct order"""
        print("🔧 Starting integrated ETL system...")
        
        # Start orchestrator API first
        if not self.start_orchestrator_api():
            print("❌ Failed to start Orchestrator API")
            return False
        
        # Wait a bit for orchestrator to fully initialize
        time.sleep(5)
        
        # Start DevOps interface
        if not self.start_devops_interface():
            print("❌ Failed to start DevOps Interface")
            return False
        
        # Start monitoring
        self.start_monitoring()
        
        print("✅ All services started successfully!")
        print("📊 DevOps Interface: http://localhost:5000")
        print("📡 Orchestrator API: http://localhost:8001/api/orchestrator/health")
        
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # Check all services
        for service_name in self.services:
            self.check_service_health(service_name)
        
        status = {
            'overall_status': 'healthy' if all(s.status == 'running' for s in self.services.values()) else 'degraded',
            'services': {},
            'integration_status': 'unknown',
            'last_updated': datetime.utcnow().isoformat()
        }
        
        for name, service in self.services.items():
            status['services'][name] = {
                'name': service.name,
                'status': service.status,
                'url': service.url,
                'last_check': service.last_check.isoformat(),
                'error_message': service.error_message
            }
        
        # Test integration
        try:
            if self.services['devops_interface'].status == 'running':
                response = requests.get(f"{self.devops_interface_url}/api/orchestrator/health", timeout=5)
                if response.status_code == 200:
                    status['integration_status'] = 'connected'
                else:
                    status['integration_status'] = 'connection_error'
            else:
                status['integration_status'] = 'devops_interface_down'
        except:
            status['integration_status'] = 'integration_error'
        
        return status
    
    def start_monitoring(self):
        """Start continuous monitoring of services"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_services():
            while self.monitoring_active:
                try:
                    # Check service health every 30 seconds
                    for service_name in self.services:
                        self.check_service_health(service_name)
                    
                    time.sleep(30)
                    
                except Exception as e:
                    print(f"⚠️ Monitoring error: {e}")
                    time.sleep(10)
        
        self.monitoring_thread = threading.Thread(target=monitor_services)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        print("📊 Service monitoring started")
    
    def stop_monitoring(self):
        """Stop service monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("⏹️ Service monitoring stopped")
    
    def test_integration(self) -> Dict[str, Any]:
        """Test the integration between services"""
        print("🧪 Testing service integration...")
        
        test_results = {
            'orchestrator_health': False,
            'devops_interface_health': False,
            'integration_working': False,
            'test_timestamp': datetime.utcnow().isoformat(),
            'errors': []
        }
        
        try:
            # Test orchestrator API directly
            response = requests.get(f"{self.orchestrator_api_url}/api/orchestrator/health", timeout=10)
            if response.status_code == 200:
                result = response.json()
                test_results['orchestrator_health'] = result.get('status') == 'healthy'
            
            # Test DevOps interface
            response = requests.get(f"{self.devops_interface_url}/", timeout=10)
            test_results['devops_interface_health'] = response.status_code == 200
            
            # Test integration via DevOps interface
            response = requests.get(f"{self.devops_interface_url}/api/orchestrator/health", timeout=10)
            if response.status_code == 200:
                result = response.json()
                test_results['integration_working'] = result.get('status') == 'healthy'
            
        except Exception as e:
            test_results['errors'].append(str(e))
        
        # Print results
        print("📋 Integration Test Results:")
        print(f"  🤖 Orchestrator API Health: {'✅' if test_results['orchestrator_health'] else '❌'}")
        print(f"  🏗️ DevOps Interface Health: {'✅' if test_results['devops_interface_health'] else '❌'}")
        print(f"  🔗 Integration Working: {'✅' if test_results['integration_working'] else '❌'}")
        
        if test_results['errors']:
            print("  ❌ Errors:")
            for error in test_results['errors']:
                print(f"    - {error}")
        
        return test_results
    
    def restart_services(self) -> bool:
        """Restart all services"""
        print("🔄 Restarting all services...")
        
        # Note: In a production environment, you'd implement proper service stopping
        # For development, we'll just try to start services (they should handle duplicates)
        
        return self.start_all_services()


def main():
    """Main integration service entry point"""
    controller = IntegrationServiceController()
    
    try:
        # Start all services
        if controller.start_all_services():
            print("\n🎉 ETL System Integration successful!")
            print("\n📖 Usage:")
            print("  1. Open http://localhost:5000 for DevOps Interface")
            print("  2. Create a project and user story")
            print("  3. Click 'Process with ETL Agent' to use the orchestrator")
            print("  4. Check http://localhost:8001/api/orchestrator/health for API status")
            
            # Run integration test
            print("\n🧪 Running integration test...")
            test_results = controller.test_integration()
            
            if test_results['integration_working']:
                print("\n✅ Integration test passed! System is ready.")
            else:
                print("\n⚠️ Integration test had issues. Check service logs.")
            
            # Keep services running
            print("\n⏳ Services are running. Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(60)
                    status = controller.get_system_status()
                    if status['overall_status'] != 'healthy':
                        print(f"⚠️ System status: {status['overall_status']}")
            except KeyboardInterrupt:
                print("\n🛑 Stopping services...")
                controller.stop_monitoring()
                
        else:
            print("❌ Failed to start services")
            return False
            
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()
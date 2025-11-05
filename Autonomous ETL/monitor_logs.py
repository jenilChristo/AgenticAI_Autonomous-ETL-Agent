#!/usr/bin/env python3
"""
Real-time log monitor for orchestrator and agent logs
"""
import asyncio
import aiohttp
import time
import json
from datetime import datetime

class LogMonitor:
    def __init__(self):
        self.orchestrator_url = "http://127.0.0.1:8001"
        self.devops_url = "http://127.0.0.1:5000"
        
    def log_with_timestamp(self, service, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "INFO": "\033[94m",     # Blue
            "ERROR": "\033[91m",    # Red  
            "SUCCESS": "\033[92m",  # Green
            "WARNING": "\033[93m",  # Yellow
            "RESET": "\033[0m"      # Reset
        }
        
        color = colors.get(level, colors["INFO"])
        reset = colors["RESET"]
        
        print(f"{color}[{timestamp}] {service:12} | {message}{reset}")
        
    async def test_orchestrator_health(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.orchestrator_url}/api/orchestrator/health", 
                                     timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.log_with_timestamp("ORCHESTRATOR", f"✅ Health OK - {data.get('status', 'unknown')}", "SUCCESS")
                        return True
                    else:
                        self.log_with_timestamp("ORCHESTRATOR", f"❌ Health check failed: {response.status}", "ERROR")
                        return False
        except Exception as e:
            self.log_with_timestamp("ORCHESTRATOR", f"❌ Connection failed: {str(e)}", "ERROR")
            return False
    
    async def test_devops_health(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.devops_url}/", 
                                     timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status == 200:
                        self.log_with_timestamp("DEVOPS UI", "✅ Interface responding", "SUCCESS")
                        return True
                    else:
                        self.log_with_timestamp("DEVOPS UI", f"❌ Interface error: {response.status}", "ERROR")
                        return False
        except Exception as e:
            self.log_with_timestamp("DEVOPS UI", f"❌ Connection failed: {str(e)}", "ERROR")
            return False
    
    async def get_processing_status(self, story_id=2):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.devops_url}/api/story/{story_id}/status", 
                                     timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get('orchestrator_status', 'unknown')
                        step = data.get('current_step', 'N/A')
                        
                        if status == 'processing':
                            self.log_with_timestamp("PROCESSING", f"🔄 Step: {step}", "WARNING")
                        elif status == 'completed':
                            self.log_with_timestamp("PROCESSING", f"✅ Completed!", "SUCCESS")
                        elif status == 'error':
                            error_msg = data.get('error_message', 'Unknown error')
                            self.log_with_timestamp("PROCESSING", f"❌ Error: {error_msg}", "ERROR")
                        else:
                            self.log_with_timestamp("PROCESSING", f"⏳ Status: {status}", "INFO")
                        return True
                    else:
                        return False
        except Exception as e:
            return False
    
    async def trigger_processing(self, story_id=2):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.devops_url}/process_with_etl/{story_id}",
                                      timeout=aiohttp.ClientTimeout(total=10),
                                      allow_redirects=False) as response:
                    self.log_with_timestamp("TRIGGER", f"🚀 Processing triggered for story {story_id}: {response.status}", "INFO")
                    return True
        except Exception as e:
            self.log_with_timestamp("TRIGGER", f"❌ Failed to trigger: {str(e)}", "ERROR")
            return False

    async def monitor_loop(self):
        print("🔍 Starting Real-Time Log Monitor...")
        print("=" * 60)
        
        # Initial health checks
        await self.test_orchestrator_health()
        await self.test_devops_health()
        
        print("\n📊 Monitoring logs (Press Ctrl+C to stop)...")
        print("-" * 60)
        
        last_health_check = 0
        monitoring_processing = False
        
        while True:
            try:
                current_time = time.time()
                
                # Health check every 30 seconds
                if current_time - last_health_check > 30:
                    await self.test_orchestrator_health()
                    await self.test_devops_health()
                    last_health_check = current_time
                
                # Check processing status every 2 seconds if not processing
                await self.get_processing_status()
                
                await asyncio.sleep(2)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                self.log_with_timestamp("MONITOR", f"❌ Monitor error: {str(e)}", "ERROR")
                await asyncio.sleep(5)

async def main():
    monitor = LogMonitor()
    
    # Show menu
    print("🎛️  ETL System Log Monitor")
    print("=" * 40)
    print("1. Monitor logs continuously")
    print("2. Trigger processing for story #2")
    print("3. Health check only")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        await monitor.monitor_loop()
    elif choice == "2":
        await monitor.trigger_processing()
        await monitor.monitor_loop()
    elif choice == "3":
        await monitor.test_orchestrator_health()
        await monitor.test_devops_health()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    asyncio.run(main())
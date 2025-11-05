"""
Database cleanup script for DevOps Interface
Clears all records from the SQLite database tables
"""

import os
import sys
import sqlite3
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devops_interface.app import db, Project, UserStory, Task, ETLAgentExecution, app

def clean_database():
    """Clean all records from the DevOps interface database"""
    
    print("🧹 Starting database cleanup...")
    
    with app.app_context():
        try:
            # Get record counts before cleanup
            projects_count = Project.query.count()
            stories_count = UserStory.query.count()
            tasks_count = Task.query.count()
            executions_count = ETLAgentExecution.query.count()
            
            print(f"📊 Records before cleanup:")
            print(f"   - Projects: {projects_count}")
            print(f"   - User Stories: {stories_count}")
            print(f"   - Tasks: {tasks_count}")
            print(f"   - ETL Executions: {executions_count}")
            
            # Delete all records (foreign key constraints will handle cascading)
            print("\n🗑️ Deleting records...")
            
            # Delete in reverse order of dependencies
            ETLAgentExecution.query.delete()
            print("   ✅ ETL executions deleted")
            
            Task.query.delete()
            print("   ✅ Tasks deleted")
            
            UserStory.query.delete()
            print("   ✅ User stories deleted")
            
            Project.query.delete()
            print("   ✅ Projects deleted")
            
            # Commit all changes
            db.session.commit()
            
            print("\n✅ Database cleanup completed successfully!")
            print(f"🕐 Cleanup timestamp: {datetime.now()}")
            
            # Verify cleanup
            projects_after = Project.query.count()
            stories_after = UserStory.query.count()
            tasks_after = Task.query.count()
            executions_after = ETLAgentExecution.query.count()
            
            print(f"\n📊 Records after cleanup:")
            print(f"   - Projects: {projects_after}")
            print(f"   - User Stories: {stories_after}")
            print(f"   - Tasks: {tasks_after}")
            print(f"   - ETL Executions: {executions_after}")
            
            if projects_after == 0 and stories_after == 0 and tasks_after == 0 and executions_after == 0:
                print("\n🎉 All records successfully cleared!")
            else:
                print("\n⚠️ Some records may still exist")
                
        except Exception as e:
            print(f"❌ Error during cleanup: {str(e)}")
            db.session.rollback()
            raise
            
def reset_auto_increment():
    """Reset auto-increment counters for SQLite"""
    
    print("\n🔄 Resetting auto-increment counters...")
    
    try:
        # Get direct SQLite connection
        db_path = os.path.join(os.path.dirname(__file__), 'devops_interface', 'devops_interface.db')
        
        # Check if database exists
        if not os.path.exists(db_path):
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'devops_interface', 'devops_interface.db')
            
        if not os.path.exists(db_path):
            print(f"📁 Database file not found at expected paths")
            return
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Reset sqlite_sequence table
        cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('project', 'user_story', 'task', 'etl_agent_execution')")
        
        conn.commit()
        conn.close()
        
        print("   ✅ Auto-increment counters reset")
        
    except Exception as e:
        print(f"   ⚠️ Could not reset auto-increment: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧹 DevOps Interface Database Cleanup")
    print("=" * 60)
    
    # Confirm cleanup
    response = input("\n⚠️ This will delete ALL records from the database. Continue? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        clean_database()
        reset_auto_increment()
        print("\n🎯 Database cleanup completed!")
    else:
        print("\n🚫 Cleanup cancelled by user")
        
    print("\n" + "=" * 60)
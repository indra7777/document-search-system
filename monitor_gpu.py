#!/usr/bin/env python3
"""
GPU Memory Monitor for Document Search System
"""
import time
import torch
import psutil
import os

def get_gpu_memory():
    """Get GPU memory information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "total": total,
            "utilization": (allocated / total) * 100
        }
    return None

def get_system_memory():
    """Get system memory information"""
    memory = psutil.virtual_memory()
    return {
        "used": memory.used / 1024**3,
        "total": memory.total / 1024**3,
        "percent": memory.percent
    }

def monitor_memory(interval=5):
    """Monitor memory usage continuously"""
    print("🔍 GPU Memory Monitor Started")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("=" * 60)
            print("📊 DOCUMENT SEARCH SYSTEM - MEMORY MONITOR")
            print("=" * 60)
            
            # GPU Memory
            gpu_mem = get_gpu_memory()
            if gpu_mem:
                print(f"🎮 GPU Memory:")
                print(f"   Allocated: {gpu_mem['allocated']:.2f} GB ({gpu_mem['utilization']:.1f}%)")
                print(f"   Reserved:  {gpu_mem['reserved']:.2f} GB")
                print(f"   Free:      {gpu_mem['free']:.2f} GB")
                print(f"   Total:     {gpu_mem['total']:.2f} GB")
                
                # Warning for high usage
                if gpu_mem['utilization'] > 80:
                    print("   ⚠️  HIGH GPU MEMORY USAGE!")
                elif gpu_mem['utilization'] > 60:
                    print("   ⚡ Moderate GPU memory usage")
                else:
                    print("   ✅ Normal GPU memory usage")
            else:
                print("🎮 GPU: Not available")
            
            print()
            
            # System Memory
            sys_mem = get_system_memory()
            print(f"💻 System Memory:")
            print(f"   Used:  {sys_mem['used']:.2f} GB ({sys_mem['percent']:.1f}%)")
            print(f"   Total: {sys_mem['total']:.2f} GB")
            
            if sys_mem['percent'] > 80:
                print("   ⚠️  HIGH SYSTEM MEMORY USAGE!")
            elif sys_mem['percent'] > 60:
                print("   ⚡ Moderate system memory usage")
            else:
                print("   ✅ Normal system memory usage")
            
            print()
            
            # Process info
            streamlit_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    if 'streamlit' in proc.info['name'].lower():
                        memory_mb = proc.info['memory_info'].rss / 1024**2
                        streamlit_processes.append((proc.info['pid'], memory_mb))
                except:
                    continue
            
            if streamlit_processes:
                print("🔄 Streamlit Processes:")
                for pid, memory_mb in streamlit_processes:
                    print(f"   PID {pid}: {memory_mb:.1f} MB")
            else:
                print("🔄 Streamlit: Not running")
            
            print(f"\n⏰ Last updated: {time.strftime('%H:%M:%S')}")
            print("Press Ctrl+C to stop monitoring")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n👋 Memory monitoring stopped")

if __name__ == "__main__":
    monitor_memory()
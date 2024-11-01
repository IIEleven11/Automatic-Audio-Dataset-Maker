# Sometimes GPU VRAM gets stuck. Use this to clear it.


import torch
import gc
import os
import subprocess
import psutil
import signal

def get_gpu_processes():
    try:
        # Run nvidia-smi command to get process information
        cmd = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split()).decode()
        processes = []
        
        for line in output.strip().split('\n'):
            if line:
                pid, memory = line.split(',')
                pid = int(pid)
                memory = int(memory)
                try:
                    process = psutil.Process(pid)
                    processes.append({
                        'pid': pid,
                        'name': process.name(),
                        'memory': memory,
                        'cmdline': ' '.join(process.cmdline())
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        return processes
    except subprocess.CalledProcessError:
        return []


def clear_gpu_memory(kill_processes=True):
    try:
        print("\nCurrent GPU Memory Usage:")
        os.system('nvidia-smi')

        gpu_processes = get_gpu_processes()
        if gpu_processes:
            print("\nProcesses using GPU:")
            for proc in gpu_processes:
                print(f"PID: {proc['pid']}, Name: {proc['name']}, Memory: {proc['memory']}MB")
                print(f"Command: {proc['cmdline']}")
            
            if kill_processes:
                print("\nTerminating GPU processes...")
                for proc in gpu_processes:
                    try:
                        os.kill(proc['pid'], signal.SIGTERM)
                        print(f"Terminated process {proc['pid']}")
                    except ProcessLookupError:
                        continue
                    except PermissionError:
                        print(f"Permission denied to kill process {proc['pid']}")

        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            print(f"\nGPU Memory before clearing cache: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # Delete all cached tensors
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj):
                        del obj
                except Exception:
                    pass
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"GPU Memory after clearing cache: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # Show final GPU status
            print("\nFinal GPU Status:")
            os.system('nvidia-smi')
            
            return True
    except Exception as e:
        print(f"Error clearing GPU memory: {str(e)}")
        return False

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} device(s)")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        success = clear_gpu_memory(kill_processes=True)
        if success:
            print("\nGPU memory cleared successfully")
        else:
            print("\nFailed to clear GPU memory")
    else:
        print("CUDA is not available")